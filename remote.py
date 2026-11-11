#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Copyright contributors to the tango-ucx project
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Reach cluster nodes over SSH with a password: one prompt a run, one login a host.

SSH keys cannot be used on this cluster. The password is asked for once and kept in this
process's memory; OpenSSH gets it from a temporary askpass helper over a private Unix socket,
so it is never in a file, an argument, an environment variable or a result. Each host's login
is an OpenSSH control master. Commands then go through it with BatchMode, so a lost login
fails the command rather than logging in again.

    remote.py check --hosts node1,node2              log in to each and print its name
    remote.py on --hosts node1,node2 -- 'df -h /tmp'  run a shell command on each, in turn
    remote.py relay --host node1 -- 'COMMAND'        run COMMAND on node1, where it may log in to
                                                     other hosts with this password, not asked again

A relay gives the one host a private socket, forwarded back here for that connection only and
removed when the command ends: logins made there ask this process for the password. Its root
could use the socket while the command runs, as it could see a password typed there.
    remote.py run --hosts node1,node2 -- CMD ARGS    run CMD here with those logins open;
                                                     CMD reaches the hosts with "$REMOTE_SSH"
"""
import argparse
from contextlib import ExitStack, contextmanager
import getpass
import math
import os
from pathlib import Path
import secrets
import shlex
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import warnings

PASSWORD_SOCKET = 'REMOTE_PASSWORD_SOCKET'
# Set by a relay on the host it runs a command on: the socket there leads to the password.
RELAYED = 'REMOTE_PASSWORD_RELAYED'
# sshd's default MaxSessions: the commands one login carries at once.
SESSIONS_PER_LOGIN = 10
# A login left behind by a killed run ends after this long unused.
LOGIN_IDLE_LIMIT = '30m'
# What a host without a home directory says at every login: no news.
NOISE = 'Could not chdir to home directory'


def meaningful(text):
    """The lines of what a host said, without the noise of its logins."""
    return [line for line in text.splitlines() if line.strip() and NOISE not in line]


def askpass(prompt):
    # Reuse only password responses. OTP/MFA challenges still need their own answer.
    if 'password' not in prompt.lower():
        print(getpass.getpass(prompt))
        return
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(os.environ[PASSWORD_SOCKET])
        with client.makefile('r') as response:
            sys.stdout.write(response.read())


def private_directory(prefix):
    """A directory only this user can enter, with a path short enough for Unix sockets."""
    base = tempfile.gettempdir()
    return tempfile.TemporaryDirectory(prefix=prefix, dir=base if len(base) <= 40 else '/tmp')


def write_helper(directory):
    helper = directory / 'askpass'
    helper.write_text('#!/bin/sh\nexec ' + shlex.join(
        [sys.executable, str(Path(__file__).resolve()), '--askpass']) + ' "$@"\n')
    helper.chmod(0o700)
    return helper


@contextmanager
def password_environment(enabled, prompt='SSH password (reused for all selected nodes): '):
    """Serve one in-memory password to SSH_ASKPASS via a private local socket. Under a relay,
    the relay's socket serves it, and nothing is asked."""
    env = dict(os.environ)
    if not enabled:
        yield env
        return
    relayed = os.environ.get(PASSWORD_SOCKET) if os.environ.get(RELAYED) else None
    if relayed and Path(relayed).exists():
        with private_directory('remote-ssh-') as directory:
            env.update(SSH_ASKPASS=str(write_helper(Path(directory))), SSH_ASKPASS_REQUIRE='force',
                       DISPLAY=env.get('DISPLAY') or ':0', **{PASSWORD_SOCKET: relayed})
            yield env
        return
    try:
        with warnings.catch_warnings():  # without a terminal getpass warns, then reads stdin
            warnings.simplefilter('ignore', getpass.GetPassWarning)
            password = getpass.getpass(prompt)
    except EOFError:
        raise RuntimeError('no terminal here to ask for the SSH password: run this through '
                           '"remote.py relay" from where you can type it') from None
    with private_directory('remote-ssh-') as directory:
        directory = Path(directory)
        endpoint = directory / 'password.sock'
        helper = write_helper(directory)

        class PasswordHandler(socketserver.StreamRequestHandler):
            def handle(self):
                self.wfile.write((password + '\n').encode())

        # Threads: under a relay, several hosts' logins can ask at once.
        with socketserver.ThreadingUnixStreamServer(str(endpoint), PasswordHandler) as server:
            worker = threading.Thread(target=server.serve_forever,
                                      kwargs={'poll_interval': 0.1}, daemon=True)
            worker.start()
            env.update(SSH_ASKPASS=str(helper), SSH_ASKPASS_REQUIRE='force',
                       DISPLAY=env.get('DISPLAY') or ':0', **{PASSWORD_SOCKET: str(endpoint)})
            try:
                yield env
            finally:
                server.shutdown()
                worker.join()


def ssh_options(extra=(), batch=False):
    """This cluster's SSH settings. SSH takes an option's first value, so extra ones win."""
    options = []
    for option in extra:
        options += ['-o', option]
    options += ['-o', 'ClearAllForwardings=yes', '-o', 'ConnectTimeout=10',
                '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null',
                '-o', 'GlobalKnownHostsFile=/dev/null', '-o', 'LogLevel=ERROR']
    if batch:
        options += ['-o', 'BatchMode=yes']
    else:
        options += ['-o', 'BatchMode=no', '-o', 'PubkeyAuthentication=no',
                    '-o', 'PreferredAuthentications=password,keyboard-interactive',
                    '-o', 'NumberOfPasswordPrompts=1']
    return options


def add_arguments(parser, batch_flag='--batch'):
    """The options of a tool that reaches hosts through a Session. A tool whose --batch means
    something else names this one otherwise."""
    group = parser.add_argument_group('SSH (password authentication, one login per host)')
    group.add_argument('--user', help='SSH username (otherwise SSH config or the current user)')
    group.add_argument('--ssh-option', action='append', default=[], metavar='KEY=VALUE',
                       help='extra SSH -o option, e.g. ProxyJump=login-host; repeat as needed')
    group.add_argument('--password-per-node', action='store_true',
                       help='let SSH ask for each host\'s password instead of reusing one')
    group.add_argument(batch_flag, dest='ssh_batch', action='store_true',
                       help='no authentication prompts, for access that needs none')
    group.add_argument('--login-timeout', type=float, default=120,
                       help='seconds to log in to a host, authentication included (120)')
    return group


class Session:
    """The run's logins, each host's made once, and the commands that use them.

    Open each host before using it: a host needs one login for every SESSIONS_PER_LOGIN
    commands it runs at once. Extra logins reuse the password from memory, without a prompt.
    """

    def __init__(self, user=None, extra=(), batch=False, password_per_node=False,
                 timeout=120, hosts=None, sessions_per_login=SESSIONS_PER_LOGIN):
        self.user, self.extra, self.batch = user, list(extra), batch
        self.password_per_node, self.timeout = password_per_node, timeout
        self.sessions_per_login = sessions_per_login
        self.wanted = dict(hosts or {})
        self.logins = {}
        self.env = None
        self.directory = None
        self._stack = None

    @classmethod
    def from_args(cls, args, hosts=None):
        return cls(user=args.user, extra=args.ssh_option, batch=args.ssh_batch,
                   password_per_node=args.password_per_node, timeout=args.login_timeout,
                   hosts=hosts)

    def __enter__(self):
        self._stack = ExitStack()
        try:
            self.directory = Path(self._stack.enter_context(private_directory('ssh-')))
            for host, sessions in self.wanted.items():
                self.open(host, sessions)
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        for target, count in self.logins.items():
            for index in range(count):
                try:
                    subprocess.run(['ssh', *self._control(target, index), '-O', 'exit', target],
                                   stdin=subprocess.DEVNULL, capture_output=True, timeout=15)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        self.logins = {}
        if self._stack:
            self._stack.close()
            self._stack = None

    def target(self, host):
        return f'{self.user}@{host}' if self.user and '@' not in host else host

    def _control(self, target, index):
        return ['-o', f'ControlPath={self.directory}/%C-{index}']

    def open(self, host, sessions=1):
        """Log in to host, enough times for sessions commands at once."""
        target = self.target(host)
        wanted = max(1, math.ceil(sessions / self.sessions_per_login))
        while self.logins.get(target, 0) < wanted:
            index = self.logins.get(target, 0)
            self._login(target, index)
            self.logins[target] = index + 1

    def _login(self, target, index):
        if self.env is None:
            reuse = not self.batch and not self.password_per_node
            self.env = self._stack.enter_context(password_environment(reuse))
        argv = ['ssh', '-T', '-x', '-n', *self._control(target, index),
                '-o', 'ControlMaster=yes', '-o', f'ControlPersist={LOGIN_IDLE_LIMIT}',
                *ssh_options(self.extra, self.batch), target, 'true']
        # The login stays in the background holding stderr: a file, so nothing waits on it.
        with tempfile.TemporaryFile('w+') as said:
            try:
                done = subprocess.run(argv, env=self.env, stdin=subprocess.DEVNULL,
                                      stdout=subprocess.DEVNULL, stderr=said, text=True,
                                      timeout=self.timeout)
            except subprocess.TimeoutExpired:
                raise RuntimeError(f'{target}: no login within {self.timeout:g} s') from None
            if done.returncode:
                said.seek(0)
                lines = [line for line in said.read().splitlines() if line.strip()]
                raise RuntimeError(f'{target}: login failed (exit {done.returncode})'
                                   + (f': {lines[-1]}' if lines else ''))

    def argv(self, host, command, session=0):
        """The ssh command that runs command, a remote shell line, through host's login."""
        target = self.target(host)
        index = session // self.sessions_per_login
        if index >= self.logins.get(target, 0):
            raise RuntimeError(f'{target}: command {session} has no login; open the host for '
                               f'{session + 1} commands first')
        return ['ssh', '-T', '-x', *self._control(target, index), '-o', 'ControlMaster=no',
                *ssh_options(self.extra, batch=True), target, command]

    def wrapper(self):
        """An executable that runs ssh through the first login of whichever host it is given."""
        path = self.directory / 'ssh'
        options = ['-T', '-x', '-o', f'ControlPath={self.directory}/%C-0',
                   '-o', 'ControlMaster=no', *ssh_options(self.extra, batch=True)]
        path.write_text('#!/bin/sh\nexec ssh ' + shlex.join(options) + ' "$@"\n')
        path.chmod(0o700)
        return path


def run_relayed(args, command):
    """command on args.host, with the password relayed to it: its output as it comes."""
    if args.ssh_batch or args.password_per_node:
        print('remote: a relay serves the password: not with --batch or --password-per-node',
              file=sys.stderr)
        return 2
    target = f'{args.user}@{args.host}' if args.user and '@' not in args.host else args.host
    there = f'/tmp/remote-password-{secrets.token_hex(8)}.sock'
    line = (f'{PASSWORD_SOCKET}={there} {RELAYED}=1; export {PASSWORD_SOCKET} {RELAYED}; '
            f'{command}\ns=$?; rm -f {there}; exit $s')
    try:
        with password_environment(True) as env:
            # This connection alone forwards: the socket there, private to the user, to here.
            argv = ['ssh', '-T', '-x', '-o', 'ClearAllForwardings=no',
                    '-o', 'ExitOnForwardFailure=yes', '-o', 'StreamLocalBindUnlink=yes',
                    '-o', 'StreamLocalBindMask=0177', '-R', f'{there}:{env[PASSWORD_SOCKET]}',
                    *ssh_options(args.ssh_option), target, line]
            with subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
                for said in process.stdout:
                    if NOISE not in said:
                        sys.stdout.write(said)
                        sys.stdout.flush()
            return process.returncode
    except KeyboardInterrupt:
        return 130


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='what', required=True)
    check = commands.add_parser('check', help='log in to each host and print its name')
    run = commands.add_parser('run', help='run a command here while the logins are open')
    run.add_argument('command', nargs=argparse.REMAINDER,
                     help='after --: the command, which reaches hosts with "$REMOTE_SSH" HOST')
    on = commands.add_parser('on', help='run a shell command on each host and show what it says')
    on.add_argument('command', nargs=argparse.REMAINDER,
                    help='after --: the shell command, one line run by the login shell there')
    relay = commands.add_parser('relay', help='run a shell command on one host, which may log in '
                                              'to others with this password')
    relay.add_argument('--host', required=True, help='the host that runs the command')
    relay.add_argument('command', nargs=argparse.REMAINDER,
                       help='after --: the shell command, one line run by the login shell there')
    for sub in (check, run, on):
        sub.add_argument('--hosts', required=True, help='hosts, with commas between')
    for sub in (check, run, on, relay):
        add_arguments(sub)
    args = parser.parse_args(argv)
    if args.what == 'relay':
        command = list(args.command)
        if command[:1] == ['--']:
            command = command[1:]
        if not command:
            parser.error('relay needs a command after --')
        return run_relayed(args, ' '.join(command))
    hosts = [host.strip() for host in args.hosts.split(',') if host.strip()]
    if not hosts:
        parser.error('--hosts names no host')
    command = list(getattr(args, 'command', []))
    if command[:1] == ['--']:
        command = command[1:]
    if args.what in ('run', 'on') and not command:
        parser.error(f'{args.what} needs a command after --')
    try:
        with Session.from_args(args, hosts={host: 1 for host in hosts}) as session:
            if args.what == 'check':
                failed = 0
                for host in hosts:
                    done = subprocess.run(session.argv(host, 'hostname'), capture_output=True,
                                          text=True, stdin=subprocess.DEVNULL)
                    name = done.stdout.strip() or done.stderr.strip()
                    print(f"{session.target(host)}: {'ok' if done.returncode == 0 else 'FAILED'}"
                          f" ({name})")
                    failed += done.returncode != 0
                return 1 if failed else 0
            if args.what == 'on':
                failed = 0
                for host in hosts:
                    done = subprocess.run(session.argv(host, ' '.join(command)), text=True,
                                          capture_output=True, stdin=subprocess.DEVNULL)
                    for line in meaningful(done.stdout + done.stderr):
                        print(f'{session.target(host)}: {line}')
                    if done.returncode:
                        print(f'{session.target(host)}: exit {done.returncode}')
                        failed += 1
                return 1 if failed else 0
            env = dict(os.environ, REMOTE_SSH=str(session.wrapper()))
            return subprocess.run(command, env=env).returncode
    except RuntimeError as error:
        print(f'remote: {error}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    if sys.argv[1:2] == ['--askpass']:
        askpass(sys.argv[2] if len(sys.argv) > 2 else 'Password: ')
    else:
        sys.exit(main())
