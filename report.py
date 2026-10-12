#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Copyright contributors to the tango-ucx project
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Make a Markdown report of one sweep, or compare several.

    report.py results/a.jsonl                      one sweep: tango-ucx against ucx_perftest
    report.py results/a.jsonl results/b.jsonl      and b against a, point by point
    report.py a.jsonl b.jsonl --label before,after --out compared.md

The first file is what the others are compared with.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

RAW = "ucx_perftest -e"  # UCX as tango-ucx asks: what "of raw" is a share of
NOISY = 0.15  # a point whose best and worst differ by more of its median is marked


def frame(count):
    for unit, scale in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if count >= scale:
            return f"{count / scale:g} {unit}"
    return f"{count} B"


def rate(value):
    return "" if value is None else f"{value:,.0f}"


def load(path):
    """A sweep's host record and its points, each the runs of one program, size and depth."""
    meta, points = {}, {}
    with open(path) as lines:
        for line in lines:
            record = json.loads(line)
            if "meta" in record:
                meta = record["meta"]
                continue
            depth = record.get("depth_asked")
            if depth is None and isinstance(record.get("depth"), dict):
                depth = record["depth"]["subscription"]  # a file from before depth_asked
            key = (record.get("program", "tango-ucx"), record["frame_bytes"], depth)
            point = points.setdefault(key, {"runs": [], "failed": 0, "skipped": None, "path": ""})
            if "skipped" in record:
                point["skipped"] = record["skipped"]
            elif not record.get("ok"):
                point["failed"] += 1
                point["why"] = record.get("failure") or record.get("outcome", "")
            else:
                point["runs"].append(record)
                # A bar would end a table's cell: an older file joined alternatives with one.
                point["path"] = (record.get("protocol") or record.get("lanes", "")).replace(
                    " | ", "; or ")
    for point in points.values():
        rates = sorted(run["MiB_per_second"] for run in point["runs"])
        if rates:
            point["median"] = statistics.median(rates)
            point["best"], point["worst"] = rates[-1], rates[0]
            point["spread"] = (rates[-1] - rates[0]) / point["median"]
            point["frames_per_second"] = statistics.median(
                run["frames_per_second"] for run in point["runs"])
    return meta, points


def table(header, rows, words=1):
    """A Markdown table; its first `words` columns are text, the rest numbers."""
    if not rows:
        return ["Nothing was measured for this.", ""]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" if i < words else "---:" for i in range(len(header))) + "|"]
    return lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows] + [""]


def median_of(points, program, size, depth=None):
    return points.get((program, size, depth), {}).get("median")


def share(points, size, depth):
    mine, raw = median_of(points, "tango-ucx", size, depth), median_of(points, RAW, size)
    return f"{100 * mine / raw:.0f}%" if mine and raw else ""


def host_section(labels, sweeps):
    rows = []
    names = [("Date", "date"), ("Host", "host"), ("CPU", "cpu"), ("CPUs", "cpus"),
             ("Kernel", "kernel"), ("Governor", "governor"), ("Load at the start", "load"),
             ("tango-ucx", "tango_ucx"), ("tango-ucx-benchmarks", "benchmarks"),
             ("Tier", "tier"), ("Options", "options"), ("Settings", "settings")]
    for title, key in names:
        rows.append([title] + [str(meta.get(key, "not recorded")) for meta, _ in sweeps])
    lines = ["## What was measured", ""]
    lines += ["| | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    lines += ["| " + " | ".join(row) + " |" for row in rows] + [""]
    for label, (meta, _) in zip(labels, sweeps):
        notes = []
        if meta.get("governor") not in (None, "performance", "unknown"):
            notes.append(f"the CPU governor was {meta['governor']}")
        if isinstance(meta.get("load"), (int, float)) and meta["load"] > 1.0:
            notes.append(f"the host was busy, load {meta['load']:.2f}")
        if "dirty" in str(meta.get("tango_ucx", "")) + str(meta.get("benchmarks", "")):
            notes.append("a checkout had changes that are in no commit")
        if notes:
            lines.append(f"- **{label}:** " + "; ".join(notes) + ".")
    return lines + [""]


def share_section(label, points):
    sizes = sorted({size for _, size, _ in points})
    depths = sorted({depth for program, _, depth in points
                     if program == "tango-ucx" and depth is not None})
    header = ["Frame", "ucx_perftest", RAW]
    for depth in depths:
        header += [f"tango-ucx, depth {depth}", "of raw"]
    rows = []
    for size in sizes:
        row = [frame(size), rate(median_of(points, "ucx_perftest", size)),
               rate(median_of(points, RAW, size))]
        for depth in depths:
            row += [rate(median_of(points, "tango-ucx", size, depth)), share(points, size, depth)]
        rows.append(row)
    return [f"## tango-ucx as a share of raw UCX: {label}", "",
            "Median MiB/s. `ucx_perftest` is UCX as it chooses; `" + RAW + "` is UCX as "
            "tango-ucx asks, and \"of raw\" is a share of it. `ucx_perftest` sends from one "
            "buffer, so the least depth is the nearer comparison.", ""] + table(header, rows)


def small_frame_section(label, points):
    depths = sorted({depth for program, _, depth in points
                     if program == "tango-ucx" and depth is not None})
    rows = []
    for size in sorted({size for _, size, _ in points if size <= 64 << 10}):
        raw = points.get((RAW, size, None), {}).get("frames_per_second")
        for depth in depths:
            mine = points.get(("tango-ucx", size, depth), {}).get("frames_per_second")
            if raw and mine:
                rows.append([frame(size), depth, f"{mine:,.0f}", f"{1e9 / mine:,.0f}",
                             f"{1e9 / raw:,.0f}", f"{1e9 / mine - 1e9 / raw:,.0f}"])
    return [f"## The cost of a small frame: {label}", "",
            "Nanoseconds a frame, from the frames a second: what tango-ucx adds to UCX's own "
            "for each frame.", ""] + table(
                ["Frame", "Depth", "tango-ucx frames/s", "tango-ucx ns", "UCX ns", "Added ns"],
                rows)


def compared_section(labels, sweeps):
    base = sweeps[0][1]
    keys = sorted({key for _, points in sweeps for key in points},
                  key=lambda k: (k[1], k[0] != "ucx_perftest", k[0] != RAW, k[2] or 0))
    header = ["Frame", "Program", "Depth", labels[0]]
    for label in labels[1:]:
        header += [label, f"{label} against {labels[0]}"]
    rows = []
    for key in keys:
        program, size, depth = key
        first = base.get(key, {})
        row = [frame(size), program, "" if depth is None else depth, rate(first.get("median"))]
        for _, points in sweeps[1:]:
            other = points.get(key, {})
            change = ""
            if first.get("median") and other.get("median"):
                moved = other["median"] / first["median"] - 1
                # A change smaller than what either sweep's own runs differ by shows nothing,
                # and one run has no spread to compare it with.
                if min(len(first["runs"]), len(other["runs"])) < 2:
                    note = " (one run: may be noise)"
                elif abs(moved) <= max(first["spread"], other["spread"]):
                    note = " (within the spread)"
                else:
                    note = ""
                change = f"{moved:+.0%}{note}"
            row += [rate(other.get("median")), change]
        rows.append(row)
    return ["## Compared", "",
            f"Median MiB/s of each point, and each sweep against {labels[0]}. A change no larger "
            "than the spread of the runs of either is marked: it shows nothing.", ""] + table(
                header, rows, words=2)


def trouble_section(labels, sweeps):
    lines = ["## Points to read with care", ""]
    found = False
    for label, (_, points) in zip(labels, sweeps):
        for (program, size, depth), point in sorted(points.items(), key=lambda i: (i[0][1], i[0][0])):
            where = f"{label}: {frame(size)}, {program}" + ("" if depth is None else f" at depth {depth}")
            if point["skipped"]:
                lines.append(f"- {where} was left out: {point['skipped']}.")
            elif point["failed"]:
                lines.append(f"- {where} failed in {point['failed']} of "
                             f"{point['failed'] + len(point['runs'])} runs: {point.get('why', '')}")
            elif point.get("spread", 0) > NOISY:
                lines.append(f"- {where}: best {point['best']:,.0f} and worst "
                             f"{point['worst']:,.0f} MiB/s are {point['spread']:.0%} of the "
                             "median apart.")
            else:
                continue
            found = True
    return lines + ([""] if found else ["None.", ""])


def path_section(label, points):
    sizes = sorted({size for _, size, _ in points})
    rows = []
    for size in sizes:
        mine = next((p["path"] for (program, s, _), p in sorted(points.items(), key=str)
                     if program == "tango-ucx" and s == size and p["path"]), "")
        rows.append([frame(size), points.get(("ucx_perftest", size, None), {}).get("path", ""),
                     points.get((RAW, size, None), {}).get("path", ""), mine])
    return [f"## The path UCX chose: {label}", "",
            "For `ucx_perftest`, what `UCX_PROTO_INFO` names for a frame of that size; where an "
            "endpoint's tables disagree each is given. For tango-ucx, the lanes of the "
            "subscription: its protocol is not recorded yet.", ""] + table(
                ["Frame", "ucx_perftest", RAW, "tango-ucx lanes"], rows, words=4)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="the .jsonl files of sweeps")
    parser.add_argument("--label", help="a name for each file, with commas between")
    parser.add_argument("--out", help="write the report here and not to the terminal")
    args = parser.parse_args()
    labels = args.label.split(",") if args.label else [Path(path).stem for path in args.results]
    if len(labels) != len(args.results):
        parser.error("as many labels as files")
    sweeps = [load(path) for path in args.results]

    lines = ["# tango-ucx against ucx_perftest", ""]
    lines += host_section(labels, sweeps)
    if len(sweeps) > 1:
        lines += compared_section(labels, sweeps)
    for label, (_, points) in zip(labels, sweeps):
        lines += share_section(label, points)
    for label, (_, points) in zip(labels, sweeps):
        lines += small_frame_section(label, points)
    lines += trouble_section(labels, sweeps)
    for label, (_, points) in zip(labels, sweeps):
        lines += path_section(label, points)
    text = "\n".join(lines).rstrip() + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
