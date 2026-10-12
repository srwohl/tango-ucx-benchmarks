// SPDX-FileCopyrightText: 2026 Copyright contributors to the tango-ucx project
// SPDX-License-Identifier: LGPL-3.0-or-later

// A local, two-process UCP baseline. This is our benchmark, not ucx_perftest.
// It rotates registered payload rings, copies eager receives, and returns credit only after
// consumption. Each role combines application work and progress on one thread.
#include "common.h"
#include "memory.h"

#include <ucp/api/ucp.h>

#include <algorithm>
#include <cerrno>
#include <iostream>
#include <limits>
#include <poll.h>
#include <sched.h>
#include <signal.h>
#include <stdexcept>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace
{
using bench::Json;
struct Options
{
    std::uint64_t bytes = 1024, frames = 1'000'000, rate = 0, corrupt_every = 0;
    std::uint32_t slots = 128, window = 128;
    std::optional<std::uint64_t> warmup;
    int sender_cpu = 0, receiver_cpu = 4;
    bool wait = false, verify = false, cuda = false;
    std::string sender_gpu = "0", receiver_gpu = "0";
};
void check(ucs_status_t status, const char *where)
{
    if(status != UCS_OK)
        throw std::runtime_error(std::string(where) + ": " + ucs_status_string(status));
}
std::uint64_t cpu_ns()
{
    timespec t{};
    if(clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t))
        throw std::runtime_error("clock_gettime");
    return std::uint64_t(t.tv_sec) * 1'000'000'000 + std::uint64_t(t.tv_nsec);
}
void pin(int cpu)
{
    cpu_set_t allowed;
    if(cpu < 0 || cpu >= CPU_SETSIZE || sched_getaffinity(0, sizeof(allowed), &allowed) ||
       !CPU_ISSET(cpu, &allowed))
        throw std::invalid_argument("CPU is outside the inherited affinity");
    CPU_ZERO(&allowed);
    CPU_SET(cpu, &allowed);
    if(sched_setaffinity(0, sizeof(allowed), &allowed))
        throw std::runtime_error("sched_setaffinity");
}
void io(int fd, void *data, std::size_t bytes, bool writing)
{
    auto *at = static_cast<std::byte *>(data);
    while(bytes)
    {
        auto n = writing ? ::send(fd, at, bytes, MSG_NOSIGNAL) : ::recv(fd, at, bytes, 0);
        if(n < 0 && errno == EINTR)
            continue;
        if(n <= 0)
            throw std::runtime_error("setup/result socket closed");
        at += n;
        bytes -= std::size_t(n);
    }
}
void send_text(int fd, std::string text)
{
    auto n = std::uint64_t(text.size());
    io(fd, &n, sizeof(n), true);
    io(fd, text.data(), text.size(), true);
}
std::string receive_text(int fd)
{
    std::uint64_t n;
    io(fd, &n, sizeof(n), false);
    if(n > 1'000'000)
        throw std::runtime_error("oversized setup/result");
    std::string text(n, '\0');
    io(fd, text.data(), text.size(), false);
    return text;
}
struct World;
struct Slot
{
    World *owner = nullptr;
    std::uint64_t index = 0, timestamp = 0;
    std::array<std::uint64_t, 5> header{}; // 40-byte frame header
    bool pending = false, landed = false;
};
struct World
{
    Options options;
    ucp_context_h context = nullptr;
    ucp_worker_h worker = nullptr;
    ucp_ep_h endpoint = nullptr;
    ucp_mem_h memh = nullptr;
    int event_fd = -1;
    bench::Memory memory;
    std::byte *payload = nullptr;
    std::vector<Slot> slots;
    std::array<std::uint64_t, 3> credit{}; // 24-byte cumulative credit header
    bool credit_pending = false;
    std::uint64_t acknowledged = 0, consumed = 0, credit_sent = 0;
    std::uint64_t eager_frames = 0, rendezvous_frames = 0;
    std::string failure;

    World(const Options &wanted, int setup_fd) : options(wanted), memory(wanted.cuda)
    {
        ucp_config_t *cfg;
        check(ucp_config_read("TANGO_UCX", nullptr, &cfg), "ucp_config_read");
        auto status = ucp_config_modify(cfg, "MM_ERROR_HANDLING", "y");
        if(status == UCS_OK)
            status = ucp_config_modify(cfg, "KEEPALIVE_INTERVAL", "2s");
        ucp_params_t params{};
        params.field_mask = UCP_PARAM_FIELD_FEATURES;
        params.features = UCP_FEATURE_AM | UCP_FEATURE_WAKEUP;
        if(status == UCS_OK)
            status = ucp_init(&params, cfg, &context);
        ucp_config_release(cfg);
        check(status, "UCP setup");
        ucp_worker_params_t worker_params{};
        worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
        worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
        check(ucp_worker_create(context, &worker_params, &worker), "ucp_worker_create");
        check(ucp_worker_get_efd(worker, &event_fd), "ucp_worker_get_efd");
        slots.resize(options.slots);
        for(auto &slot : slots)
            slot.owner = this;
        payload = memory.allocate(options.bytes * options.slots);
        for(std::uint32_t i = 0; i < options.slots; ++i)
            memory.fill(data(i), 0);
        ucp_mem_map_params_t mapping{};
        mapping.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                             UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
        mapping.address = payload;
        mapping.length = options.bytes * options.slots;
        mapping.memory_type = options.cuda ? UCS_MEMORY_TYPE_CUDA : UCS_MEMORY_TYPE_HOST;
        check(ucp_mem_map(context, &mapping, &memh), "ucp_mem_map");
        handler(1, frame);
        handler(2, on_credit);
        ucp_address_t *address;
        std::size_t address_bytes;
        check(ucp_worker_get_address(worker, &address, &address_bytes), "ucp_worker_get_address");
        send_text(setup_fd, std::string(reinterpret_cast<char *>(address), address_bytes));
        ucp_worker_release_address(worker, address);
        auto peer = receive_text(setup_fd);
        ucp_ep_params_t ep_params{};
        ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
                               UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                               UCP_EP_PARAM_FIELD_ERR_HANDLER;
        ep_params.address = reinterpret_cast<const ucp_address_t *>(peer.data());
        ep_params.err_mode = UCP_ERR_HANDLING_MODE_PEER;
        ep_params.err_handler.cb = error;
        ep_params.err_handler.arg = this;
        check(ucp_ep_create(worker, &ep_params, &endpoint), "ucp_ep_create");
    }
    ~World()
    {
        if(endpoint)
        {
            ucp_request_param_t params{};
            params.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
            params.flags = UCP_EP_CLOSE_FLAG_FORCE;
            auto request = ucp_ep_close_nbx(endpoint, &params);
            if(UCS_PTR_IS_PTR(request))
            {
                while(ucp_request_check_status(request) == UCS_INPROGRESS)
                    ucp_worker_progress(worker);
                ucp_request_free(request);
            }
        }
        if(memh)
            ucp_mem_unmap(context, memh);
        if(worker)
            ucp_worker_destroy(worker);
        if(context)
            ucp_cleanup(context);
    }
    std::span<std::byte> data(std::size_t slot)
    {
        return {payload + slot * options.bytes, std::size_t(options.bytes)};
    }
    void handler(unsigned id, ucp_am_recv_callback_t callback)
    {
        ucp_am_handler_param_t params{};
        params.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID | UCP_AM_HANDLER_PARAM_FIELD_CB |
                            UCP_AM_HANDLER_PARAM_FIELD_ARG | UCP_AM_HANDLER_PARAM_FIELD_FLAGS;
        params.id = id;
        params.cb = callback;
        params.arg = this;
        params.flags = UCP_AM_FLAG_WHOLE_MSG;
        check(ucp_worker_set_am_recv_handler(worker, &params), "AM handler");
    }
    void check_failure()
    {
        if(!failure.empty())
            throw std::runtime_error(failure);
    }
    static void error(void *arg, ucp_ep_h, ucs_status_t status)
    {
        static_cast<World *>(arg)->failure = ucs_status_string(status);
    }
    static void sent(void *request, ucs_status_t status, void *arg)
    {
        auto &slot = *static_cast<Slot *>(arg);
        if(status != UCS_OK)
            slot.owner->failure = ucs_status_string(status);
        slot.pending = false;
        ucp_request_free(request);
    }
    static void received(void *request, ucs_status_t status, std::size_t, void *arg)
    {
        auto &slot = *static_cast<Slot *>(arg);
        if(status != UCS_OK)
            slot.owner->failure = ucs_status_string(status);
        slot.landed = true;
        slot.pending = false;
        ucp_request_free(request);
    }
    static ucs_status_t frame(void *arg, const void *header, std::size_t header_bytes,
                              void *incoming, std::size_t bytes, const ucp_am_recv_param_t *params)
    {
        auto &w = *static_cast<World *>(arg);
        if(header_bytes != 40 || bytes != w.options.bytes)
        {
            w.failure = "wrong frame shape";
            return UCS_OK;
        }
        std::array<std::uint64_t, 5> h;
        std::memcpy(h.data(), header, 40);
        auto &slot = w.slots[h[0] % w.options.slots];
        if(slot.pending || slot.landed || h[0] < w.consumed ||
           h[0] - w.consumed >= w.options.window)
        {
            w.failure = "receive slot reused before consumption, or frame outside window";
            return UCS_OK;
        }
        slot.index = h[0];
        slot.timestamp = h[1];
        if(params->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV)
        {
            ++w.rendezvous_frames;
            slot.pending = true;
            ucp_request_param_t receive{};
            receive.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA |
                                   UCP_OP_ATTR_FIELD_MEMH | UCP_OP_ATTR_FIELD_MEMORY_TYPE;
            receive.cb.recv_am = received;
            receive.user_data = &slot;
            receive.memh = w.memh;
            receive.memory_type = w.options.cuda ? UCS_MEMORY_TYPE_CUDA : UCS_MEMORY_TYPE_HOST;
            auto request = ucp_am_recv_data_nbx(
                w.worker, incoming, w.data(h[0] % w.options.slots).data(), bytes, &receive);
            if(UCS_PTR_IS_ERR(request))
                w.failure = ucs_status_string(UCS_PTR_STATUS(request));
            else if(!request)
                slot.pending = false, slot.landed = true;
            return UCS_INPROGRESS;
        }
        if(w.options.cuda)
        {
            w.failure = "a CUDA receive must arrive by rendezvous";
            return UCS_OK;
        }
        ++w.eager_frames;
        std::memcpy(w.data(h[0] % w.options.slots).data(), incoming, bytes);
        slot.landed = true;
        return UCS_OK;
    }
    static ucs_status_t on_credit(void *arg, const void *header, std::size_t bytes, void *,
                                  std::size_t, const ucp_am_recv_param_t *)
    {
        auto &w = *static_cast<World *>(arg);
        std::uint64_t count;
        if(bytes != 24)
            w.failure = "wrong credit shape";
        else
        {
            std::memcpy(&count, header, 8);
            if(count < w.acknowledged)
                w.failure = "credit went backwards";
            w.acknowledged = count;
        }
        return UCS_OK;
    }
    void send_frame(std::uint64_t index)
    {
        auto &slot = slots[index % options.slots];
        if(options.verify)
            memory.fill(data(index % options.slots), index,
                        options.corrupt_every &&
                            index % options.corrupt_every == options.corrupt_every - 1);
        auto timestamp =
            options.rate ? std::uint64_t(bench::now_ns())
                         : std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                             std::chrono::system_clock::now().time_since_epoch())
                                             .count());
        slot.header = {index, timestamp, options.bytes, 0, 0};
        slot.pending = true;
        ucp_request_param_t params{};
        params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA |
                              UCP_OP_ATTR_FIELD_MEMH | UCP_OP_ATTR_FIELD_MEMORY_TYPE |
                              UCP_OP_ATTR_FIELD_FLAGS;
        params.cb.send = sent;
        params.user_data = &slot;
        params.memh = memh;
        params.memory_type = options.cuda ? UCS_MEMORY_TYPE_CUDA : UCS_MEMORY_TYPE_HOST;
        params.flags = options.cuda ? UCP_AM_SEND_FLAG_RNDV : 0;
        auto request = ucp_am_send_nbx(endpoint, 1, slot.header.data(), 40,
                                       data(index % options.slots).data(), options.bytes, &params);
        if(UCS_PTR_IS_ERR(request))
            check(UCS_PTR_STATUS(request), "send frame");
        else if(!request)
            slot.pending = false;
    }
    static void credited(void *request, ucs_status_t status, void *arg)
    {
        auto &w = *static_cast<World *>(arg);
        if(status != UCS_OK)
            w.failure = ucs_status_string(status);
        w.credit_pending = false;
        ucp_request_free(request);
    }
    bool send_credit()
    {
        if(credit_pending || credit_sent == consumed)
            return false;
        credit = {consumed, consumed * options.bytes, 0};
        credit_sent = consumed;
        credit_pending = true;
        ucp_request_param_t params{};
        params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
        params.cb.send = credited;
        params.user_data = this;
        auto request = ucp_am_send_nbx(endpoint, 2, credit.data(), 24, nullptr, 0, &params);
        if(UCS_PTR_IS_ERR(request))
            check(UCS_PTR_STATUS(request), "send credit");
        else if(!request)
            credit_pending = false;
        return true;
    }
    void idle(bool work, std::int64_t now, std::int64_t &last_work, std::int64_t timeout)
    {
        if(work)
            last_work = now;
        else if(options.wait || now - last_work >= 200'000)
        {
            auto status = ucp_worker_arm(worker);
            if(status == UCS_ERR_BUSY)
                return;
            check(status, "ucp_worker_arm");
            pollfd fd{event_fd, POLLIN, 0};
            auto ns = std::clamp<std::int64_t>(timeout, 1'000, 1'000'000);
            timespec t{0, long(ns)};
            (void)::ppoll(&fd, 1, &t, nullptr);
        }
    }
};

Json receive(World &w, std::uint64_t warmup)
{
    auto total = warmup + w.options.frames;
    std::vector<std::uint64_t> latencies;
    if(w.options.rate)
        latencies.reserve(w.options.frames);
    std::uint64_t cpu_start = 0, cpu_stop = 0, mismatched = 0;
    std::int64_t start = 0, stop = 0, last_work = 0;
    auto until = bench::now_ns() + std::int64_t(w.options.rate ? total * 1e9 / w.options.rate : 0) +
                 60'000'000'000LL;
    while(w.consumed < total || w.credit_pending || w.credit_sent < total)
    {
        bool work = ucp_worker_progress(w.worker) != 0;
        while(w.consumed < total && w.slots[w.consumed % w.options.slots].landed)
        {
            auto &slot = w.slots[w.consumed % w.options.slots];
            if(slot.index != w.consumed)
                throw std::runtime_error("out of order frame");
            if(w.options.rate && w.consumed >= warmup)
                latencies.push_back(std::uint64_t(bench::now_ns()) - slot.timestamp);
            if(w.options.verify)
                mismatched += !w.memory.matches(w.data(w.consumed % w.options.slots), w.consumed);
            slot.landed = false;
            ++w.consumed;
            stop = bench::now_ns();
            if(w.consumed == warmup)
                start = stop, cpu_start = cpu_ns();
            if(w.consumed == total)
                cpu_stop = cpu_ns();
            work = true;
        }
        work |= w.send_credit();
        w.check_failure();
        auto now = bench::now_ns();
        if(now > until)
            throw std::runtime_error("receiver timed out");
        w.idle(work, now, last_work, 1'000'000);
    }
    Json latency = nullptr;
    if(!latencies.empty())
    {
        std::sort(latencies.begin(), latencies.end());
        latency = {{"samples", latencies.size()},
                   {"p50", latencies[(latencies.size() - 1) / 2]},
                   {"p99", latencies[std::size_t(0.99 * double(latencies.size() - 1))]},
                   {"max", latencies.back()}};
    }
    auto expected = w.options.corrupt_every ? total / w.options.corrupt_every : 0;
    return {{"ok", mismatched == expected},
            {"frames", w.options.frames},
            {"warmup", warmup},
            {"seconds", double(stop - start) / 1e9},
            {"cpu_seconds", double(cpu_stop - cpu_start) / 1e9},
            {"mismatched", mismatched},
            {"mismatched_expected", expected},
            {"eager_frames", w.eager_frames},
            {"rendezvous_frames", w.rendezvous_frames},
            {"publish_to_read_ns", latency}};
}
Json transmit(World &w, std::uint64_t warmup)
{
    auto total = warmup + w.options.frames;
    (void)::prctl(PR_SET_TIMERSLACK, 1UL, 0UL, 0UL, 0UL);
    auto timer_slack = ::prctl(PR_GET_TIMERSLACK, 0UL, 0UL, 0UL, 0UL);
    auto cpu_start = cpu_ns();
    auto start = bench::now_ns(), last_work = start;
    auto period = w.options.rate ? 1'000'000'000 / w.options.rate : 0;
    auto until = start + std::int64_t(total * period) + 60'000'000'000LL;
    std::uint64_t published = 0, late = 0;
    while(w.acknowledged < total)
    {
        bool work = ucp_worker_progress(w.worker) != 0;
        auto now = bench::now_ns();
        auto due = start + std::int64_t(published * period);
        if(published < total && published - w.acknowledged < w.options.window &&
           !w.slots[published % w.options.slots].pending && now >= due)
        {
            late += period && now > due + std::int64_t(period);
            w.send_frame(published++);
            work = true;
        }
        w.check_failure();
        if(now > until || w.acknowledged > published)
            throw std::runtime_error("sender timeout or invalid credit");
        w.idle(work, now, last_work, period && due > now ? due - now : 1'000'000);
    }
    while(std::ranges::any_of(w.slots, [](const Slot &s) { return s.pending; }))
    {
        ucp_worker_progress(w.worker);
        w.check_failure();
        if(bench::now_ns() > until)
            throw std::runtime_error("send completion timeout");
    }
    return {{"published", published},
            {"cpu_seconds", double(cpu_ns() - cpu_start) / 1e9},
            {"seconds", double(bench::now_ns() - start) / 1e9},
            {"pacing_late_frames", late},
            {"pacing_timer_slack_ns", timer_slack}};
}
Options parse(int argc, char **argv)
{
    Options o;
    for(int i = 1; i < argc; ++i)
    {
        std::string option = argv[i];
        if(option == "--verify")
            o.verify = true;
        else if(option == "--wait")
            o.wait = true;
        else if(option == "--cuda")
            o.cuda = true;
        else if(i + 1 >= argc)
            throw std::invalid_argument("missing value: " + option);
        else
        {
            if(option == "--sender-gpu" || option == "--receiver-gpu")
            {
                (option == "--sender-gpu" ? o.sender_gpu : o.receiver_gpu) = argv[++i];
                continue;
            }
            auto value = std::stoull(argv[++i]);
            if(option == "--bytes")
                o.bytes = value;
            else if(option == "--frames")
                o.frames = value;
            else if(option == "--rate")
                o.rate = value;
            else if(option == "--corrupt-every")
                o.corrupt_every = value;
            else if(option == "--warmup")
                o.warmup = value;
            else if(option == "--slots")
                o.slots = std::uint32_t(value);
            else if(option == "--window")
                o.window = std::uint32_t(value);
            else if(option == "--sender-cpu")
                o.sender_cpu = int(value);
            else if(option == "--receiver-cpu")
                o.receiver_cpu = int(value);
            else
                throw std::invalid_argument("unknown option: " + option);
        }
    }
    if(!o.bytes || !o.frames || !o.slots || !o.window || o.window > o.slots ||
       o.bytes > std::numeric_limits<std::size_t>::max() / o.slots || o.rate > 1'000'000'000 ||
       (o.corrupt_every && !o.verify) || (o.warmup && !*o.warmup))
        throw std::invalid_argument("positive sizes/counts required, window <= slots, rate <= 1e9");
    pin(o.sender_cpu);
    // Check the receiver against the original CPU set in main, before pinning either role.
    return o;
}
} // namespace

int main(int argc, char **argv)
{
    pid_t child = -1;
    try
    {
        cpu_set_t original;
        if(sched_getaffinity(0, sizeof(original), &original))
            throw std::runtime_error("sched_getaffinity");
        auto options = parse(argc, argv);
        if(options.receiver_cpu < 0 || options.receiver_cpu >= CPU_SETSIZE ||
           !CPU_ISSET(options.receiver_cpu, &original))
            throw std::invalid_argument("receiver CPU is outside inherited affinity");
        // Both processes inherit the original set and pin before UCX initialization.
        if(sched_setaffinity(0, sizeof(original), &original))
            throw std::runtime_error("restore affinity");
        int sockets[2];
        if(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets))
            throw std::runtime_error("socketpair");
        auto warmup = options.warmup.value_or(
            std::max(options.frames / 10, std::uint64_t(2) * options.slots));
        child = fork();
        if(child < 0)
            throw std::runtime_error("fork");
        if(child == 0)
        {
            ::close(sockets[0]);
            pin(options.receiver_cpu);
            if(options.cuda && setenv("CUDA_VISIBLE_DEVICES", options.receiver_gpu.c_str(), 1))
                throw std::runtime_error("select receiver GPU");
            {
                World receiver(options, sockets[1]);
                send_text(sockets[1], "ready");
                send_text(sockets[1], receive(receiver, warmup).dump());
                (void)receive_text(sockets[1]);
            }
            ::close(sockets[1]);
            _exit(0);
        }
        ::close(sockets[1]);
        pin(options.sender_cpu);
        if(options.cuda && setenv("CUDA_VISIBLE_DEVICES", options.sender_gpu.c_str(), 1))
            throw std::runtime_error("select sender GPU");
        Json result;
        {
            World sender(options, sockets[0]);
            (void)receive_text(sockets[0]);
            auto tx = transmit(sender, warmup);
            result = Json::parse(receive_text(sockets[0]));
            send_text(sockets[0], "done");
            result["sender"] = tx;
        }
        ::close(sockets[0]);
        int status;
        if(waitpid(child, &status, 0) < 0 || !WIFEXITED(status) || WEXITSTATUS(status))
            throw std::runtime_error("receiver failed");
        child = -1;
        result["program"] = "ucp_ring (local custom baseline)";
        result["frame_bytes"] = options.bytes;
        result["slots"] = options.slots;
        result["window"] = options.window;
        result["payload_working_set_bytes"] = 2 * options.bytes * options.slots;
        result["rate_requested"] = options.rate;
        result["idle_mode"] = options.wait ? "wait" : "default";
        result["verified"] = options.verify;
        result["memory"] = options.cuda ? "cuda to cuda" : "host to host";
        result["sender_gpu"] = options.cuda ? Json(options.sender_gpu) : Json(nullptr);
        result["receiver_gpu"] = options.cuda ? Json(options.receiver_gpu) : Json(nullptr);
        result["frames_per_second"] = double(options.frames) / result["seconds"].get<double>();
        auto cpu = result["cpu_seconds"].get<double>() / double(options.frames) +
                   result["sender"]["cpu_seconds"].get<double>() / double(warmup + options.frames);
        result["cpu_us_per_frame"] = cpu * 1e6;
        result["cpu_cores"] = cpu * result["frames_per_second"].get<double>();
        unsigned major, minor, patch;
        ucp_get_version(&major, &minor, &patch);
        result["ucx"] =
            std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
        std::cout << result.dump() << std::endl;
        return result["ok"].get<bool>() ? 0 : 1;
    }
    catch(const std::exception &e)
    {
        if(child > 0)
        {
            kill(child, SIGTERM);
            waitpid(child, nullptr, 0);
        }
        std::cerr << "ucp_ring: " << e.what() << std::endl;
        return 1;
    }
}
