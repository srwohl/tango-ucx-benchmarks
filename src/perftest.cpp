// SPDX-FileCopyrightText: 2026 Copyright contributors to the tango-ucx project
// SPDX-License-Identifier: LGPL-3.0-or-later

// tango_ucx_perftest: the subscriber's side of the suite. It tells a tango_ucx_perf_publisher
// what to publish, receives it through an every subscription, and reports the rate.
#include "common.h"
#include "memory.h"

#include <tango-ucx/tango.h>
#include <tango/tango.h>
#include <ucp/api/ucp.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <getopt.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace
{
using namespace TangoUcx;
using namespace std::chrono_literals;
using bench::Json;

const char *const k_usage =
    "usage: tango_ucx_perftest <device> [options]\n"
    "       tango_ucx_perftest --version\n"
    "  <device>            the publisher, as host:port/bench/ucx/1#dbase=no\n"
    "  -t bw               the test: bandwidth of every frame, in order (the only one so far)\n"
    "  -s <bytes>          frame size; K, M and G are powers of 1024 (1M)\n"
    "  -n <frames>         frames to measure (1000)\n"
    "  -w <frames>         frames before the clock starts (a tenth of -n; at least both rings "
    "twice)\n"
    "  -d <depth>          ring depth on both sides (32); the publisher has at least 3 slots\n"
    "  -b <frames>         batch size (1)\n"
    "  --rate <frames/s>   pace the publisher; latency uses the same host's steady clock\n"
    "  --memory host|cuda  payload rings at both ends (host)\n"
    "  --thread-cpus p,g,s,r publisher loop, generator, subscriber loop, reader CPUs\n"
    "  --verify            fill every frame and check every byte; costs bandwidth\n"
    "  --corrupt-every <n> with --verify: damage every nth frame, and expect to find them\n"
    "  --wait <seconds>    how long to wait for the device to answer (30)\n"
    "  --json              one JSON line instead of the table\n";

struct Options
{
    std::string device, test = "bw", memory = "host";
    std::uint64_t frame_bytes = 1 << 20, frames = 1000, corrupt_every = 0, rate = 0;
    std::optional<std::uint64_t> warmup;
    std::uint32_t depth = 32, batch = 1;
    bool verify = false, json = false;
    double wait_seconds = 30;
    std::array<int, 4> cpus{-1, -1, -1, -1};
};

std::uint64_t bytes_of(std::string_view text)
{
    std::size_t used = 0;
    auto value = std::stoull(std::string(text), &used);
    auto suffix = text.substr(used);
    if(suffix.empty())
        return value;
    if(suffix == "K" || suffix == "k")
        return value << 10;
    if(suffix == "M" || suffix == "m")
        return value << 20;
    if(suffix == "G" || suffix == "g")
        return value << 30;
    throw std::invalid_argument("a size ends in K, M or G: " + std::string(text));
}

// A publisher makes a UCX context, a worker and its loop: the library built in one environment
// runs with the UCX of this one.
bool library_starts(std::string &why)
{
    try
    {
        PublisherLimits limits;
        limits.budget = 8ull << 20;
        Publisher publisher({Element::U8, {128}, 0, {}}, limits);
        return true;
    }
    catch(const std::exception &error)
    {
        why = error.what();
        return false;
    }
}
Json software()
{
    unsigned major = 0, minor = 0, release = 0;
    ucp_get_version(&major, &minor, &release);
    return {{"benchmarks", {{"version", BENCHMARKS_VERSION}, {"build", BENCHMARKS_BUILD}}},
            {"tango_ucx", {{"version", TANGO_UCX_VERSION}, {"build", TANGO_UCX_BUILD}}},
            {"ucx",
             {{"headers", std::to_string(UCP_API_MAJOR) + "." + std::to_string(UCP_API_MINOR)},
              {"library", std::to_string(major) + "." + std::to_string(minor) + "." +
                              std::to_string(release)}}},
            {"compiler", __VERSION__}};
}

// The device answers once its server has started; until then the proxy or its first call throws.
Tango::DeviceProxy connect(const Options &options)
{
    auto until = bench::now_ns() + std::int64_t(options.wait_seconds * 1e9);
    for(;;)
    {
        try
        {
            Tango::DeviceProxy proxy(options.device);
            proxy.ping();
            proxy.set_timeout_millis(30'000); // Configure registers the whole source
            return proxy;
        }
        catch(const Tango::DevFailed &)
        {
            if(bench::now_ns() >= until)
                throw std::runtime_error("no answer from " + options.device);
        }
        std::this_thread::sleep_for(50ms);
    }
}
// A command of the publisher that answers with a JSON document.
Json command(Tango::DeviceProxy &proxy, const char *name, const Json *document = nullptr)
{
    Tango::DeviceData reply;
    if(document)
    {
        Tango::DeviceData in;
        in << document->dump();
        reply = proxy.command_inout(name, in);
    }
    else
        reply = proxy.command_inout(name);
    std::string text;
    reply >> text;
    return Json::parse(text);
}

// The library derives a ring's depth from a budget and does not take a depth. This finds the
// budget of a depth. A deeper ring never costs less, so once one budget is known to be too small
// and one too large, the budget wanted is between them. `depth_of` makes the ring of a budget and
// says how deep it is, or 0 when the budget was refused.
std::uint64_t budget_for(std::uint32_t want, std::uint64_t frame_bytes,
                         const std::function<std::uint32_t(std::uint64_t)> &depth_of,
                         std::uint32_t &got)
{
    const std::uint64_t slot = frame_bytes + 1536; // a guess at what a slot costs
    std::uint64_t budget = (256u << 10) + want * slot, small = 0, large = 0;
    for(int tries = 0; tries < 48; ++tries)
    {
        got = depth_of(budget);
        if(got == want)
            return budget;
        (got < want ? small : large) = budget;
        if(small && large)
        {
            if(large - small < 2)
                break; // no budget gives this depth: the library rounds a depth down
            budget = small + (large - small) / 2;
        }
        else if(got < want)
            budget += (want - got) * slot;
        else
            budget = std::max<std::uint64_t>(1, budget / 2);
    }
    if(!got && large)
        got = depth_of(budget = large); // the least ring that was made
    return budget;
}

Json bandwidth(const Options &options)
{
    bool gpu = options.memory == "cuda";
    bench::Memory memory(gpu);
    auto proxy = connect(options);
    // Before the clock starts every slot of both rings is used twice: the first use of a page
    // faults it in, and the publisher fills each of its slots once.
    auto deepest = std::max<std::uint64_t>(options.depth, 3);
    auto warmup = std::max<std::uint64_t>(
        1, options.warmup.value_or(std::max(options.frames / 10, 2 * deepest)));
    auto total = warmup + options.frames;

    Json wanted{{"frame_bytes", options.frame_bytes},
                {"frames", total},
                {"rate", options.rate},
                {"verify", options.verify},
                {"corrupt_every", options.corrupt_every},
                {"publisher_cpu", options.cpus[0]},
                {"producer_cpu", options.cpus[1]},
                {"memory", options.memory}};
    std::uint32_t slots = 0, depth = 0;
    std::string refused; // why the last budget was
    auto publisher_budget = budget_for(
        std::max<std::uint32_t>(options.depth, 3), options.frame_bytes,
        [&](std::uint64_t budget)
        {
            wanted["budget"] = budget;
            try
            {
                return command(proxy, "Configure", &wanted).at("slots").get<std::uint32_t>();
            }
            catch(const Tango::DevFailed &error)
            {
                refused = error.errors.length() ? std::string(error.errors[0].desc) : "";
                return 0u; // a budget too small for the sessions
            }
        },
        slots);
    if(!slots)
        throw std::runtime_error("the publisher refused every budget tried: " + refused);

    // Every joins before the start. A subscription of the wrong depth is closed, and its session
    // with it, before the next is opened.
    std::optional<Subscription>
        subscription; // its default constructor needs the library's own headers
    SubscriptionOptions subscribe;
    subscribe.batch = options.batch;
    subscribe.label = "tango_ucx_perftest";
    subscribe.gpu = gpu ? 0 : -1;
    subscribe.progress_cpu = options.cpus[2];
    auto subscription_budget = budget_for(
        options.depth, options.frame_bytes,
        [&](std::uint64_t budget)
        {
            if(subscription)
                subscription->close(Deadline::clock::now() + 5s);
            subscription.reset(); // release an explicit progress-CPU reservation before retrying
            subscribe.budget = budget;
            try
            {
                subscription.emplace(every(proxy, subscribe));
            }
            catch(const std::invalid_argument &error)
            {
                refused = error.what();
                return 0u;
            }
            std::uint32_t deep = 0;
            for(auto until = bench::now_ns() + 2'000'000'000; !deep && bench::now_ns() < until;
                std::this_thread::sleep_for(1ms))
                deep = subscription->health().slots_free;
            return deep;
        },
        depth);
    if(!depth)
        throw std::runtime_error("no subscription could be opened: " + refused);

    bench::pin(options.cpus[3]);
    proxy.command_inout("Start");
    std::uint64_t seen = 0, measured_from = 0, mismatched = 0;
    std::int64_t clock_start = 0, clock_stop = 0;
    std::map<std::string, std::uint64_t> cpu;
    std::vector<std::uint64_t> latencies;
    if(options.rate)
        latencies.reserve(options.frames);
    for(;;)
    {
        auto batch = subscription->read(Deadline::clock::now() + 10s);
        if(!batch)
        {
            if(subscription->outcome())
                break;
            throw std::runtime_error("no frame for 10 seconds: " + subscription->health().failure);
        }
        if(options.rate)
        {
            const auto read_at = bench::now_ns();
            // This array benchmark has no user fields: each record is index, timestamp.
            const auto *records = static_cast<const std::byte *>(batch->records());
            for(std::size_t i = 0; i < batch->frames(); ++i)
                if(clock_start)
                {
                    std::uint64_t published_at;
                    std::memcpy(&published_at, records + i * 16 + 8, 8);
                    if(published_at > std::uint64_t(read_at))
                        throw std::runtime_error(
                            "paced latency requires both roles on the same host");
                    latencies.push_back(std::uint64_t(read_at) - published_at);
                }
        }
        auto count = batch->frames();
        if(gpu)
        {
            std::vector<std::uint64_t> indices;
            if(options.verify)
                for(std::size_t i = 0; i < count; ++i)
                    indices.push_back(batch->index(i));
            auto view = std::move(*batch).gpu_view(memory.stream());
            if(options.verify)
                for(std::size_t i = 0; i < count; ++i)
                    mismatched += !memory.matches(
                        {static_cast<const std::byte *>(view.payload(i)), view.payload_bytes(i)},
                        indices[i]);
        } // GPU release records an event after the consumer's last queued operation.
        else if(options.verify)
            for(std::size_t i = 0; i < count; ++i)
                mismatched += !memory.matches(batch->payload(i), batch->index(i));
        seen += count;
        batch.reset(); // released: the clock stops on a release, not on an arrival
        if(gpu && seen == total)
            memory.synchronize(); // include completion of the final GPU consumption/release event
        clock_stop = bench::now_ns();
        if(!clock_start && seen >= warmup)
        {
            clock_start = clock_stop;
            measured_from = seen;
            cpu = bench::thread_cpu_ns();
        }
    }
    auto health = subscription->health();
    auto outcome = subscription->outcome();
    auto cpu_seconds = bench::cpu_seconds_since(cpu);
    subscription->close(Deadline::clock::now() + 5s);

    Json publisher;
    for(auto until = bench::now_ns() + 10'000'000'000;; std::this_thread::sleep_for(5ms))
    {
        publisher = command(proxy, "Result");
        if(!publisher.value("running", false) || bench::now_ns() >= until)
            break;
    }

    auto frames = seen - measured_from;
    auto seconds = double(clock_stop - clock_start) / 1e9;
    auto expected = options.verify && options.corrupt_every ? total / options.corrupt_every : 0;
    auto bytes = double(frames) * double(options.frame_bytes);
    bool complete = outcome == Outcome::End && seen == total && clock_start && seconds > 0;
    Json latency = nullptr;
    if(!latencies.empty())
    {
        std::sort(latencies.begin(), latencies.end());
        auto percentile = [&](double p)
        { return latencies[std::size_t(p * double(latencies.size() - 1))]; };
        latency = {{"samples", latencies.size()},
                   {"p50", percentile(0.5)},
                   {"p99", percentile(0.99)},
                   {"max", latencies.back()}};
    }
    return {
        {"test", "bw"},
        {"ok", complete && mismatched == expected},
        {"frame_bytes", options.frame_bytes},
        {"frames", frames},
        {"warmup", measured_from},
        {"batch", options.batch},
        {"depth", {{"publisher", slots}, {"subscription", depth}}},
        {"budget", {{"publisher", publisher_budget}, {"subscription", subscription_budget}}},
        {"delivery", "every"},
        {"memory", gpu ? "cuda to cuda" : "host to host"},
        {"cuda", gpu},
        {"thread_cpus", options.cpus},
        {"reader_cpu", sched_getcpu()},
        {"seconds", seconds},
        {"frames_per_second", complete ? double(frames) / seconds : 0.0},
        {"rate_requested", options.rate},
        {"publish_to_read_ns", latency},
        {"MiB_per_second", complete ? bytes / seconds / double(1 << 20) : 0.0},
        {"Gb_per_second", complete ? bytes * 8 / seconds / 1e9 : 0.0},
        {"verified", options.verify},
        {"mismatched", mismatched},
        {"mismatched_expected", expected},
        {"outcome", outcome == Outcome::End      ? "end"
                    : outcome == Outcome::Failed ? "failed"
                    : outcome                    ? "cancelled"
                                                 : "none"},
        {"lanes", health.transport},
        {"subscription", {{"health", bench::health_json(health)}, {"cpu_seconds", cpu_seconds}}},
        {"publisher", publisher},
        {"software", software()}};
}

void print_table(const Json &r)
{
    std::printf("tango_ucx_perftest -t bw   every, %s   lanes: %s%s\n",
                r["memory"].get<std::string>().c_str(), r["lanes"].get<std::string>().c_str(),
                r["verified"].get<bool>() ? "   verified" : "");
    std::printf("%12s %10s %7s %7s %14s %12s %10s\n", "#bytes", "#frames", "depth", "batch",
                "frames/s", "MiB/s", "Gb/s");
    std::printf("%12llu %10llu %7u %7u %14.1f %12.1f %10.2f\n",
                static_cast<unsigned long long>(r["frame_bytes"].get<std::uint64_t>()),
                static_cast<unsigned long long>(r["frames"].get<std::uint64_t>()),
                r["depth"]["subscription"].get<unsigned>(), r["batch"].get<unsigned>(),
                r["frames_per_second"].get<double>(), r["MiB_per_second"].get<double>(),
                r["Gb_per_second"].get<double>());
    if(!r["ok"].get<bool>())
        std::printf("FAILED: outcome %s, %llu frames mismatched of %llu expected. %s\n",
                    r["outcome"].get<std::string>().c_str(),
                    static_cast<unsigned long long>(r["mismatched"].get<std::uint64_t>()),
                    static_cast<unsigned long long>(r["mismatched_expected"].get<std::uint64_t>()),
                    r["subscription"]["health"]["failure"].get<std::string>().c_str());
    if(r["rate_requested"].get<std::uint64_t>() && !r["publish_to_read_ns"].is_null())
        std::printf("publish to read: p50 %.1f us, p99 %.1f us, max %.1f us\n",
                    r["publish_to_read_ns"]["p50"].get<double>() / 1000,
                    r["publish_to_read_ns"]["p99"].get<double>() / 1000,
                    r["publish_to_read_ns"]["max"].get<double>() / 1000);
}

Options parse(int argc, char **argv)
{
    Options options;
    static const option longs[] = {{"verify", no_argument, nullptr, 1},
                                   {"corrupt-every", required_argument, nullptr, 2},
                                   {"wait", required_argument, nullptr, 3},
                                   {"json", no_argument, nullptr, 4},
                                   {"rate", required_argument, nullptr, 5},
                                   {"memory", required_argument, nullptr, 6},
                                   {"thread-cpus", required_argument, nullptr, 7},
                                   {nullptr, 0, nullptr, 0}};
    for(int c; (c = getopt_long(argc, argv, "t:s:n:w:d:b:", longs, nullptr)) != -1;)
        switch(c)
        {
            case 't':
                options.test = optarg;
                break;
            case 's':
                options.frame_bytes = bytes_of(optarg);
                break;
            case 'n':
                options.frames = std::stoull(optarg);
                break;
            case 'w':
                options.warmup = std::stoull(optarg);
                break;
            case 'd':
                options.depth = std::uint32_t(std::stoul(optarg));
                break;
            case 'b':
                options.batch = std::uint32_t(std::stoul(optarg));
                break;
            case 1:
                options.verify = true;
                break;
            case 2:
                options.corrupt_every = std::stoull(optarg);
                break;
            case 3:
                options.wait_seconds = std::stod(optarg);
                break;
            case 4:
                options.json = true;
                break;
            case 5:
                options.rate = std::stoull(optarg);
                break;
            case 6:
                options.memory = optarg;
                break;
            case 7:
            {
                std::string_view list(optarg);
                for(auto &cpu : options.cpus)
                {
                    auto comma = list.find(',');
                    auto value = list.substr(0, comma);
                    std::size_t used;
                    cpu = std::stoi(std::string(value), &used);
                    if(used != value.size() || cpu < 0)
                        throw std::invalid_argument("thread-cpus needs four nonnegative CPU IDs");
                    list = comma == list.npos ? std::string_view{} : list.substr(comma + 1);
                }
                if(!list.empty())
                    throw std::invalid_argument("thread-cpus needs exactly four CPU IDs");
                break;
            }
            default:
                throw std::invalid_argument("unknown option");
        }
    if(optind + 1 != argc)
        throw std::invalid_argument("one device is needed");
    options.device = argv[optind];
    if(options.test != "bw")
        throw std::invalid_argument("the only test so far is bw");
    if(options.memory != "host" && options.memory != "cuda")
        throw std::invalid_argument("memory is host or cuda");
    if(!options.frame_bytes || !options.frames || !options.depth || !options.batch)
        throw std::invalid_argument("sizes and counts are positive");
    if(options.batch > options.depth)
        throw std::invalid_argument("a batch is no larger than the ring is deep");
    if(options.rate > 1'000'000'000)
        throw std::invalid_argument("rate is at most one billion frames/s");
    return options;
}
} // namespace

int main(int argc, char **argv)
{
    if(argc == 2 && std::string_view(argv[1]) == "--version")
    {
        auto record = software();
        std::string why;
        record["library_starts"] = library_starts(why);
        if(!why.empty())
            record["failure"] = why;
        std::puts(record.dump(2).c_str());
        return record["library_starts"].get<bool>() ? 0 : 1;
    }
    Options options;
    try
    {
        options = parse(argc, argv);
    }
    catch(const std::exception &error)
    {
        std::fprintf(stderr, "tango_ucx_perftest: %s\n%s", error.what(), k_usage);
        return 2;
    }
    try
    {
        auto result = bandwidth(options);
        if(options.json)
            std::puts(result.dump().c_str());
        else
            print_table(result);
        return result["ok"].get<bool>() ? 0 : 1;
    }
    catch(const Tango::DevFailed &error)
    {
        Tango::Except::print_exception(error);
    }
    catch(const std::exception &error)
    {
        std::fprintf(stderr, "tango_ucx_perftest: %s\n", error.what());
    }
    return 1;
}
