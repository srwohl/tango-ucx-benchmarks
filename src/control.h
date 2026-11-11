// SPDX-FileCopyrightText: 2026 Copyright contributors to the tango-ucx project
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

// What a program that drives tango_ucx_perf_publisher needs: the device, its JSON commands, the
// budget that gives a ring the depth wanted, and the pullers of a writer process.
#include "common.h"

#include <tango-ucx/tango.h>
#include <tango/tango.h>

#include <algorithm>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace bench
{
using namespace std::chrono_literals;

// The device answers once its server has started; until then the proxy or its first call throws.
inline Tango::DeviceProxy connect(const std::string &device, double wait_seconds)
{
    auto until = bench::now_ns() + std::int64_t(wait_seconds * 1e9);
    for(;;)
    {
        try
        {
            Tango::DeviceProxy proxy(device);
            proxy.ping();
            proxy.set_timeout_millis(30'000); // Configure registers the whole source
            return proxy;
        }
        catch(const Tango::DevFailed &)
        {
            if(bench::now_ns() >= until)
                throw std::runtime_error("no answer from " + device);
        }
        std::this_thread::sleep_for(50ms);
    }
}
// A command of the publisher that answers with a JSON document.
inline Json command(Tango::DeviceProxy &proxy, const char *name, const Json *document = nullptr)
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
inline std::uint64_t budget_for(std::uint32_t want, std::uint64_t frame_bytes,
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

// The Unix clock, which the hosts of a run share: now_ns is each host's own.
inline std::int64_t unix_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

// The publisher's Configure at a source depth, as tango_ucx_perftest makes it.
inline Json configure_publisher(const std::string &device, double wait_seconds, std::uint32_t depth,
                                Json wanted)
{
    auto proxy = connect(device, wait_seconds);
    Json source = {{"kind", "generated"}};
    if(wanted.contains("hdf5"))
    {
        Json describe{{"hdf5", wanted["hdf5"]}};
        source = command(proxy, "Describe", &describe);
        wanted["frame_bytes"] = source.at("frame_bytes");
    }
    std::uint32_t slots = 0;
    std::string refused;
    auto budget = budget_for(
        depth, wanted.at("frame_bytes").get<std::uint64_t>(),
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
                return 0u;
            }
        },
        slots);
    if(!slots)
        throw std::runtime_error("the publisher refused every budget tried: " + refused);
    return {{"slots", slots},
            {"budget", budget},
            {"frame_bytes", wanted["frame_bytes"]},
            {"source", source}};
}

// The publisher's Result, once its run is over or the wait is.
inline Json publisher_result(const std::string &device, double wait_seconds)
{
    auto proxy = connect(device, wait_seconds);
    Json publisher;
    for(auto until = now_ns() + std::int64_t(wait_seconds * 1e9);;
        std::this_thread::sleep_for(20ms))
    {
        publisher = command(proxy, "Result");
        if(!publisher.value("running", false) || now_ns() >= until)
            return publisher;
    }
}

// Opens a puller; returns it once its ring is known, with the depth it has.
inline std::pair<TangoUcx::Subscription, std::uint32_t>
open_puller(Tango::DeviceProxy &proxy, std::uint32_t range, TangoUcx::SubscriptionOptions options)
{
    auto subscription = TangoUcx::pull(proxy, range, options);
    std::uint32_t depth = 0;
    for(auto until = now_ns() + 2'000'000'000; !depth && now_ns() < until;
        std::this_thread::sleep_for(1ms))
        depth = subscription.health().slots_free;
    return {std::move(subscription), depth};
}

// `count` pullers of `range` frames, each with a ring `depth` frames deep. The first finds the
// budget of that depth, which is left in options.budget; the others are opened with it.
inline std::vector<TangoUcx::Subscription>
open_pullers(Tango::DeviceProxy &proxy, std::uint32_t count, std::uint32_t range,
             std::uint32_t depth, std::uint64_t frame_bytes, TangoUcx::SubscriptionOptions &options)
{
    std::optional<TangoUcx::Subscription> first;
    std::string refused;
    std::uint32_t got = 0;
    options.budget = budget_for(
        depth, frame_bytes,
        [&](std::uint64_t budget)
        {
            if(first)
                first->close(TangoUcx::Deadline::clock::now() + 5s);
            first.reset();
            options.budget = budget;
            try
            {
                auto [subscription, deep] = open_puller(proxy, range, options);
                first.emplace(std::move(subscription));
                return deep;
            }
            catch(const std::invalid_argument &error)
            {
                refused = error.what();
                return 0u;
            }
        },
        got);
    if(got != depth || !first)
        throw std::runtime_error(
            "no receive ring of " + std::to_string(depth) + " frames could be opened: " +
            (refused.empty() ? "the depth found is " + std::to_string(got) : refused));
    std::vector<TangoUcx::Subscription> pullers;
    pullers.push_back(std::move(*first));
    for(std::uint32_t t = 1; t < count; ++t)
    {
        auto [subscription, deep] = open_puller(proxy, range, options);
        if(deep != depth)
            throw std::runtime_error("a later puller's ring is not as deep as the first");
        pullers.push_back(std::move(subscription));
    }
    return pullers;
}
} // namespace bench
