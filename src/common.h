// SPDX-FileCopyrightText: 2026 Copyright contributors to the tango-ucx project
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

// What the publisher device and the subscriber share: the bytes of a frame, and what a role
// reports of itself.
#include <tango-ucx/ucx.h>

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sched.h>
#include <span>
#include <string>

namespace bench
{
using Json = nlohmann::ordered_json;

inline void pin(int cpu)
{
    if(cpu < 0)
        return;
    cpu_set_t allowed;
    if(cpu >= CPU_SETSIZE || sched_getaffinity(0, sizeof(allowed), &allowed) ||
       !CPU_ISSET(cpu, &allowed))
        throw std::invalid_argument("CPU is outside the inherited affinity");
    CPU_ZERO(&allowed);
    CPU_SET(cpu, &allowed);
    if(sched_setaffinity(0, sizeof(allowed), &allowed))
        throw std::runtime_error("sched_setaffinity");
}

inline std::int64_t now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// A frame's bytes are 64-bit words, each the frame's seed plus the word's position, so a frame
// from another slot, a torn one and a short one all differ. The last word may be cut.
inline std::uint64_t seed_of(std::uint64_t frame_index)
{
    return (frame_index + 1) * 0x9e3779b97f4a7c15ull;
}
inline void fill(std::span<std::byte> payload, std::uint64_t frame_index)
{
    auto seed = seed_of(frame_index);
    auto *at = payload.data();
    std::size_t words = payload.size() / 8;
    for(std::size_t i = 0; i < words; ++i, at += 8)
    {
        std::uint64_t word = seed + i;
        std::memcpy(at, &word, 8);
    }
    std::uint64_t last = seed + words;
    std::memcpy(at, &last, payload.size() % 8);
}
inline bool matches(std::span<const std::byte> payload, std::uint64_t frame_index)
{
    auto seed = seed_of(frame_index);
    auto *at = payload.data();
    std::size_t words = payload.size() / 8;
    std::uint64_t differ = 0;
    for(std::size_t i = 0; i < words; ++i, at += 8)
    {
        std::uint64_t word;
        std::memcpy(&word, at, 8);
        differ |= word ^ (seed + i);
    }
    std::uint64_t last = seed + words;
    return !differ && std::memcmp(at, &last, payload.size() % 8) == 0;
}

// The time each thread of this process has spent on a CPU, by thread name. The library names its
// loops, so a loop's time is told from the role's own.
inline std::map<std::string, std::uint64_t> thread_cpu_ns()
{
    std::map<std::string, std::uint64_t> by_name;
    std::error_code ignored;
    for(const auto &task : std::filesystem::directory_iterator("/proc/self/task", ignored))
    {
        std::string name;
        std::uint64_t on_cpu = 0;
        std::getline(std::ifstream(task.path() / "comm"), name);
        if(std::ifstream(task.path() / "schedstat") >> on_cpu)
            by_name[name] += on_cpu;
    }
    return by_name;
}
inline Json cpu_seconds_since(const std::map<std::string, std::uint64_t> &before)
{
    Json seconds = Json::object();
    for(const auto &[name, ns] : thread_cpu_ns())
    {
        auto was = before.find(name);
        auto spent = ns - (was == before.end() ? 0 : was->second);
        if(spent >= 1'000'000) // a thread that did a millisecond of work
            seconds[name] = double(spent) / 1e9;
    }
    return seconds;
}

inline Json health_json(const TangoUcx::Health &health)
{
    return {{"frames", health.frames},
            {"skipped", health.skipped},
            {"pressure", health.pressure},
            {"slots_free", health.slots_free},
            {"slots_held", health.slots_held},
            {"transfer_mean_ns", health.transfer_mean_ns},
            {"transfer_max_ns", health.transfer_max_ns},
            {"quarantined_bytes", health.quarantined_bytes},
            {"transport", health.transport},
            {"topology",
             {{"cpu", health.topology.cpu},
              {"numa_node", health.topology.numa_node},
              {"ring_numa_node", health.topology.ring_numa_node},
              {"allowed_cpus", health.topology.allowed_cpus},
              {"sibling_cpus", health.topology.sibling_cpus}}},
            {"failure", health.failure}};
}
} // namespace bench
