// SPDX-FileCopyrightText: 2026 Copyright contributors to the tango-ucx project
// SPDX-License-Identifier: LGPL-3.0-or-later

// tango_ucx_perftest: the subscriber's side of the suite. So far it only reports what it was
// built from, which every result will carry.
#include <tango-ucx/ucx.h>

#include <nlohmann/json.hpp>
#include <ucp/api/ucp.h>

#include <cstdio>
#include <exception>
#include <string>
#include <string_view>

namespace
{
// A publisher makes a UCX context, a worker and its loop: the library built in one environment
// runs with the UCX of this one.
bool library_starts(std::string &why)
{
    try
    {
        TangoUcx::PublisherLimits limits;
        limits.budget = 8ull << 20;
        TangoUcx::Publisher publisher({TangoUcx::Element::U8, {128}, 0, {}}, limits);
        return true;
    }
    catch(const std::exception &error)
    {
        why = error.what();
        return false;
    }
}
nlohmann::ordered_json software()
{
    unsigned major = 0, minor = 0, release = 0;
    ucp_get_version(&major, &minor, &release);
    std::string why;
    bool starts = library_starts(why);
    nlohmann::ordered_json record{
        {"benchmarks", {{"version", BENCHMARKS_VERSION}, {"build", BENCHMARKS_BUILD}}},
        {"tango_ucx", {{"version", TANGO_UCX_VERSION}, {"build", TANGO_UCX_BUILD}}},
        {"ucx",
         {{"headers", std::to_string(UCP_API_MAJOR) + "." + std::to_string(UCP_API_MINOR)},
          {"library",
           std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(release)}}},
        {"compiler", __VERSION__},
        {"library_starts", starts}};
    if(!starts)
        record["failure"] = why;
    return record;
}
} // namespace

int main(int argc, char **argv)
{
    if(argc == 2 && std::string_view(argv[1]) == "--version")
    {
        auto record = software();
        std::puts(record.dump(2).c_str());
        return record["library_starts"].get<bool>() ? 0 : 1;
    }
    std::fputs("usage: tango_ucx_perftest --version\n"
               "No test is built yet: see README.md.\n",
               stderr);
    return 2;
}
