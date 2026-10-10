# SPDX-FileCopyrightText: 2026 Copyright contributors to the tango-ucx project
# SPDX-License-Identifier: LGPL-3.0-or-later

cmake_minimum_required(VERSION 3.24)
project(tango-ucx-benchmarks VERSION 0.1.0 LANGUAGES CXX)
include(CTest)

# A rate from a build without optimization says nothing, so none is made by default.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_options(-fno-omit-frame-pointer)

find_package(tango-ucx CONFIG REQUIRED)
find_package(nlohmann_json 3.12 CONFIG REQUIRED)

# The same holds for the library. Its package records the build type it was installed from; a
# build with no type has none, as `pixi run build` gives where CMAKE_ARGS is empty.
option(BENCHMARKS_ANY_LIBRARY_BUILD "Accept a tango-ucx that was not built Release or RelWithDebInfo" OFF)
get_target_property(TANGO_UCX_BUILD tango-ucx::transport IMPORTED_CONFIGURATIONS)
if(NOT TANGO_UCX_BUILD MATCHES "RELEASE|RELWITHDEBINFO" AND NOT BENCHMARKS_ANY_LIBRARY_BUILD)
    message(FATAL_ERROR
        "tango-ucx in ${tango-ucx_DIR} was built as '${TANGO_UCX_BUILD}', without optimization. "
        "Build it with `pixi run build-profile` and run `pixi run stage` here.")
endif()

add_executable(tango_ucx_perftest src/perftest.cpp)
target_link_libraries(tango_ucx_perftest PRIVATE tango-ucx::ucx nlohmann_json::nlohmann_json)
target_compile_definitions(tango_ucx_perftest PRIVATE
    BENCHMARKS_VERSION="${PROJECT_VERSION}"
    BENCHMARKS_BUILD="${CMAKE_BUILD_TYPE}"
    TANGO_UCX_VERSION="${tango-ucx_VERSION}"
    TANGO_UCX_BUILD="${TANGO_UCX_BUILD}")

if(BUILD_TESTING)
    # The library starts in this environment, and both builds are optimized.
    add_test(NAME version COMMAND tango_ucx_perftest --version)
    set_tests_properties(version PROPERTIES TIMEOUT 60
        PASS_REGULAR_EXPRESSION "\"library_starts\": true"
        FAIL_REGULAR_EXPRESSION "NOCONFIG;\"build\": \"(Debug)?\"")
endif()
