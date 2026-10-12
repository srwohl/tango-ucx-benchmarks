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
option(BENCHMARKS_CUDA "Build CUDA source and receive benchmark rings" OFF)

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

add_executable(tango_ucx_perf_publisher src/publisher.cpp)
target_link_libraries(tango_ucx_perf_publisher PRIVATE tango-ucx::ucx nlohmann_json::nlohmann_json)
add_executable(tango_ucx_perftest src/perftest.cpp)
target_link_libraries(tango_ucx_perftest PRIVATE tango-ucx::ucx nlohmann_json::nlohmann_json)
target_compile_definitions(tango_ucx_perftest PRIVATE
    BENCHMARKS_VERSION="${PROJECT_VERSION}"
    BENCHMARKS_BUILD="${CMAKE_BUILD_TYPE}"
    TANGO_UCX_VERSION="${tango-ucx_VERSION}"
    TANGO_UCX_BUILD="${TANGO_UCX_BUILD}")

add_executable(ucp_ring src/ucp_ring.cpp)
target_link_libraries(ucp_ring PRIVATE ucx::ucp ucx::ucs nlohmann_json::nlohmann_json)
target_include_directories(ucp_ring PRIVATE
    $<TARGET_PROPERTY:tango-ucx::transport,INTERFACE_INCLUDE_DIRECTORIES>)

if(BENCHMARKS_CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(NOT TARGET tango-ucx::cuda)
        message(FATAL_ERROR "Install tango-ucx with TANGO_UCX_CUDA=ON for GPU benchmarks")
    endif()
    foreach(program tango_ucx_perf_publisher tango_ucx_perftest ucp_ring)
        target_compile_definitions(${program} PRIVATE BENCHMARKS_CUDA)
        target_link_libraries(${program} PRIVATE CUDA::cudart)
    endforeach()
    foreach(program tango_ucx_perf_publisher tango_ucx_perftest)
        # The imported object registers the library's GPU runtime during startup.
        target_sources(${program} PRIVATE $<TARGET_OBJECTS:tango-ucx::cuda>)
        target_link_libraries(${program} PRIVATE tango-ucx::cuda)
    endforeach()
endif()

if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    if(BENCHMARKS_CUDA)
        add_test(NAME cuda_verified COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/compare_ucx.py --build ${CMAKE_CURRENT_BINARY_DIR}
            --memory cuda --self-test)
        set_tests_properties(cuda_verified PROPERTIES TIMEOUT 180 LABELS gpu RUN_SERIAL TRUE)
    endif()
    add_test(NAME ucp_ring_verified COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/compare_ucx.py --build ${CMAKE_CURRENT_BINARY_DIR} --self-test)
    set_tests_properties(ucp_ring_verified PROPERTIES TIMEOUT 120)
    # The library starts in this environment, and both builds are optimized.
    add_test(NAME version COMMAND tango_ucx_perftest --version)
    set_tests_properties(version PROPERTIES TIMEOUT 60
        PASS_REGULAR_EXPRESSION "\"library_starts\": true"
        FAIL_REGULAR_EXPRESSION "NOCONFIG;\"build\": \"(Debug)?\"")
    # Both roles on this host: an eager frame, a frame in fragments and a rendezvous one, with
    # every byte checked. Then the check itself: of frames damaged on purpose it finds each.
    set(bench ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/bench.py local
        --build ${CMAKE_CURRENT_BINARY_DIR} --runs 1 --bytes 32M)
    add_test(NAME bw_verified COMMAND ${bench} --sizes 1K,100K,1M --verify)
    add_test(NAME bw_finds_damage COMMAND ${bench} --sizes 64K --corrupt-every 10)
    add_test(NAME bw_batches COMMAND ${bench} --sizes 4K --batch 8 --verify)
    add_test(NAME bw_paced COMMAND ${bench} --sizes 4K --bytes 1M --rate 2000 --verify)
    add_test(NAME bw_wait_mode COMMAND ${bench} --sizes 1K,100K,1M --verify)
    set_tests_properties(bw_paced bw_wait_mode PROPERTIES
        TIMEOUT 120 FAIL_REGULAR_EXPRESSION "FAILED")
    set_tests_properties(bw_wait_mode PROPERTIES ENVIRONMENT "TANGO_UCX_IDLE=wait")
    # ucx_perftest beside it, both ways, and tango-ucx as a share of it.
    add_test(NAME bw_reference COMMAND ${bench} --sizes 1M --reference)
    set_tests_properties(bw_verified bw_finds_damage bw_batches bw_reference PROPERTIES
        TIMEOUT 120 FAIL_REGULAR_EXPRESSION "FAILED")
    # A sweep's file, and the report of it compared with itself.
    set(swept ${CMAKE_CURRENT_BINARY_DIR}/test-sweep.jsonl)
    add_test(NAME sweep_clears COMMAND ${CMAKE_COMMAND} -E rm -f ${swept})
    add_test(NAME sweep COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/bench.py local
        --build ${CMAKE_CURRENT_BINARY_DIR} --tier quick --sizes 4K,1M --runs 2 --bytes 32M
        --out ${swept})
    add_test(NAME report COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/report.py
        ${swept} ${swept} --label before,after)
    set_tests_properties(sweep_clears PROPERTIES FIXTURES_SETUP cleared)
    set_tests_properties(sweep PROPERTIES FIXTURES_REQUIRED cleared FIXTURES_SETUP swept
        TIMEOUT 120 FAIL_REGULAR_EXPRESSION "FAILED")
    set_tests_properties(report PROPERTIES FIXTURES_REQUIRED swept
        PASS_REGULAR_EXPRESSION "after against before.*\n.*\n.* tango-ucx .* \\+0% ")
    set_tests_properties(bw_reference PROPERTIES
        PASS_REGULAR_EXPRESSION "ucx_perftest -e .*\n.* tango-ucx .*%")
endif()
