// SPDX-FileCopyrightText: 2026 Copyright contributors to the tango-ucx project
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "common.h"

#ifdef BENCHMARKS_CUDA
#include <cuda_runtime.h>
#endif

#include <stdexcept>
#include <vector>

namespace bench
{
// Each role owns one non-default CUDA stream. Payload generation and full verification
// use host staging only when requested; the unverified timed loop reuses prefilled slots.
class Memory
{
  public:
    explicit Memory(bool cuda) : cuda_(cuda)
    {
        if(!cuda_)
            return;
#ifdef BENCHMARKS_CUDA
        check(cudaSetDevice(0), "cudaSetDevice");
        check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreate");
#else
        throw std::invalid_argument("CUDA needs a benchmark build with BENCHMARKS_CUDA=ON");
#endif
    }
    Memory(const Memory &) = delete;
    Memory &operator=(const Memory &) = delete;
    ~Memory()
    {
#ifdef BENCHMARKS_CUDA
        if(stream_)
        {
            (void)cudaStreamSynchronize(stream_);
            (void)cudaStreamDestroy(stream_);
        }
        if(device_)
            (void)cudaFree(device_);
#endif
    }
    TangoUcx::GpuStream stream() const
    {
#ifdef BENCHMARKS_CUDA
        return {stream_};
#else
        return {};
#endif
    }
    void synchronize()
    {
#ifdef BENCHMARKS_CUDA
        if(cuda_)
            check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
#endif
    }
    std::byte *allocate(std::size_t bytes)
    {
        if(!cuda_)
        {
            host_.resize(bytes);
            return host_.data();
        }
#ifdef BENCHMARKS_CUDA
        check(cudaMalloc(&device_, bytes), "cudaMalloc");
        return static_cast<std::byte *>(device_);
#else
        return nullptr;
#endif
    }
    void fill(std::span<std::byte> payload, std::uint64_t index, bool corrupt = false)
    {
        auto target = payload;
        if(cuda_)
        {
            staging_.resize(payload.size());
            target = staging_;
        }
        bench::fill(target, index);
        if(corrupt)
            target[target.size() / 2] ^= std::byte{1};
#ifdef BENCHMARKS_CUDA
        if(cuda_)
        {
            check(cudaMemcpyAsync(payload.data(), target.data(), target.size(),
                                  cudaMemcpyHostToDevice, stream_),
                  "cudaMemcpy H2D");
            synchronize(); // the generation buffer must not be rewritten while CUDA reads it
        }
#endif
    }
    bool matches(std::span<const std::byte> payload, std::uint64_t index)
    {
        if(!cuda_)
            return bench::matches(payload, index);
        staging_.resize(payload.size());
#ifdef BENCHMARKS_CUDA
        check(cudaMemcpyAsync(staging_.data(), payload.data(), payload.size(),
                              cudaMemcpyDeviceToHost, stream_),
              "cudaMemcpy D2H");
        synchronize();
#endif
        return bench::matches(staging_, index);
    }

  private:
#ifdef BENCHMARKS_CUDA
    static void check(cudaError_t status, const char *call)
    {
        if(status != cudaSuccess)
            throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
    }
    cudaStream_t stream_ = nullptr;
    void *device_ = nullptr;
#endif
    bool cuda_;
    std::vector<std::byte> host_, staging_;
};
} // namespace bench
