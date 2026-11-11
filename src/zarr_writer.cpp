// SPDX-FileCopyrightText: 2026 Copyright contributors to the tango-ucx project
// SPDX-License-Identifier: LGPL-3.0-or-later

// tango_ucx_zarr_writer: frames of a tango_ucx_perf_publisher written to Zarr v3 parts by
// acquire-zarr. The counterpart of tango_ucx_hdf5_writer: each writer thread is a puller of the
// publisher's pull set and owns one part, a store with one array, /images, that grows by the
// frames it is given. A batch is appended straight from the receive ring and then released.
// Unlike H5Dwrite, an append copies the frames into the stream's queue and returns, and the
// stream's own threads compress and write them, so a writer's credit returns before its frames
// are on disk. ZarrStream_close waits for them and makes each shard file durable. Streams share
// nothing: there is no lock between the threads of a process.
//
// A part is made of whole shards of the stream: the publisher gives a puller ranges of R frames
// that start at multiples of R, and R is a whole number of shards. zarr_write.py then makes one
// native Zarr store of the run whose shard files are links to the parts' shard files.
//
//   configure <device> --depth <d> <json>   the publisher's Configure, at the source depth d
//   start <device>                          its Start
//   result <device>                         its Result, once the run is over
//   write <device> <prefix> [options]       join the pull set, say ready, write until the end
#include "common.h"
#include "control.h"

#include <tango-ucx/tango.h>
#include <tango/tango.h>

#include <acquire.zarr.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <optional>
#include <pthread.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace
{
using namespace TangoUcx;
using namespace std::chrono_literals;
using bench::Json;
using bench::unix_ns;

const char *const k_usage =
    "usage: tango_ucx_zarr_writer configure <device> --depth <slots> <json>\n"
    "       tango_ucx_zarr_writer start <device>\n"
    "       tango_ucx_zarr_writer result <device>\n"
    "       tango_ucx_zarr_writer write <device> <prefix> [options]\n"
    "  write: part n is the store <prefix>.part_<nnn>.zarr; existing parts are refused\n"
    "  --frame-bytes <n>        the stream's frame size, as configure gave it (needed)\n"
    "  --threads <t>            writer threads in this process, each its own part (1)\n"
    "  --first-part <n>         the part of the first thread (0)\n"
    "  --frames-per-block <f>   frames in one append: the subscription's batch (8)\n"
    "  --blocks-per-stripe <b>  blocks a writer takes at its turn: the pull range is f * b (1)\n"
    "  --queue <q>              ranges each receive ring holds (2)\n"
    "  --chunk-frames <c>       frames in a Zarr chunk (the block)\n"
    "  --shard-chunks <s>       chunks in a shard file; c * s must divide the range (range / c)\n"
    "  --compression <kind>     none, blosc-lz4, blosc-zstd or zstd (none)\n"
    "  --level <n>              the compression level (1)\n"
    "  --shuffle <n>            blosc's shuffle: 0 none, 1 bytes, 2 bits (1; 0 without blosc)\n"
    "  --zarr-threads <n>       acquire-zarr's threads for each stream (0: its own choice)\n"
    "  --direct-io              write with O_DIRECT (ZARR_DIRECT_IO=1)\n"
    "  --wait <seconds>         the longest wait for the device or for a frame (120)\n";

void check(ZarrStatusCode status, const char *call)
{
    if(status != ZarrStatusCode_Success)
        throw std::runtime_error(std::string(call) + ": " + Zarr_get_status_message(status));
}

std::pair<ZarrDataType, std::size_t> zarr_type(Element element)
{
    switch(element)
    {
        case Element::U8:
            return {ZarrDataType_uint8, 1};
        case Element::U16:
            return {ZarrDataType_uint16, 2};
        case Element::U32:
            return {ZarrDataType_uint32, 4};
        case Element::U64:
            return {ZarrDataType_uint64, 8};
        case Element::I8:
            return {ZarrDataType_int8, 1};
        case Element::I16:
            return {ZarrDataType_int16, 2};
        case Element::I32:
            return {ZarrDataType_int32, 4};
        case Element::I64:
            return {ZarrDataType_int64, 8};
        case Element::F32:
            return {ZarrDataType_float32, 4};
        case Element::F64:
            return {ZarrDataType_float64, 8};
        case Element::Bytes:
            break;
    }
    throw std::invalid_argument("an opaque stream has no Zarr image type");
}

struct Compression
{
    ZarrCompressor compressor = ZarrCompressor_None;
    ZarrCompressionCodec codec = ZarrCompressionCodec_None;
    std::string name = "none";
};
Compression compression_of(std::string_view name)
{
    if(name == "none")
        return {};
    if(name == "blosc-lz4")
        return {ZarrCompressor_Blosc1, ZarrCompressionCodec_BloscLZ4, std::string(name)};
    if(name == "blosc-zstd")
        return {ZarrCompressor_Blosc1, ZarrCompressionCodec_BloscZstd, std::string(name)};
    if(name == "zstd")
        return {ZarrCompressor_Zstd, ZarrCompressionCodec_Zstd, std::string(name)};
    throw std::invalid_argument("--compression is none, blosc-lz4, blosc-zstd or zstd");
}

struct Options
{
    std::string command, device, prefix;
    std::uint32_t threads = 1, first_part = 0, frames_per_block = 8, blocks_per_stripe = 1,
                  queue = 2, chunk_frames = 0, shard_chunks = 0, depth = 0, zarr_threads = 0;
    std::uint64_t frame_bytes = 0;
    Compression compression;
    std::uint8_t level = 1, shuffle = 1;
    bool direct_io = false;
    double wait_seconds = 120;
    Json document;
};

// The array's axes: the frame axis, which grows, then the frame's own. acquire-zarr takes the
// last two for an image's height and width, so a frame of one axis is stored as [1, n].
std::vector<std::uint64_t> image_axes(const Description &stream)
{
    std::vector<std::uint64_t> axes(stream.shape.begin(), stream.shape.end());
    if(axes.empty())
        throw std::invalid_argument("a stream without a shape has no Zarr image");
    if(axes.size() == 1)
        axes.insert(axes.begin(), 1);
    for(auto extent : axes)
        if(!extent || extent > UINT32_MAX)
            throw std::invalid_argument("a frame axis of " + std::to_string(extent) +
                                        " elements is not one acquire-zarr takes");
    return axes;
}

// One part: a Zarr v3 store holding /images, [frame, ...image], sharded along the frames. The
// stream indices of its frames go into the array's attributes at the end, as runs.
class Part
{
  public:
    Part(std::filesystem::path path, const Description &stream, const Options &o,
         std::uint32_t chunk_frames, std::uint32_t shard_chunks)
        : path_(std::move(path))
    {
        if(std::filesystem::exists(path_))
            throw std::runtime_error(path_.string() + " exists");
        auto [type, element_bytes] = zarr_type(stream.element);
        auto axes = image_axes(stream);
        static const char *const k_names[] = {"z", "y", "x"};
        names_.push_back("frame");
        for(std::size_t i = 0; i < axes.size(); ++i)
            names_.push_back(axes.size() <= 3 ? k_names[3 - axes.size() + i]
                                              : "axis_" + std::to_string(i));
        dimensions_.push_back({names_[0].c_str(), ZarrDimensionType_Time, 0, chunk_frames,
                               shard_chunks, nullptr, 1.0});
        frame_bytes_ = element_bytes;
        for(std::size_t i = 0; i < axes.size(); ++i)
        {
            auto extent = std::uint32_t(axes[i]);
            dimensions_.push_back(
                {names_[i + 1].c_str(), ZarrDimensionType_Space, extent, extent, 1, nullptr, 1.0});
            frame_bytes_ *= extent;
        }
        compression_ = {o.compression.compressor, o.compression.codec, o.level, o.shuffle};
        ZarrArraySettings array{};
        array.output_key = "images";
        array.compression_settings =
            o.compression.compressor == ZarrCompressor_None ? nullptr : &compression_;
        array.dimensions = dimensions_.data();
        array.dimension_count = dimensions_.size();
        array.data_type = type;
        array.is_ngff = false; // a plain array: the stitched store is not OME
        ZarrStreamSettings settings{};
        settings.store_path = path_.c_str();
        settings.max_threads = o.zarr_threads;
        settings.overwrite = false;
        settings.arrays = &array;
        settings.array_count = 1;
        check(ZarrStreamSettings_estimate_max_memory_usage(&settings, &memory_estimate_),
              "ZarrStreamSettings_estimate_max_memory_usage");
        stream_ = ZarrStream_create(&settings);
        if(!stream_)
            throw std::runtime_error("ZarrStream_create " + path_.string() +
                                     " failed: see acquire-zarr's log");
    }
    Part(const Part &) = delete;
    Part &operator=(const Part &) = delete;
    ~Part()
    {
        if(stream_)
            ZarrStream_destroy(stream_);
    }

    // Frames adjacent in the receive ring, after the part's last. Returns once they are copied.
    void write(const void *frames, std::size_t count)
    {
        std::size_t bytes = count * frame_bytes_, out = 0;
        check(ZarrStream_append(stream_, frames, bytes, &out, "images"), "ZarrStream_append");
        if(out != bytes)
            throw std::runtime_error("ZarrStream_append took " + std::to_string(out) + " of " +
                                     std::to_string(bytes) + " bytes");
        std::size_t usage = 0;
        if(ZarrStream_get_current_memory_usage(stream_, &usage) == ZarrStatusCode_Success)
            memory_peak_ = std::max(memory_peak_, usage);
    }

    // Adds the attributes, then waits for every frame to be written and each shard to be durable.
    void close(const Json &attributes)
    {
        auto *stream = std::exchange(stream_, nullptr);
        auto written = ZarrStream_write_custom_metadata(stream, "images", "tango_ucx",
                                                        attributes.dump().c_str());
        auto closed = ZarrStream_close(stream);
        check(written, "ZarrStream_write_custom_metadata");
        check(closed, "ZarrStream_close");
    }

    const std::filesystem::path &path() const
    {
        return path_;
    }
    std::uint64_t frame_bytes() const
    {
        return frame_bytes_;
    }
    std::size_t memory_estimate() const
    {
        return memory_estimate_;
    }
    std::size_t memory_peak() const
    {
        return memory_peak_;
    }

  private:
    std::filesystem::path path_;
    std::vector<std::string> names_;
    std::vector<ZarrDimensionProperties> dimensions_;
    ZarrCompressionSettings compression_{};
    std::uint64_t frame_bytes_ = 0;
    std::size_t memory_estimate_ = 0, memory_peak_ = 0;
    ZarrStream *stream_ = nullptr;
};

std::filesystem::path part_path(const std::string &prefix, std::uint32_t part)
{
    char suffix[32];
    std::snprintf(suffix, sizeof(suffix), ".part_%03u.zarr", part);
    return prefix + suffix;
}

// Consecutive values as [first, count] runs.
struct Runs
{
    Json runs = Json::array();
    std::uint64_t next = UINT64_MAX;
    void add(std::uint64_t value)
    {
        if(value == next)
            runs.back()[1] = runs.back()[1].get<std::uint64_t>() + 1;
        else
            runs.push_back(Json::array({value, 1}));
        next = value + 1;
    }
};

struct Writer
{
    std::uint32_t part_index = 0;
    std::optional<Subscription> subscription;
    std::optional<Part> part;
    Json attributes, result;
};

void write_frames(Writer &writer, const Options &options)
{
    char name[16];
    std::snprintf(name, sizeof(name), "zarr-write-%u", writer.part_index % 1000);
    pthread_setname_np(pthread_self(), name);
    auto &subscription = *writer.subscription;
    auto &part = *writer.part;
    const auto &stream = subscription.description();
    const std::size_t record_stride =
        16 + ((std::size_t(stream.field_bytes) + 7) & ~std::size_t(7));
    const bool from_file = stream.field_bytes == sizeof(bench::FileFields);
    Runs indices, sources;
    std::map<std::int64_t, std::uint64_t> bytes_by_tenth; // 100 ms intervals of the Unix clock
    std::uint64_t frames = 0, batches = 0, partial = 0;
    std::int64_t first_receive = 0, last_append = 0, append_ns = 0, waiting_since = bench::now_ns();
    std::string failure;
    try
    {
        for(;;)
        {
            auto batch = subscription.read(Deadline::clock::now() + 1s);
            if(!batch)
            {
                if(subscription.outcome())
                    break;
                if(bench::now_ns() - waiting_since > std::int64_t(options.wait_seconds * 1e9))
                    throw std::runtime_error("no frame for " +
                                             std::to_string(options.wait_seconds) +
                                             " s: " + subscription.health().failure);
                continue;
            }
            if(!first_receive)
                first_receive = unix_ns();
            auto count = batch->frames();
            const auto *records = static_cast<const std::byte *>(batch->records());
            for(std::size_t i = 0; i < count; ++i)
            {
                indices.add(batch->index(i));
                if(from_file)
                {
                    bench::FileFields fields;
                    std::memcpy(&fields, records + i * record_stride + 16, sizeof(fields));
                    sources.add(fields.source_frame);
                }
            }
            // A batch ends where the ring wraps, so its frames follow one another.
            if(batch->payload(count - 1).data() !=
               static_cast<const std::byte *>(batch->payload()) + (count - 1) * part.frame_bytes())
                throw std::logic_error("a batch whose frames are not adjacent");
            auto from = bench::now_ns();
            part.write(batch->payload(), count);
            append_ns += bench::now_ns() - from;
            batch.reset(); // the credit returns once the frames are copied
            last_append = unix_ns();
            waiting_since = bench::now_ns();
            frames += count;
            ++batches;
            partial += count < options.frames_per_block;
            bytes_by_tenth[last_append / 100'000'000] += count * part.frame_bytes();
        }
    }
    catch(const std::exception &error)
    {
        failure = error.what();
    }
    auto health = subscription.health();
    auto outcome = subscription.outcome();
    subscription.close(Deadline::clock::now() + 5s);
    auto closing = bench::now_ns();
    std::int64_t closed = 0;
    try
    {
        writer.attributes["frame_runs"] = std::move(indices.runs);
        if(from_file)
            writer.attributes["source_runs"] = std::move(sources.runs);
        part.close(writer.attributes);
        closed = unix_ns();
    }
    catch(const std::exception &error)
    {
        if(failure.empty())
            failure = error.what();
    }
    auto close_seconds = double(bench::now_ns() - closing) / 1e9;
    Json tenths = Json::object();
    for(const auto &[tenth, bytes] : bytes_by_tenth)
        tenths[std::to_string(tenth)] = bytes;
    writer.result = {{"part", std::filesystem::absolute(part.path()).string()},
                     {"part_index", writer.part_index},
                     {"frames", frames},
                     {"bytes", frames * part.frame_bytes()},
                     {"batches", batches},
                     {"partial_batches", partial},
                     {"first_receive_unix_ns", first_receive},
                     {"last_append_unix_ns", last_append},
                     {"closed_unix_ns", closed},
                     {"append_seconds", double(append_ns) / 1e9},
                     {"close_seconds", close_seconds},
                     {"memory_estimate_bytes", part.memory_estimate()},
                     {"memory_peak_bytes", part.memory_peak()},
                     {"bytes_by_tenth_second", tenths},
                     {"outcome", outcome == Outcome::End      ? "end"
                                 : outcome == Outcome::Failed ? "failed"
                                 : outcome                    ? "cancelled"
                                                              : "none"},
                     {"health", bench::health_json(health)},
                     {"failure", failure}};
}

Json write(const Options &options)
{
    auto started = bench::now_ns();
    auto cpu = bench::thread_cpu_ns();
    // acquire-zarr reads this once, at its first file open.
    if(options.direct_io)
        setenv("ZARR_DIRECT_IO", "1", 1);
    const char *direct = std::getenv("ZARR_DIRECT_IO");
    char host[256] = {};
    gethostname(host, sizeof(host) - 1);

    const std::uint32_t range = options.frames_per_block * options.blocks_per_stripe;
    const std::uint32_t depth = range * options.queue;
    const std::uint32_t chunk =
        options.chunk_frames ? options.chunk_frames : options.frames_per_block;
    if(range % chunk)
        throw std::invalid_argument("--chunk-frames " + std::to_string(chunk) +
                                    " does not divide the range of " + std::to_string(range));
    const std::uint32_t shard = options.shard_chunks ? options.shard_chunks : range / chunk;
    if(range % (chunk * shard))
        throw std::invalid_argument("a shard of " + std::to_string(chunk * shard) +
                                    " frames does not divide the range of " +
                                    std::to_string(range) +
                                    ": the parts could not be stitched into one store");

    auto proxy = bench::connect(options.device, options.wait_seconds);
    SubscriptionOptions subscribe;
    subscribe.batch = options.frames_per_block;
    subscribe.max_wait = 1s; // a block is whole unless the stream ends
    subscribe.label = "tango_ucx_zarr_writer";
    std::vector<Writer> writers(options.threads);
    auto pullers =
        bench::open_pullers(proxy, options.threads, range, depth, options.frame_bytes, subscribe);
    for(std::uint32_t t = 0; t < options.threads; ++t)
        writers[t].subscription.emplace(std::move(pullers[t]));
    const auto &stream = writers[0].subscription->description();
    for(std::uint32_t t = 0; t < options.threads; ++t)
    {
        auto &writer = writers[t];
        writer.part_index = options.first_part + t;
        writer.part.emplace(part_path(options.prefix, writer.part_index), stream, options, chunk,
                            shard);
        if(writer.part->frame_bytes() != options.frame_bytes)
            throw std::runtime_error("the stream's frames are " +
                                     std::to_string(writer.part->frame_bytes()) + " bytes, not " +
                                     std::to_string(options.frame_bytes));
        writer.attributes = {{"part_index", writer.part_index},
                             {"frame_shape", stream.shape},
                             {"frames_per_block", options.frames_per_block},
                             {"blocks_per_stripe", options.blocks_per_stripe},
                             {"process_threads", options.threads},
                             {"host", std::string(host)}};
    }

    // The driver starts the publisher once every writer of every process has said this.
    std::printf("{\"ready\": %u}\n", options.threads);
    std::fflush(stdout);
    auto writing = bench::now_ns();
    {
        std::vector<std::jthread> threads;
        for(auto &writer : writers)
            threads.emplace_back([&writer, &options] { write_frames(writer, options); });
    }
    Json results = Json::array();
    bool ok = true;
    for(auto &writer : writers)
    {
        ok = ok && writer.result["failure"].get<std::string>().empty() &&
             writer.result["outcome"] == "end";
        results.push_back(std::move(writer.result));
    }
    return {{"ok", ok},
            {"host", std::string(host)},
            {"pid", ::getpid()},
            {"threads", options.threads},
            {"range", range},
            {"subscription",
             {{"depth", depth}, {"budget", subscribe.budget}, {"batch", subscribe.batch}}},
            {"zarr",
             {{"release", ACQUIRE_ZARR_RELEASE},
              {"api_version", Zarr_get_api_version()},
              {"chunk_frames", chunk},
              {"shard_chunks", shard},
              {"compression", options.compression.name},
              {"level", options.level},
              {"shuffle", options.shuffle},
              {"max_threads", options.zarr_threads},
              {"direct_io", direct ? direct : ""}}},
            {"seconds_joining", double(writing - started) / 1e9},
            {"seconds_writing", double(bench::now_ns() - writing) / 1e9},
            {"cpu_seconds", bench::cpu_seconds_since(cpu)},
            {"writers", results}};
}

std::uint32_t count_of(const char *text, const char *option, std::uint32_t least)
{
    std::size_t used = 0;
    auto value = std::stoul(text, &used);
    if(text[used] || value < least || value > 1'000'000)
        throw std::invalid_argument(std::string(option) + " takes a count of " +
                                    std::to_string(least) + " or more");
    return std::uint32_t(value);
}

Options parse(int argc, char **argv)
{
    Options o;
    std::vector<std::string> words;
    for(int i = 1; i < argc; ++i)
    {
        std::string_view a = argv[i];
        auto value = [&]
        {
            if(i + 1 >= argc)
                throw std::invalid_argument(std::string(a) + " needs a value");
            return argv[++i];
        };
        if(a == "--threads")
            o.threads = count_of(value(), "--threads", 1);
        else if(a == "--first-part")
            o.first_part = count_of(value(), "--first-part", 0);
        else if(a == "--frames-per-block")
            o.frames_per_block = count_of(value(), "--frames-per-block", 1);
        else if(a == "--blocks-per-stripe")
            o.blocks_per_stripe = count_of(value(), "--blocks-per-stripe", 1);
        else if(a == "--queue")
            o.queue = count_of(value(), "--queue", 1);
        else if(a == "--chunk-frames")
            o.chunk_frames = count_of(value(), "--chunk-frames", 1);
        else if(a == "--shard-chunks")
            o.shard_chunks = count_of(value(), "--shard-chunks", 1);
        else if(a == "--compression")
            o.compression = compression_of(value());
        else if(a == "--level")
            o.level = std::uint8_t(std::min(count_of(value(), "--level", 0), 255u));
        else if(a == "--shuffle")
            o.shuffle = std::uint8_t(std::min(count_of(value(), "--shuffle", 0), 2u));
        else if(a == "--zarr-threads")
            o.zarr_threads = count_of(value(), "--zarr-threads", 0);
        else if(a == "--direct-io")
            o.direct_io = true;
        else if(a == "--frame-bytes")
            o.frame_bytes = std::stoull(value());
        else if(a == "--depth")
            o.depth = count_of(value(), "--depth", 1);
        else if(a == "--wait")
            o.wait_seconds = std::stod(value());
        else if(a.starts_with("--"))
            throw std::invalid_argument("unknown option " + std::string(a));
        else
            words.emplace_back(a);
    }
    if(words.size() < 2)
        throw std::invalid_argument("a command and a device are needed");
    o.command = words[0];
    o.device = words[1];
    if(o.command == "write")
    {
        if(words.size() != 3 || !o.frame_bytes)
            throw std::invalid_argument("write takes a device, a part prefix and --frame-bytes");
        o.prefix = words[2];
    }
    else if(o.command == "configure")
    {
        if(words.size() != 3 || !o.depth)
            throw std::invalid_argument("configure takes a device, --depth and a JSON document");
        o.document = Json::parse(words[2]);
    }
    else if((o.command != "start" && o.command != "result") || words.size() != 2)
        throw std::invalid_argument("the commands are configure, start, result and write");
    if(o.compression.compressor != ZarrCompressor_Blosc1)
        o.shuffle = 0; // a blosc setting: acquire-zarr refuses it with plain zstd
    return o;
}
} // namespace

int main(int argc, char **argv)
{
    Options options;
    try
    {
        options = parse(argc, argv);
    }
    catch(const std::exception &error)
    {
        std::fprintf(stderr, "tango_ucx_zarr_writer: %s\n%s", error.what(), k_usage);
        return 2;
    }
    try
    {
        Json out;
        if(options.command == "write")
            out = write(options);
        else if(options.command == "configure")
            out = bench::configure_publisher(options.device, options.wait_seconds, options.depth,
                                             options.document);
        else if(options.command == "start")
        {
            auto proxy = bench::connect(options.device, options.wait_seconds);
            proxy.command_inout("Start");
            out = {{"started", true}};
        }
        else
            out = bench::publisher_result(options.device, options.wait_seconds);
        std::puts(out.dump().c_str());
        return out.value("ok", true) ? 0 : 1;
    }
    catch(const Tango::DevFailed &error)
    {
        Tango::Except::print_exception(error);
    }
    catch(const std::exception &error)
    {
        std::fprintf(stderr, "tango_ucx_zarr_writer: %s\n", error.what());
    }
    return 1;
}
