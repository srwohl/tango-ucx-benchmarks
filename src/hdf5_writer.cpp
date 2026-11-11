// SPDX-FileCopyrightText: 2026 Copyright contributors to the tango-ucx project
// SPDX-License-Identifier: LGPL-3.0-or-later

// tango_ucx_hdf5_writer: frames of a tango_ucx_perf_publisher written to HDF5 shard files.
// Ported from tango-bulk's example_hdf5_client. Each writer thread is a puller of the publisher's
// pull set and owns one shard file: a range of frames at a time goes to one writer, which writes
// it with H5Dwrite straight from its receive ring and then releases it, so the publisher's
// credit returns only once the write is done. Threads in one process share the HDF5 library and
// its lock; separate processes, on this host or others, do not. hdf5_write.py runs the commands
// and makes the master file from the shards' frame indices.
//
//   configure <device> --depth <d> <json>   the publisher's Configure, at the source depth d
//   start <device>                          its Start
//   result <device>                         its Result, once the run is over
//   write <device> <prefix> [options]       join the pull set, say ready, write until the end
#include "common.h"
#include "control.h"

#include <tango-ucx/tango.h>
#include <tango/tango.h>

#include <hdf5.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <map>
#include <mutex>
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

const char *const k_usage =
    "usage: tango_ucx_hdf5_writer configure <device> --depth <slots> <json>\n"
    "       tango_ucx_hdf5_writer start <device>\n"
    "       tango_ucx_hdf5_writer result <device>\n"
    "       tango_ucx_hdf5_writer write <device> <prefix> [options]\n"
    "  write: shard n is <prefix>.part_<nnn>.h5; existing shards are refused\n"
    "  --frame-bytes <n>        the stream's frame size, as configure gave it (needed)\n"
    "  --threads <t>            writer threads in this process, each its own shard (1)\n"
    "  --first-shard <n>        the shard of the first thread (0)\n"
    "  --frames-per-block <f>   frames in one H5Dwrite: the subscription's batch (8)\n"
    "  --blocks-per-stripe <b>  blocks a writer takes at its turn: the pull range is f * b (1)\n"
    "  --queue <q>              ranges each receive ring holds, so the next arrives while one is\n"
    "                           written (2)\n"
    "  --chunk-frames <c>       frames in an HDF5 chunk (the block)\n"
    "  --no-sync                leave out the fsync of each shard after it is closed\n"
    "  --wait <seconds>         the longest wait for the device or for a frame (120)\n";

using bench::unix_ns;

// Every HDF5 call of this process is made holding this, so a thread's wait for the library is
// measured apart from its work. The thread-safe library holds a lock of its own around each call:
// this one changes no order, it makes the wait visible.
std::mutex hdf5_mutex;
struct Hdf5Time
{
    std::int64_t waited_ns = 0, held_ns = 0;
};
template <class Work> auto with_hdf5(Hdf5Time &time, Work &&work)
{
    auto asked = bench::now_ns();
    std::lock_guard lock(hdf5_mutex);
    auto got = bench::now_ns();
    time.waited_ns += got - asked;
    struct Held
    {
        Hdf5Time &time;
        std::int64_t from;
        ~Held()
        {
            time.held_ns += bench::now_ns() - from;
        }
    } held{time, got};
    return work();
}

void check(herr_t status, const char *call)
{
    if(status < 0)
        throw std::runtime_error(std::string(call) + " failed");
}
hid_t checked(hid_t id, const char *call)
{
    if(id < 0)
        throw std::runtime_error(std::string(call) + " failed");
    return id;
}
struct Handle
{
    hid_t id = H5I_INVALID_HID;
    herr_t (*close)(hid_t) = nullptr;
    Handle() = default;
    Handle(hid_t i, herr_t (*c)(hid_t), const char *call) : id(checked(i, call)), close(c) {}
    Handle(Handle &&o) noexcept : id(std::exchange(o.id, H5I_INVALID_HID)), close(o.close) {}
    Handle &operator=(Handle &&o) noexcept
    {
        std::swap(id, o.id);
        std::swap(close, o.close);
        return *this;
    }
    ~Handle()
    {
        reset();
    }
    herr_t reset()
    {
        herr_t status = 0;
        if(id >= 0 && close)
            status = close(std::exchange(id, H5I_INVALID_HID));
        return status;
    }
};

// The memory and file types of an element: the file is little-endian, as the stream is.
std::pair<hid_t, hid_t> hdf5_types(Element element)
{
    switch(element)
    {
        case Element::U8:
            return {H5T_NATIVE_UINT8, H5T_STD_U8LE};
        case Element::U16:
            return {H5T_NATIVE_UINT16, H5T_STD_U16LE};
        case Element::U32:
            return {H5T_NATIVE_UINT32, H5T_STD_U32LE};
        case Element::U64:
            return {H5T_NATIVE_UINT64, H5T_STD_U64LE};
        case Element::I8:
            return {H5T_NATIVE_INT8, H5T_STD_I8LE};
        case Element::I16:
            return {H5T_NATIVE_INT16, H5T_STD_I16LE};
        case Element::I32:
            return {H5T_NATIVE_INT32, H5T_STD_I32LE};
        case Element::I64:
            return {H5T_NATIVE_INT64, H5T_STD_I64LE};
        case Element::F32:
            return {H5T_NATIVE_FLOAT, H5T_IEEE_F32LE};
        case Element::F64:
            return {H5T_NATIVE_DOUBLE, H5T_IEEE_F64LE};
        case Element::Bytes:
            break;
    }
    throw std::invalid_argument("an opaque stream has no HDF5 image type");
}

void write_attribute(hid_t object, const char *name, std::uint64_t value)
{
    Handle space(H5Screate(H5S_SCALAR), H5Sclose, "H5Screate");
    Handle attribute(H5Acreate2(object, name, H5T_STD_U64LE, space.id, H5P_DEFAULT, H5P_DEFAULT),
                     H5Aclose, "H5Acreate2");
    check(H5Awrite(attribute.id, H5T_NATIVE_UINT64, &value), "H5Awrite");
}
void write_attribute(hid_t object, const char *name, const std::string &value)
{
    Handle type(H5Tcopy(H5T_C_S1), H5Tclose, "H5Tcopy");
    check(H5Tset_size(type.id, value.size() + 1), "H5Tset_size");
    Handle space(H5Screate(H5S_SCALAR), H5Sclose, "H5Screate");
    Handle attribute(H5Acreate2(object, name, type.id, space.id, H5P_DEFAULT, H5P_DEFAULT),
                     H5Aclose, "H5Acreate2");
    check(H5Awrite(attribute.id, type.id, value.c_str()), "H5Awrite");
}
void write_vector(hid_t file, const char *name, const std::vector<std::uint64_t> &values)
{
    hsize_t count = values.size();
    Handle space(H5Screate_simple(1, &count, nullptr), H5Sclose, "H5Screate_simple");
    Handle dataset(
        H5Dcreate2(file, name, H5T_STD_U64LE, space.id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
        H5Dclose, "H5Dcreate2");
    if(count)
        check(H5Dwrite(dataset.id, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()),
              "H5Dwrite");
}

// One shard: /images, [frame, ...image], grown a chunk at a time as the writer is given frames.
// Which frames those are is not known before the run, so /frame_index records the stream index of
// each, and /source_frame the dataset frame each was read from when the publisher read a file.
class Shard
{
  public:
    Shard(std::filesystem::path path, const Description &stream, std::uint64_t chunk_frames,
          const Json &attributes)
        : path_(std::move(path)), shape_(stream.shape.begin(), stream.shape.end()),
          types_(hdf5_types(stream.element))
    {
        frame_bytes_ = H5Tget_size(types_.second);
        for(auto extent : shape_)
            frame_bytes_ *= extent;
        file_ = Handle(H5Fcreate(path_.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT), H5Fclose,
                       ("H5Fcreate " + path_.string()).c_str());
        std::vector<hsize_t> dimensions{0}, most{H5S_UNLIMITED}, chunk{chunk_frames};
        dimensions.insert(dimensions.end(), shape_.begin(), shape_.end());
        most.insert(most.end(), shape_.begin(), shape_.end());
        chunk.insert(chunk.end(), shape_.begin(), shape_.end());
        Handle space(H5Screate_simple(int(dimensions.size()), dimensions.data(), most.data()),
                     H5Sclose, "H5Screate_simple");
        Handle creation(H5Pcreate(H5P_DATASET_CREATE), H5Pclose, "H5Pcreate");
        check(H5Pset_chunk(creation.id, int(chunk.size()), chunk.data()), "H5Pset_chunk");
        // Whole chunks are written: filling them first would write each byte twice.
        check(H5Pset_fill_time(creation.id, H5D_FILL_TIME_NEVER), "H5Pset_fill_time");
        images_ = Handle(H5Dcreate2(file_.id, "/images", types_.second, space.id, H5P_DEFAULT,
                                    creation.id, H5P_DEFAULT),
                         H5Dclose, "H5Dcreate2 /images");
        for(const auto &[name, value] : attributes.items())
            if(value.is_string())
                write_attribute(file_.id, name.c_str(), value.get<std::string>());
            else
                write_attribute(file_.id, name.c_str(), value.get<std::uint64_t>());
    }

    // Frames that are adjacent in the receive ring, written after the shard's last. The memory
    // is described as the dense array it is: as one 1-D selection, HDF5 can take minutes to set
    // up a block of large frames.
    void write(const void *frames, std::size_t count)
    {
        if(written_ + count > capacity_)
        {
            capacity_ = std::max<hsize_t>(written_ + count, capacity_ * 2);
            std::vector<hsize_t> extent{capacity_};
            extent.insert(extent.end(), shape_.begin(), shape_.end());
            check(H5Dset_extent(images_.id, extent.data()), "H5Dset_extent");
        }
        std::vector<hsize_t> start(shape_.size() + 1, 0), counts{hsize_t(count)};
        start.front() = written_;
        counts.insert(counts.end(), shape_.begin(), shape_.end());
        Handle file_space(H5Dget_space(images_.id), H5Sclose, "H5Dget_space");
        check(H5Sselect_hyperslab(file_space.id, H5S_SELECT_SET, start.data(), nullptr,
                                  counts.data(), nullptr),
              "H5Sselect_hyperslab");
        Handle memory_space(H5Screate_simple(int(counts.size()), counts.data(), nullptr), H5Sclose,
                            "H5Screate_simple");
        check(
            H5Dwrite(images_.id, types_.first, memory_space.id, file_space.id, H5P_DEFAULT, frames),
            "H5Dwrite");
        written_ += count;
    }

    // Trims /images to the frames written and adds the indices.
    void close(const std::vector<std::uint64_t> &indices, const std::vector<std::uint64_t> &sources)
    {
        std::vector<hsize_t> extent{written_};
        extent.insert(extent.end(), shape_.begin(), shape_.end());
        check(H5Dset_extent(images_.id, extent.data()), "H5Dset_extent");
        write_vector(file_.id, "/frame_index", indices);
        if(!sources.empty())
            write_vector(file_.id, "/source_frame", sources);
        check(images_.reset(), "H5Dclose");
        check(file_.reset(), "H5Fclose");
    }

    const std::filesystem::path &path() const
    {
        return path_;
    }
    std::uint64_t frame_bytes() const
    {
        return frame_bytes_;
    }

  private:
    std::filesystem::path path_;
    std::vector<hsize_t> shape_;
    std::pair<hid_t, hid_t> types_;
    std::uint64_t frame_bytes_ = 0;
    hsize_t written_ = 0, capacity_ = 0;
    Handle file_, images_;
};

struct Options
{
    std::string command, device, prefix;
    std::uint32_t threads = 1, first_shard = 0, frames_per_block = 8, blocks_per_stripe = 1,
                  queue = 2, chunk_frames = 0, depth = 0;
    std::uint64_t frame_bytes = 0;
    bool sync = true;
    double wait_seconds = 120;
    Json document;
};

std::filesystem::path shard_path(const std::string &prefix, std::uint32_t shard)
{
    char suffix[32];
    std::snprintf(suffix, sizeof(suffix), ".part_%03u.h5", shard);
    return prefix + suffix;
}

struct Writer
{
    std::uint32_t shard_index = 0;
    std::optional<Subscription> subscription;
    std::optional<Shard> shard;
    Json result;
};

void write_frames(Writer &writer, const Options &options)
{
    char name[16];
    std::snprintf(name, sizeof(name), "hdf5-write-%u", writer.shard_index % 1000);
    pthread_setname_np(pthread_self(), name);
    auto &subscription = *writer.subscription;
    auto &shard = *writer.shard;
    const auto &stream = subscription.description();
    const std::size_t record_stride =
        16 + ((std::size_t(stream.field_bytes) + 7) & ~std::size_t(7));
    const bool from_file = stream.field_bytes == sizeof(bench::FileFields);
    Hdf5Time hdf5;
    std::vector<std::uint64_t> indices, sources;
    std::map<std::int64_t, std::uint64_t> bytes_by_tenth; // 100 ms intervals of the Unix clock
    std::uint64_t frames = 0, batches = 0, partial = 0;
    std::int64_t first_receive = 0, last_write = 0, waiting_since = bench::now_ns();
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
                indices.push_back(batch->index(i));
                if(from_file)
                {
                    bench::FileFields fields;
                    std::memcpy(&fields, records + i * record_stride + 16, sizeof(fields));
                    sources.push_back(fields.source_frame);
                }
            }
            // A batch ends where the ring wraps, so its frames follow one another.
            if(batch->payload(count - 1).data() !=
               static_cast<const std::byte *>(batch->payload()) + (count - 1) * shard.frame_bytes())
                throw std::logic_error("a batch whose frames are not adjacent");
            with_hdf5(hdf5, [&] { shard.write(batch->payload(), count); });
            batch.reset(); // the credit returns once the write is done
            last_write = unix_ns();
            waiting_since = bench::now_ns();
            frames += count;
            ++batches;
            partial += count < options.frames_per_block;
            bytes_by_tenth[last_write / 100'000'000] += count * shard.frame_bytes();
        }
    }
    catch(const std::exception &error)
    {
        failure = error.what();
    }
    auto health = subscription.health();
    auto outcome = subscription.outcome();
    subscription.close(Deadline::clock::now() + 5s);
    Hdf5Time closing;
    double sync_seconds = 0;
    try
    {
        with_hdf5(closing, [&] { shard.close(indices, sources); });
        if(options.sync)
        {
            auto from = bench::now_ns();
            int fd = ::open(shard.path().c_str(), O_RDONLY | O_CLOEXEC);
            if(fd < 0 || ::fsync(fd) != 0)
                throw std::runtime_error("fsync " + shard.path().string());
            ::close(fd);
            sync_seconds = double(bench::now_ns() - from) / 1e9;
        }
    }
    catch(const std::exception &error)
    {
        if(failure.empty())
            failure = error.what();
    }
    const auto closed = unix_ns(); // closed, and durable unless --no-sync
    Json tenths = Json::object();
    for(const auto &[tenth, bytes] : bytes_by_tenth)
        tenths[std::to_string(tenth)] = bytes;
    writer.result = {{"shard", std::filesystem::absolute(shard.path()).string()},
                     {"shard_index", writer.shard_index},
                     {"frames", frames},
                     {"bytes", frames * shard.frame_bytes()},
                     {"batches", batches},
                     {"partial_batches", partial},
                     {"first_receive_unix_ns", first_receive},
                     {"last_write_unix_ns", last_write},
                     {"closed_unix_ns", closed},
                     {"write_seconds", double(hdf5.held_ns) / 1e9},
                     {"hdf5_wait_seconds", double(hdf5.waited_ns) / 1e9},
                     {"close_seconds", double(closing.held_ns) / 1e9},
                     {"close_wait_seconds", double(closing.waited_ns) / 1e9},
                     {"sync_seconds", sync_seconds},
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
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    hbool_t threadsafe = false;
    H5is_library_threadsafe(&threadsafe);
    unsigned major = 0, minor = 0, release = 0;
    H5get_libversion(&major, &minor, &release);
    char host[256] = {};
    gethostname(host, sizeof(host) - 1);

    const std::uint32_t range = options.frames_per_block * options.blocks_per_stripe;
    const std::uint32_t depth = range * options.queue;
    auto proxy = bench::connect(options.device, options.wait_seconds);
    SubscriptionOptions subscribe;
    subscribe.batch = options.frames_per_block;
    subscribe.max_wait = 1s; // a block is whole unless the stream ends
    subscribe.label = "tango_ucx_hdf5_writer";

    std::vector<Writer> writers(options.threads);
    auto pullers =
        bench::open_pullers(proxy, options.threads, range, depth, options.frame_bytes, subscribe);
    const auto budget = subscribe.budget;
    for(std::uint32_t t = 0; t < options.threads; ++t)
        writers[t].subscription.emplace(std::move(pullers[t]));
    const auto &stream = writers[0].subscription->description();
    auto chunk = options.chunk_frames ? options.chunk_frames : options.frames_per_block;
    for(std::uint32_t t = 0; t < options.threads; ++t)
    {
        writers[t].shard_index = options.first_shard + t;
        Json attributes{{"shard_index", writers[t].shard_index},
                        {"frames_per_block", options.frames_per_block},
                        {"blocks_per_stripe", options.blocks_per_stripe},
                        {"process_threads", options.threads},
                        {"host", std::string(host)}};
        Hdf5Time opening;
        with_hdf5(opening,
                  [&]
                  {
                      writers[t].shard.emplace(shard_path(options.prefix, writers[t].shard_index),
                                               stream, chunk, attributes);
                  });
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
            {"subscription", {{"depth", depth}, {"budget", budget}, {"batch", subscribe.batch}}},
            {"chunk_frames", chunk},
            {"hdf5",
             {{"version",
               std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(release)},
              {"threadsafe", bool(threadsafe)}}},
            {"seconds_joining", double(writing - started) / 1e9},
            {"seconds_writing", double(bench::now_ns() - writing) / 1e9},
            {"cpu_seconds", bench::cpu_seconds_since(cpu)},
            {"writers", results}};
}

std::uint32_t positive(const char *text, const char *option)
{
    std::size_t used = 0;
    auto value = std::stoul(text, &used);
    if(text[used] || !value || value > 1'000'000)
        throw std::invalid_argument(std::string(option) + " takes a positive count");
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
            o.threads = positive(value(), "--threads");
        else if(a == "--first-shard")
            o.first_shard = std::uint32_t(std::stoul(value()));
        else if(a == "--frames-per-block")
            o.frames_per_block = positive(value(), "--frames-per-block");
        else if(a == "--blocks-per-stripe")
            o.blocks_per_stripe = positive(value(), "--blocks-per-stripe");
        else if(a == "--queue")
            o.queue = positive(value(), "--queue");
        else if(a == "--chunk-frames")
            o.chunk_frames = positive(value(), "--chunk-frames");
        else if(a == "--frame-bytes")
            o.frame_bytes = std::stoull(value());
        else if(a == "--depth")
            o.depth = positive(value(), "--depth");
        else if(a == "--no-sync")
            o.sync = false;
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
            throw std::invalid_argument("write takes a device, a shard prefix and --frame-bytes");
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
        std::fprintf(stderr, "tango_ucx_hdf5_writer: %s\n%s", error.what(), k_usage);
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
        std::fprintf(stderr, "tango_ucx_hdf5_writer: %s\n", error.what());
    }
    return 1;
}
