// SPDX-FileCopyrightText: 2026 Copyright contributors to the tango-ucx project
// SPDX-License-Identifier: LGPL-3.0-or-later

// tango_ucx_perf_publisher: a Tango device that publishes what tango_ucx_perftest tells it to.
//   Configure  a JSON document in: frame_bytes, budget, frames, rate, verify, corrupt_every,
//              and optionally hdf5: {file, dataset} and read_ahead (8).
//              Out: the source slots the budget gave.
//   Describe   a JSON document in: hdf5: {file, dataset}. Out: the file's frames, as Result has.
//   Start      publishes the frames at rate (0: as fast as slots are granted), and finishes.
//   Result     what the run took, as a JSON document.
//
// Generated frames are filled once a slot unless verified. Frames from an HDF5 file are read
// into their slot before each publish, the dataset's frames in order and from the first again.
// With read_ahead a reader thread does that, up to read_ahead frames before they are published,
// so a slow read does not hold up a publish; with zero the publishing thread reads each frame.
#include "common.h"
#include "memory.h"
#ifdef BENCHMARKS_HDF5
#include "hdf5_source.h"
#endif

#include <tango-ucx/tango.h>
#include <tango/tango.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <sys/prctl.h>
#include <thread>
#include <unordered_set>

namespace
{
using namespace TangoUcx;
using namespace std::chrono_literals;
using bench::Json;

#ifdef BENCHMARKS_HDF5
// Reads frame `index` of a run into its slot, with the fields that say which frame of the dataset
// it is and, verified, its checksum. Returns the nanoseconds spent reading.
std::int64_t read_into(Slot &slot, std::uint64_t index, bench::Hdf5Source &source, bool verify,
                       std::uint64_t corrupt_every)
{
    auto payload = slot.payload();
    bench::FileFields fields{index % source.frames(), 0};
    auto from = bench::now_ns();
    source.read(fields.source_frame, payload);
    auto spent = bench::now_ns() - from;
    if(verify)
    {
        fields.checksum = bench::checksum(payload.first(source.frame_bytes()));
        if(corrupt_every && index % corrupt_every == corrupt_every - 1)
            payload[source.frame_bytes() / 2] ^= std::byte{1}; // after the checksum: found
    }
    std::memcpy(slot.fields().data(), &fields, sizeof(fields));
    return spent;
}

// A thread that reads the run's frames into source slots ahead of their publish. The library
// sends slots in the order it granted them, and this thread takes them in that order, so the
// publishing thread takes the ready ones first in, first out.
class ReadAhead
{
  public:
    ReadAhead(Publisher &publisher, bench::Hdf5Source &source, std::uint64_t frames,
              std::uint32_t depth, bool verify, std::uint64_t corrupt_every)
        : publisher_(publisher), source_(source), frames_(frames), depth_(depth), verify_(verify),
          corrupt_every_(corrupt_every), thread_([this](std::stop_token stop) { run(stop); })
    {
    }
    // The next frame read, or nothing once the reader has ended: failure() says why, if it failed.
    std::optional<Slot> next(std::stop_token stop)
    {
        std::unique_lock lock(m_);
        ready_changed_.wait(lock, stop, [&] { return !ready_.empty() || ended_; });
        if(ready_.empty())
            return std::nullopt;
        auto slot = std::move(ready_.front());
        ready_.pop_front();
        ready_changed_.notify_all();
        return slot;
    }
    std::string failure()
    {
        std::lock_guard lock(m_);
        return failure_;
    }
    // What the reader's thread spent its time on. Call once next() has given every frame, or
    // the run has stopped.
    Json json()
    {
        std::lock_guard lock(m_);
        return {{"seconds_reading", double(reading_ns_) / 1e9},
                {"seconds_waiting_for_a_slot", double(waited_ns_) / 1e9},
                {"slots_refused", refused_},
                {"frames_read", read_}};
    }

  private:
    void run(std::stop_token stop)
    {
        pthread_setname_np(pthread_self(), "perf-reader");
        try
        {
            for(std::uint64_t index = 0; index < frames_ && !stop.stop_requested();)
            {
                {
                    std::unique_lock lock(m_);
                    if(!ready_changed_.wait(lock, stop, [&] { return ready_.size() < depth_; }))
                        break;
                }
                auto asked = bench::now_ns();
                auto slot = publisher_.acquire(Deadline::clock::now() + 100ms);
                auto waited = bench::now_ns() - asked;
                if(!slot)
                {
                    std::lock_guard lock(m_);
                    waited_ns_ += waited;
                    ++refused_;
                    if(failure_ = publisher_.health().failure; !failure_.empty())
                        break;
                    continue;
                }
                auto reading = read_into(*slot, index, source_, verify_, corrupt_every_);
                std::lock_guard lock(m_);
                waited_ns_ += waited;
                reading_ns_ += reading;
                ready_.push_back(std::move(*slot));
                read_ = ++index;
                ready_changed_.notify_all();
            }
        }
        catch(const std::exception &error)
        {
            std::lock_guard lock(m_);
            failure_ = error.what();
        }
        std::lock_guard lock(m_);
        ended_ = true;
        ready_changed_.notify_all();
    }

    Publisher &publisher_;
    bench::Hdf5Source &source_;
    const std::uint64_t frames_;
    const std::uint32_t depth_;
    const bool verify_;
    const std::uint64_t corrupt_every_;
    std::mutex m_;
    std::condition_variable_any ready_changed_;
    std::deque<Slot> ready_; // unpublished when the run stops: each is given back on destruction
    std::uint64_t read_ = 0, refused_ = 0;
    std::int64_t reading_ns_ = 0, waited_ns_ = 0;
    bool ended_ = false;
    std::string failure_;
    std::jthread thread_; // last: it stops and joins before the rest goes
};
#endif

class Device : public TANGO_BASE_CLASS
{
  public:
    Device(Tango::DeviceClass *device_class, const std::string &name)
        : TANGO_BASE_CLASS(device_class, name.c_str())
    {
        init_device();
    }
    ~Device() override
    {
        delete_device();
    }
    void init_device() override
    {
        set_state(Tango::ON);
    }
    std::string describe(const std::string &document)
    {
        std::lock_guard lock(commands);
        if(running)
            throw std::runtime_error("a run is in progress");
        return open_source(Json::parse(document)).dump();
    }
    void delete_device() override
    {
        frames_thread = {}; // asks it to stop and joins it, before the publisher goes
        detach_publisher(*this);
        publisher.reset();
    }
    std::string configure(const std::string &document)
    {
        std::lock_guard lock(commands);
        if(running)
            throw std::runtime_error("a run is in progress");
        auto wanted = Json::parse(document);
        auto target_rate = wanted.value("rate", std::uint64_t{0});
        if(target_rate > 1'000'000'000)
            throw std::invalid_argument("rate is at most one billion frames/s");
        PublisherLimits limits;
        limits.budget = wanted.at("budget").get<std::uint64_t>();
        limits.max_sessions = wanted.value("sessions", 1u);
        limits.allow_every = true;
        bool cuda = wanted.value("memory", std::string("host")) == "cuda";
        // The driver gives each role its physical GPU through CUDA_VISIBLE_DEVICES.
        limits.gpu = cuda ? 0 : -1;
        limits.progress_cpu = wanted.value("publisher_cpu", -1);
        auto frame_bytes = wanted.at("frame_bytes").get<std::uint64_t>();
        Description description{Element::U8, {frame_bytes}, 0, {}};
        source_json = {{"kind", "generated"}};
        if(wanted.contains("hdf5"))
        {
            if(cuda)
                throw std::invalid_argument("HDF5 frames are read into host source slots");
            source_json = open_source(wanted);
#ifdef BENCHMARKS_HDF5
            if(frame_bytes != source->frame_bytes())
                throw std::invalid_argument("frame_bytes is not the dataset's frame size, " +
                                            std::to_string(source->frame_bytes()));
            description = source->description();
#endif
        }
        else
            source.reset(); // a later generated run must not read the old file
        if(limits.progress_cpu >= 0)
        {
            // A pinned worker reserves its physical CPU until destruction. Budget trials
            // must retire that worker before creating its replacement on the same CPU.
            frames_thread = {};
            detach_publisher(*this);
            publisher.reset();
        }
        // Unpinned configurations can preserve the previous publisher on refusal.
        auto next = std::make_unique<Publisher>(std::move(description), limits);
        frames_thread = {};
        detach_publisher(*this);
        publisher = std::move(next);
        frames = wanted.at("frames").get<std::uint64_t>();
        rate = target_rate;
        verify = wanted.value("verify", false);
        gpu = cuda;
        producer_cpu = wanted.value("producer_cpu", -1);
        corrupt_every = wanted.value("corrupt_every", std::uint64_t{0});
        read_ahead = wanted.value("read_ahead", 8u);
        started = false;
        report = Json::object();
        report["source"] = source_json;
        attach_publisher(*this, *publisher);
        // Health is sampled by the loop: before the start every slot is free, and they are all
        // the slots there are.
        std::uint32_t slots = 0;
        for(auto until = bench::now_ns() + 2'000'000'000; !slots && bench::now_ns() < until;
            std::this_thread::sleep_for(1ms))
            slots = publisher->health().slots_free;
        return Json{{"slots", slots}}.dump();
    }
    void start()
    {
        std::lock_guard lock(commands);
        if(!publisher || started)
            throw std::runtime_error("Start needs a Configure of its own");
        started = running = true;
        publisher->ucx_start(); // the every subscription has joined
        frames_thread = std::jthread(
            [this](std::stop_token stop)
            {
                try
                {
                    publish(stop);
                }
                catch(const std::exception &error)
                {
                    publisher->finish();
                    std::lock_guard lock(commands);
                    report = {{"failure", error.what()}, {"source", source_json}};
                    running = false;
                }
            });
    }
    std::string result()
    {
        std::lock_guard lock(commands);
        Json document = report;
        document["running"] = running.load();
        if(publisher)
            document["health"] = bench::health_json(publisher->health());
        return document.dump();
    }

  private:
    // Opens the file a document's hdf5 names, keeping the one open when it is the same.
    Json open_source(const Json &wanted)
    {
#ifdef BENCHMARKS_HDF5
        const auto &hdf5 = wanted.at("hdf5");
        auto file = hdf5.at("file").get<std::string>();
        auto dataset = hdf5.value("dataset", std::string());
        if(!source || source->file() != file || (!dataset.empty() && source->dataset() != dataset))
        {
            source.reset();
            source = std::make_unique<bench::Hdf5Source>(file, dataset);
        }
        return source->json();
#else
        (void)wanted;
        throw std::invalid_argument("HDF5 frames need a benchmark build with BENCHMARKS_HDF5=ON");
#endif
    }
    void publish(std::stop_token stop)
    {
        pthread_setname_np(pthread_self(), "perf-frames");
        auto cpu = bench::thread_cpu_ns();
#ifdef BENCHMARKS_HDF5
        // Made before this thread is pinned, so the reader keeps the affinity it inherits.
        std::optional<ReadAhead> ahead;
        if(source && read_ahead)
            ahead.emplace(*publisher, *source, frames, read_ahead, verify, corrupt_every);
#endif
        bench::pin(producer_cpu);
        // Linux's inherited timer slack can coalesce a 20 us source into bursts. Tighten
        // only this benchmark producer's sleeps and report the setting actually obtained.
        if(rate)
            (void)::prctl(PR_SET_TIMERSLACK, 1UL, 0UL, 0UL, 0UL);
        auto timer_slack_ns = ::prctl(PR_GET_TIMERSLACK, 0UL, 0UL, 0UL, 0UL);
        bench::Memory memory(gpu);
        std::unordered_set<const std::byte *> filled;
        std::uint64_t published = 0, refused = 0, pacing_late = 0;
        std::int64_t waited_ns = 0, reading_ns = 0, waited_for_read_ns = 0;
        std::string failure;
        auto begin = bench::now_ns();
        const auto period = rate ? 1'000'000'000 / rate : 0;
        while(published < frames && !stop.stop_requested())
        {
            if(period)
            {
                auto due = begin + std::int64_t(published * period);
                std::this_thread::sleep_until(Deadline(std::chrono::nanoseconds(due)));
                pacing_late += bench::now_ns() > due + std::int64_t(period);
            }
#ifdef BENCHMARKS_HDF5
            if(ahead)
            {
                auto asked = bench::now_ns();
                auto slot = ahead->next(stop);
                waited_for_read_ns += bench::now_ns() - asked;
                if(!slot)
                {
                    failure = ahead->failure();
                    break;
                }
                std::move(*slot).publish(rate ? std::optional<std::uint64_t>(bench::now_ns())
                                              : std::nullopt);
                ++published;
                continue;
            }
#endif
            auto asked = bench::now_ns();
            auto slot = publisher->acquire(Deadline::clock::now() + 100ms);
            waited_ns += bench::now_ns() - asked;
            if(!slot)
            {
                ++refused;
                if(failure = publisher->health().failure; !failure.empty())
                    break;
                continue;
            }
            auto payload =
                gpu ? std::span(static_cast<std::byte *>(slot->gpu_payload(memory.stream())),
                                slot->payload_bytes())
                    : slot->payload();
#ifdef BENCHMARKS_HDF5
            if(source)
                reading_ns += read_into(*slot, published, *source, verify, corrupt_every);
            else
#endif
                if(verify)
                memory.fill(payload, published,
                            corrupt_every && published % corrupt_every == corrupt_every - 1);
            else if(filled.insert(payload.data()).second)
                memory.fill(payload, 0); // once for each slot: the clock is not on a fill
            // A paced local test measures publish-to-read against the host's steady clock.
            std::move(*slot).publish(rate ? std::optional<std::uint64_t>(bench::now_ns())
                                          : std::nullopt);
            ++published;
        }
        Json reader = nullptr;
#ifdef BENCHMARKS_HDF5
        if(ahead)
        {
            reader = ahead->json();
            if(published < frames && failure.empty())
                failure = ahead->failure();
            reading_ns = std::int64_t(reader["seconds_reading"].get<double>() * 1e9);
            ahead.reset(); // the slots it still holds go back before the publisher finishes
        }
#endif
        // The sessions are still there: once they end, Health has no transfer times left.
        auto health = bench::health_json(publisher->health());
        publisher->finish();
        Json outcome{{"published", published},
                     {"seconds", double(bench::now_ns() - begin) / 1e9},
                     {"seconds_waiting_for_a_slot", double(waited_ns) / 1e9},
                     {"seconds_reading", double(reading_ns) / 1e9},
                     {"seconds_waiting_for_a_read", double(waited_for_read_ns) / 1e9},
                     {"read_ahead", source ? read_ahead : 0},
                     {"reader", reader},
                     {"slots_refused", refused},
                     {"pacing_late_frames", pacing_late},
                     {"pacing_timer_slack_ns", timer_slack_ns},
                     {"cpu_seconds", bench::cpu_seconds_since(cpu)},
                     {"producer_cpu", sched_getcpu()},
                     {"health_at_finish", health},
                     {"failure", failure},
                     {"source", source_json}};
        std::lock_guard lock(commands);
        report = std::move(outcome);
        running = false;
    }
    std::mutex commands;
    std::unique_ptr<Publisher> publisher;
#ifdef BENCHMARKS_HDF5
    std::unique_ptr<bench::Hdf5Source> source; // only the frames thread reads it during a run
#else
    std::unique_ptr<int> source;
#endif
    Json source_json = {{"kind", "generated"}};
    std::uint64_t frames = 0, corrupt_every = 0, rate = 0;
    std::uint32_t read_ahead = 8;
    bool verify = false, started = false, gpu = false;
    int producer_cpu = -1;
    std::atomic<bool> running{false};
    Json report = Json::object();
    std::jthread frames_thread;
};

class Command : public Tango::Command
{
  public:
    using Run = CORBA::Any *(*)(Command &, Device &, const CORBA::Any &);
    Command(const char *name, Tango::CmdArgType in, Tango::CmdArgType out, Run run)
        : Tango::Command(name, in, out), run_(run)
    {
    }
    CORBA::Any *execute(Tango::DeviceImpl *device, const CORBA::Any &in) override
    {
        try
        {
            return run_(*this, *static_cast<Device *>(device), in);
        }
        catch(const std::exception &error)
        {
            Tango::Except::throw_exception("Refused", error.what(), "tango_ucx_perf_publisher");
        }
    }
    CORBA::Any *text(const std::string &value)
    {
        Tango::DevString out = Tango::string_dup(value.c_str());
        return insert(out);
    }

  private:
    Run run_;
};

class DeviceClass : public Tango::DeviceClass
{
  public:
    DeviceClass() : Tango::DeviceClass(std::string("UcxPerfPublisher")) {}
    void command_factory() override
    {
        command_list.push_back(new Command("Configure", Tango::DEV_STRING, Tango::DEV_STRING,
                                           [](Command &c, Device &d, const CORBA::Any &in)
                                           {
                                               Tango::DevString document = nullptr;
                                               c.extract(in, document);
                                               return c.text(d.configure(document));
                                           }));
        command_list.push_back(new Command("Describe", Tango::DEV_STRING, Tango::DEV_STRING,
                                           [](Command &c, Device &d, const CORBA::Any &in)
                                           {
                                               Tango::DevString document = nullptr;
                                               c.extract(in, document);
                                               return c.text(d.describe(document));
                                           }));
        command_list.push_back(new Command("Start", Tango::DEV_VOID, Tango::DEV_VOID,
                                           [](Command &c, Device &d, const CORBA::Any &)
                                           {
                                               d.start();
                                               return c.insert();
                                           }));
        command_list.push_back(new Command("Result", Tango::DEV_VOID, Tango::DEV_STRING,
                                           [](Command &c, Device &d, const CORBA::Any &)
                                           { return c.text(d.result()); }));
        install_ucx_commands(command_list);
    }
    void attribute_factory(std::vector<Tango::Attr *> &list) override
    {
        install_ucx_attributes(list);
    }
    void device_factory(const Tango::DevVarStringArray *names) override
    {
        for(CORBA::ULong i = 0; i < names->length(); ++i)
            device_list.push_back(new Device(this, std::string((*names)[i])));
        for(auto *device : device_list) // with -nodb the CORBA object key is how it is found
            export_device(device, device->get_name().c_str());
    }
};

class Server : public Tango::DServer
{
  public:
    using Tango::DServer::DServer;

  private:
    void class_factory() override
    {
        add_class(new DeviceClass);
    }
};
} // namespace

int main(int argc, char *argv[])
{
    try
    {
        auto *util = Tango::Util::init(argc, argv);
        util->register_dserver_constructor(
            [](Tango::DeviceClass *c, const std::string &name, const std::string &description,
               Tango::DevState state, const std::string &status) -> Tango::DServer *
            { return new Server(c, name.c_str(), description.c_str(), state, status.c_str()); });
        util->server_init();
        util->server_run();
        util->server_cleanup();
        return 0;
    }
    catch(const Tango::DevFailed &error)
    {
        Tango::Except::print_exception(error);
    }
    catch(const std::exception &error)
    {
        std::cerr << "tango_ucx_perf_publisher: " << error.what() << std::endl;
    }
    return 1;
}
