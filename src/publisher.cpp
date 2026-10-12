// SPDX-FileCopyrightText: 2026 Copyright contributors to the tango-ucx project
// SPDX-License-Identifier: LGPL-3.0-or-later

// tango_ucx_perf_publisher: a Tango device that publishes what tango_ucx_perftest tells it to.
//   Configure  a JSON document in: frame_bytes, budget, frames, rate, verify, corrupt_every.
//              Out: the source slots the budget gave.
//   Start      publishes the frames at rate (0: as fast as slots are granted), and finishes.
//   Result     what the run took, as a JSON document.
#include "common.h"
#include "memory.h"

#include <tango-ucx/tango.h>
#include <tango/tango.h>

#include <atomic>
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
        Description description{
            Element::U8, {wanted.at("frame_bytes").get<std::uint64_t>()}, 0, {}};
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
        started = false;
        report = Json::object();
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
                    report = {{"failure", error.what()}};
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
    void publish(std::stop_token stop)
    {
        pthread_setname_np(pthread_self(), "perf-frames");
        bench::pin(producer_cpu);
        // Linux's inherited timer slack can coalesce a 20 us source into bursts. Tighten
        // only this benchmark producer's sleeps and report the setting actually obtained.
        if(rate)
            (void)::prctl(PR_SET_TIMERSLACK, 1UL, 0UL, 0UL, 0UL);
        auto timer_slack_ns = ::prctl(PR_GET_TIMERSLACK, 0UL, 0UL, 0UL, 0UL);
        auto cpu = bench::thread_cpu_ns();
        bench::Memory memory(gpu);
        std::unordered_set<const std::byte *> filled;
        std::uint64_t published = 0, refused = 0, pacing_late = 0;
        std::int64_t waited_ns = 0;
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
            if(verify)
            {
                memory.fill(payload, published,
                            corrupt_every && published % corrupt_every == corrupt_every - 1);
            }
            else if(filled.insert(payload.data()).second)
                memory.fill(payload, 0); // once for each slot: the clock is not on a fill
            // A paced local test measures publish-to-read against the host's steady clock.
            std::move(*slot).publish(rate ? std::optional<std::uint64_t>(bench::now_ns())
                                          : std::nullopt);
            ++published;
        }
        // The sessions are still there: once they end, Health has no transfer times left.
        auto health = bench::health_json(publisher->health());
        publisher->finish();
        Json outcome{{"published", published},
                     {"seconds", double(bench::now_ns() - begin) / 1e9},
                     {"seconds_waiting_for_a_slot", double(waited_ns) / 1e9},
                     {"slots_refused", refused},
                     {"pacing_late_frames", pacing_late},
                     {"pacing_timer_slack_ns", timer_slack_ns},
                     {"cpu_seconds", bench::cpu_seconds_since(cpu)},
                     {"producer_cpu", sched_getcpu()},
                     {"health_at_finish", health},
                     {"failure", failure}};
        std::lock_guard lock(commands);
        report = std::move(outcome);
        running = false;
    }
    std::mutex commands;
    std::unique_ptr<Publisher> publisher;
    std::uint64_t frames = 0, corrupt_every = 0, rate = 0;
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
