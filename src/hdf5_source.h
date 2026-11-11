// SPDX-FileCopyrightText: 2026 Copyright contributors to the tango-ucx project
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

// An image stack in an HDF5 file, read a frame at a time into a publisher's source slots.
// Ported from tango-bulk's example_hdf5_device: a rank-three dataset is [frame, height, width],
// a rank-two one is a single frame, and a NeXus file's `default` and `signal` attributes lead to
// the dataset when none is named. HDF5 converts the file's byte order to this host's on read.
#include "common.h"

#include <tango-ucx/ucx.h>

#include <hdf5.h>

#include <cmath>
#include <filesystem>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bench
{
class Hdf5Source
{
  public:
    Hdf5Source(std::string file, std::string dataset) : file_name_(std::move(file))
    {
        H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr); // failures are reported as exceptions
        file_ = Handle(H5Fopen(file_name_.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
        if(!file_)
            throw std::runtime_error("cannot open HDF5 file " + file_name_);
        dataset_path_ = dataset.empty() ? nexus_signal() : std::move(dataset);
        if(!exists(dataset_path_))
            throw std::runtime_error("no dataset " + dataset_path_ + " in " + file_name_);
        dataset_ = Handle(H5Dopen2(file_.id, dataset_path_.c_str(), H5P_DEFAULT), H5Dclose);
        if(!dataset_)
            throw std::runtime_error(dataset_path_ + " in " + file_name_ + " is not a dataset");
        describe_type();

        file_space_ = Handle(H5Dget_space(dataset_.id), H5Sclose);
        int rank = H5Sget_simple_extent_ndims(file_space_.id);
        if(rank < 2 || rank > 9)
            throw std::runtime_error("an image dataset has 2 to 9 dimensions, [frame, ...image]");
        dimensions_.resize(std::size_t(rank));
        H5Sget_simple_extent_dims(file_space_.id, dimensions_.data(), nullptr);
        frames_ = rank == 2 ? 1 : dimensions_.front();
        if(!frames_)
            throw std::runtime_error("the image dataset has no frames");
        shape_.assign(dimensions_.begin() + (rank == 2 ? 0 : 1), dimensions_.end());
        frame_bytes_ = element_bytes_;
        for(auto extent : shape_)
        {
            if(!extent)
                throw std::runtime_error("the image dataset has an empty image axis");
            frame_bytes_ *= extent;
        }
        if(frame_bytes_ > 2ull << 30)
            throw std::runtime_error("a frame of the dataset is over 2 GiB");

        // One frame's selection, moved along the frame axis on each read.
        count_ = dimensions_;
        if(rank > 2)
            count_.front() = 1;
        start_.assign(dimensions_.size(), 0);
        memory_space_ = Handle(H5Screate_simple(rank, count_.data(), nullptr), H5Sclose);
    }

    const std::string &file() const
    {
        return file_name_;
    }
    const std::string &dataset() const
    {
        return dataset_path_;
    }
    std::uint64_t frames() const
    {
        return frames_;
    }
    std::uint64_t frame_bytes() const
    {
        return frame_bytes_;
    }

    TangoUcx::Description description() const
    {
        TangoUcx::Description d;
        d.element = element_;
        d.shape.assign(shape_.begin(), shape_.end());
        d.field_bytes = sizeof(FileFields);
        d.field_dtype = k_file_fields_dtype;
        return d;
    }

    // Reads frame `index` of the dataset, which must be under frames(), into `destination`.
    void read(std::uint64_t index, std::span<std::byte> destination)
    {
        if(index >= frames_ || destination.size() < frame_bytes_)
            throw std::out_of_range("an HDF5 frame outside the dataset or the slot");
        if(dimensions_.size() > 2)
            start_.front() = index;
        if(H5Sselect_hyperslab(file_space_.id, H5S_SELECT_SET, start_.data(), nullptr,
                               count_.data(), nullptr) < 0 ||
           H5Dread(dataset_.id, memory_type_, memory_space_.id, file_space_.id, H5P_DEFAULT,
                   destination.data()) < 0)
            throw std::runtime_error("reading frame " + std::to_string(index) + " of " +
                                     dataset_path_ + " failed");
    }

    // What a report needs to know of the file: its geometry, how it is stored, and the rate
    // its acquisition metadata gives, if any.
    Json json() const
    {
        Json layout = "unknown", chunk = nullptr, filters = Json::array();
        if(Handle plist(H5Dget_create_plist(dataset_.id), H5Pclose); plist)
        {
            switch(H5Pget_layout(plist.id))
            {
                case H5D_COMPACT:
                    layout = "compact";
                    break;
                case H5D_CONTIGUOUS:
                    layout = "contiguous";
                    break;
                case H5D_CHUNKED:
                {
                    layout = "chunked";
                    std::vector<hsize_t> dims(dimensions_.size());
                    if(H5Pget_chunk(plist.id, int(dims.size()), dims.data()) >= 0)
                        chunk = dims;
                    break;
                }
                case H5D_VIRTUAL:
                    layout = "virtual";
                    break;
                default:
                    break;
            }
            for(int i = 0, n = H5Pget_nfilters(plist.id); i < n; ++i)
            {
                unsigned flags = 0, config = 0;
                std::size_t values = 0;
                char name[256] = {};
                auto filter = H5Pget_filter2(plist.id, unsigned(i), &flags, &values, nullptr,
                                             sizeof(name), name, &config);
                filters.push_back(name[0] ? Json(name) : Json(int(filter)));
            }
        }
        std::error_code ignored;
        auto file_bytes = std::filesystem::file_size(file_name_, ignored);
        auto rate = nominal_rate();
        return {{"kind", "hdf5"},
                {"file", std::filesystem::absolute(file_name_, ignored).string()},
                {"file_bytes", ignored ? Json(nullptr) : Json(file_bytes)},
                {"dataset", dataset_path_},
                {"element", element_name_},
                {"shape", shape_},
                {"frame_bytes", frame_bytes_},
                {"frames", frames_},
                {"layout", layout},
                {"chunk", chunk},
                {"filters", filters},
                {"nominal_rate", rate ? Json(*rate) : Json(nullptr)}};
    }

  private:
    struct Handle
    {
        hid_t id = H5I_INVALID_HID;
        herr_t (*close)(hid_t) = nullptr;
        Handle() = default;
        Handle(hid_t i, herr_t (*c)(hid_t)) : id(i), close(c) {}
        Handle(Handle &&o) noexcept : id(std::exchange(o.id, H5I_INVALID_HID)), close(o.close) {}
        Handle &operator=(Handle &&o) noexcept
        {
            std::swap(id, o.id);
            std::swap(close, o.close);
            return *this;
        }
        ~Handle()
        {
            if(id >= 0 && close)
                close(id);
        }
        explicit operator bool() const
        {
            return id >= 0;
        }
    };

    // H5Lexists asks only of the last link, and fails when an earlier one is missing.
    bool exists(const std::string &path) const
    {
        if(path.empty() || path.front() != '/')
            return false;
        for(std::size_t at = 1; at <= path.size();)
        {
            auto slash = path.find('/', at);
            auto prefix = path.substr(0, slash);
            if(H5Lexists(file_.id, prefix.c_str(), H5P_DEFAULT) <= 0)
                return false;
            if(slash == std::string::npos)
                return true;
            at = slash + 1;
        }
        return true;
    }

    std::optional<std::string> string_attribute(const std::string &object, const char *name) const
    {
        if(H5Aexists_by_name(file_.id, object.c_str(), name, H5P_DEFAULT) <= 0)
            return std::nullopt;
        Handle attribute(H5Aopen_by_name(file_.id, object.c_str(), name, H5P_DEFAULT, H5P_DEFAULT),
                         H5Aclose);
        Handle type(H5Aget_type(attribute.id), H5Tclose);
        if(!attribute || !type || H5Tget_class(type.id) != H5T_STRING)
            return std::nullopt;
        std::string value;
        if(H5Tis_variable_str(type.id) > 0)
        {
            Handle memory(H5Tcopy(H5T_C_S1), H5Tclose);
            H5Tset_size(memory.id, H5T_VARIABLE);
            H5Tset_cset(memory.id, H5Tget_cset(type.id));
            char *text = nullptr;
            if(H5Aread(attribute.id, memory.id, &text) < 0 || !text)
                return std::nullopt;
            value = text;
            H5free_memory(text);
        }
        else
        {
            std::string buffer(H5Tget_size(type.id), '\0');
            if(H5Aread(attribute.id, type.id, buffer.data()) < 0)
                return std::nullopt;
            value = buffer.substr(0, buffer.find('\0'));
        }
        while(!value.empty() && value.back() == ' ')
            value.pop_back(); // a fixed-length string may be space padded
        return value;
    }

    static std::string join(const std::string &base, const std::string &child)
    {
        if(child.front() == '/')
            return child;
        return base == "/" ? "/" + child : base + "/" + child;
    }

    // NeXus: the root's `default` names the entry, the entry's `default` its NXdata group, and
    // that group's `signal` the dataset.
    std::string nexus_signal() const
    {
        std::string path = "/";
        for(const char *name : {"default", "default", "signal"})
        {
            auto next = string_attribute(path, name);
            if(!next || next->empty())
                throw std::runtime_error("no dataset was named and " + file_name_ +
                                         " has no NeXus default/signal chain at " + path +
                                         "; name the dataset");
            path = join(path, *next);
        }
        return path;
    }

    void describe_type()
    {
        Handle type(H5Dget_type(dataset_.id), H5Tclose);
        element_bytes_ = H5Tget_size(type.id);
        auto type_class = H5Tget_class(type.id);
        struct Choice
        {
            TangoUcx::Element element;
            hid_t memory;
            const char *name;
        };
        std::optional<Choice> choice;
        if(type_class == H5T_INTEGER)
        {
            bool sign = H5Tget_sign(type.id) != H5T_SGN_NONE;
            switch(element_bytes_)
            {
                case 1:
                    choice = sign ? Choice{TangoUcx::Element::I8, H5T_NATIVE_INT8, "i8"}
                                  : Choice{TangoUcx::Element::U8, H5T_NATIVE_UINT8, "u8"};
                    break;
                case 2:
                    choice = sign ? Choice{TangoUcx::Element::I16, H5T_NATIVE_INT16, "i16"}
                                  : Choice{TangoUcx::Element::U16, H5T_NATIVE_UINT16, "u16"};
                    break;
                case 4:
                    choice = sign ? Choice{TangoUcx::Element::I32, H5T_NATIVE_INT32, "i32"}
                                  : Choice{TangoUcx::Element::U32, H5T_NATIVE_UINT32, "u32"};
                    break;
                case 8:
                    choice = sign ? Choice{TangoUcx::Element::I64, H5T_NATIVE_INT64, "i64"}
                                  : Choice{TangoUcx::Element::U64, H5T_NATIVE_UINT64, "u64"};
                    break;
            }
        }
        else if(type_class == H5T_FLOAT && element_bytes_ == 4)
            choice = Choice{TangoUcx::Element::F32, H5T_NATIVE_FLOAT, "f32"};
        else if(type_class == H5T_FLOAT && element_bytes_ == 8)
            choice = Choice{TangoUcx::Element::F64, H5T_NATIVE_DOUBLE, "f64"};
        if(!choice)
            throw std::runtime_error(
                "the image dataset holds neither 8 to 64-bit integers nor 32 or 64-bit floats");
        element_ = choice->element;
        memory_type_ = choice->memory;
        element_name_ = choice->name;
    }

    std::optional<double> scalar(const std::string &path) const
    {
        if(!exists(path))
            return std::nullopt;
        Handle value(H5Dopen2(file_.id, path.c_str(), H5P_DEFAULT), H5Dclose);
        double result = 0;
        if(!value ||
           H5Dread(value.id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &result) < 0)
            return std::nullopt;
        return result;
    }

    // Where a soft link at `path` leads, as an absolute path, or nothing.
    std::optional<std::string> link_target(const std::string &path) const
    {
        H5L_info2_t info;
        if(H5Lget_info2(file_.id, path.c_str(), &info, H5P_DEFAULT) < 0 ||
           info.type != H5L_TYPE_SOFT)
            return std::nullopt;
        std::string target(info.u.val_size, '\0');
        if(H5Lget_val(file_.id, path.c_str(), target.data(), target.size(), H5P_DEFAULT) < 0)
            return std::nullopt;
        target = target.substr(0, target.find('\0'));
        if(target.empty())
            return std::nullopt;
        return join(path.substr(0, path.find_last_of('/')).empty()
                        ? std::string("/")
                        : path.substr(0, path.find_last_of('/')),
                    target);
    }

    // The acquisition's frame rate: 1 / (exposure_time + latency_time) from an `acquisition`
    // group beside the dataset or an ancestor. A NeXus signal is often a soft link to the
    // detector's data, with the acquisition beside the link's target: that is looked in too.
    std::optional<double> nominal_rate() const
    {
        std::vector<std::string> starts{dataset_path_};
        if(auto target = link_target(dataset_path_))
            starts.push_back(*target);
        for(const auto &start : starts)
            for(auto group = start.substr(0, start.find_last_of('/'));;
                group = group.substr(0, group.find_last_of('/')))
            {
                auto base = group + "/acquisition/";
                auto exposure = scalar(base + "exposure_time");
                auto latency = scalar(base + "latency_time");
                if(exposure && latency)
                    if(auto period = *exposure + *latency; std::isfinite(period) && period > 0)
                        return 1 / period;
                if(group.empty())
                    break;
            }
        return std::nullopt;
    }

    std::string file_name_, dataset_path_, element_name_;
    Handle file_, dataset_, file_space_, memory_space_;
    TangoUcx::Element element_ = TangoUcx::Element::U8;
    hid_t memory_type_ = H5I_INVALID_HID;
    std::size_t element_bytes_ = 0;
    std::vector<hsize_t> dimensions_, shape_, start_, count_;
    std::uint64_t frames_ = 0, frame_bytes_ = 0;
};
} // namespace bench
