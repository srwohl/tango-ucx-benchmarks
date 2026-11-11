#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Copyright contributors to the tango-ucx project
# SPDX-License-Identifier: LGPL-3.0-or-later
#
# Builds acquire-zarr ACQUIRE_ZARR_VERSION (v0.10.0) from source and installs it into stage-zarr/.
# Run by `pixi run -e zarr stage-zarr`. acquire-zarr is not on conda-forge, and its release
# libraries need a newer glibc than Debian 12's. Its build expects vcpkg: here conda-forge gives
# the packages, and small CMake files stand in for the three it finds by vcpkg's names: blosc,
# which conda-forge packages without CMake files, and the "unofficial" curlpp and inih of its S3
# client, minio-cpp. inih is not on conda-forge, so its two files are built here. The S3 client
# is linked in whether or not a stream uses S3.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
version=${ACQUIRE_ZARR_VERSION:-v0.10.0}
inih=r58
work=$PWD/build-acquire-zarr
stage=$PWD/stage-zarr
if [[ $(cat "$stage/.version" 2>/dev/null) == "$version" ]]; then
    exit 0
fi
# An install keeps each file's time: a new version replaces the stage and the build using it.
rm -rf "$work" "$stage" build-zarr
mkdir -p "$work" "$stage/lib/cmake" "$stage/include"

git clone --quiet --depth 1 --branch "$version" --recurse-submodules --shallow-submodules \
    https://github.com/acquire-project/acquire-zarr.git "$work/source"
curl -fsSL "https://github.com/benhoyt/inih/archive/refs/tags/$inih.tar.gz" | tar -xz -C "$work"

ini=$work/inih-$inih
"${CC:-cc}" -O2 -fPIC -c "$ini/ini.c" -o "$work/ini.o"
"${CXX:-c++}" -O2 -fPIC -c "$ini/cpp/INIReader.cpp" -I"$ini" -o "$work/INIReader.o"
"${AR:-ar}" rcs "$stage/lib/libinireader.a" "$work/ini.o" "$work/INIReader.o"
cp "$ini/ini.h" "$ini/cpp/INIReader.h" "$stage/include/"

mkdir -p "$stage/lib/cmake/blosc" "$stage/lib/cmake/unofficial-curlpp" \
    "$stage/lib/cmake/unofficial-inih"
cat > "$stage/lib/cmake/blosc/blosc-config.cmake" <<'EOF'
# conda-forge's blosc, as the target acquire-zarr links.
if(NOT TARGET blosc_static)
    find_library(BLOSC_LIBRARY blosc REQUIRED)
    find_path(BLOSC_INCLUDE_DIR blosc.h REQUIRED)
    add_library(blosc_static UNKNOWN IMPORTED)
    set_target_properties(blosc_static PROPERTIES IMPORTED_LOCATION "${BLOSC_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${BLOSC_INCLUDE_DIR}")
endif()
EOF
cat > "$stage/lib/cmake/unofficial-curlpp/unofficial-curlpp-config.cmake" <<'EOF'
# conda-forge's curlpp, as vcpkg names it.
if(NOT TARGET unofficial::curlpp::curlpp)
    find_package(CURL REQUIRED)
    find_library(CURLPP_LIBRARY NAMES libcurlpp.so curlpp REQUIRED)
    find_path(CURLPP_INCLUDE_DIR curlpp/cURLpp.hpp REQUIRED)
    add_library(unofficial::curlpp::curlpp UNKNOWN IMPORTED)
    set_target_properties(unofficial::curlpp::curlpp PROPERTIES IMPORTED_LOCATION "${CURLPP_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CURLPP_INCLUDE_DIR}" INTERFACE_LINK_LIBRARIES CURL::libcurl)
endif()
EOF
cat > "$stage/lib/cmake/unofficial-inih/unofficial-inih-config.cmake" <<'EOF'
# The inih built by stage_zarr.sh, as vcpkg names it.
if(NOT TARGET unofficial::inih::inireader)
    get_filename_component(_inih_prefix "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
    add_library(unofficial::inih::inireader STATIC IMPORTED)
    set_target_properties(unofficial::inih::inireader PROPERTIES
        IMPORTED_LOCATION "${_inih_prefix}/lib/libinireader.a"
        INTERFACE_INCLUDE_DIRECTORIES "${_inih_prefix}/include")
endif()
EOF

# RelWithDebInfo: the library's Release build turns on link-time optimization, whose objects
# only a linker of the same compiler reads.
cmake -GNinja -S "$work/source" -B "$work/build" -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DBUILD_TESTING=OFF -DCMAKE_PREFIX_PATH="$stage;${CONDA_PREFIX:-}" \
    -DCMAKE_INSTALL_PREFIX="$stage"
cmake --build "$work/build"
cmake --install "$work/build"
echo "$version" > "$stage/.version"
