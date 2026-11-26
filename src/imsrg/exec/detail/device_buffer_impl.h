// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_EXEC_DETAIL_DEVICE_BUFFER_IMPL_H_
#define IMSRG_EXEC_DETAIL_DEVICE_BUFFER_IMPL_H_
/// @file
/// Implementation of `DeviceBuffer<T>` (all member definitions), including
/// Kokkos. For `.cc` files in `src/imsrg/exec/` and `src/imsrg/*/kernels/` only
/// (design/20 section 4). Including this header lets a kernel TU explicitly
/// instantiate `DeviceBuffer<MyPod>`.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "imsrg/base/assert.h"
#include "imsrg/exec/device_buffer.h"

namespace imsrg::exec {

/// Kokkos execution space kernels run on.
using DefaultExecSpace = Kokkos::DefaultExecutionSpace;
/// Kokkos execution space for host-side parallel work.
using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
/// Memory space of `DeviceBuffer` data.
using DeviceMemorySpace = DefaultExecSpace::memory_space;
/// A 1-D view in the device memory space.
template <class T>
using DeviceView = Kokkos::View<T*, DeviceMemorySpace>;
/// Signed 64-bit index of all facade loops.
using Index = std::int64_t;

namespace detail {

template <class T>
using HostView = typename DeviceView<T>::host_mirror_type;

/// Device kernel of `DeviceBuffer::fill`: `v(i) = value` for all i. Complexity
/// O(n). A free function (not a member), see design/12 section 5 rule 8.
template <class T>
void FillView(const DeviceView<T>& v, T value) {
  Kokkos::parallel_for(
      "imsrg::DeviceBuffer::fill",
      Kokkos::RangePolicy<DefaultExecSpace, Kokkos::IndexType<Index>>(
          0, static_cast<Index>(v.extent(0))),
      KOKKOS_LAMBDA(const Index i) { v(i) = value; });
}

inline constexpr bool kMemoryIsHostAccessible =
    Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                               DeviceMemorySpace>::accessible;

}  // namespace detail

template <class T>
struct DeviceBuffer<T>::Impl {
  std::string label;
  DeviceView<T> dev;
  mutable detail::HostView<T> host;  // valid iff has_host
  mutable bool has_host = false;

  Impl(std::string_view lab, std::size_t n) : label(lab), dev(label, n) {}

  /// Creates the host mirror (initialized from the device data) if it does not
  /// exist yet. The mirror has the view's own `host_mirror_type`, so it aliases
  /// the device data whenever that is host accessible (including unified-memory
  /// architectures) and is a separate copy otherwise.
  void EnsureHost() const {
    if (!has_host) {
      host = Kokkos::create_mirror_view(dev);
      if (host.data() != dev.data()) {
        Kokkos::deep_copy(host, dev);
      }
      has_host = true;
    }
  }

  /// True if the host mirror exists and is the device data itself.
  bool aliases() const { return has_host && host.data() == dev.data(); }
};

namespace detail {

template <class U>
struct DeviceBufferAccess {
  static DeviceView<U> view(DeviceBuffer<U>& b) {
    return b.impl_ ? b.impl_->dev : DeviceView<U>();
  }
  static DeviceView<const U> view(const DeviceBuffer<U>& b) {
    return b.impl_ ? DeviceView<const U>(b.impl_->dev) : DeviceView<const U>();
  }
};

inline constexpr std::string_view kDefaultLabel = "imsrg::DeviceBuffer";

}  // namespace detail

// User-provided (not `= default`): a defaulted constructor is implicitly
// `__host__ __device__` under hipcc/clang and would reference the host-only
// `~unique_ptr` (exception cleanup path). A user-provided one is host-only.
template <class T>
DeviceBuffer<T>::DeviceBuffer() noexcept : impl_(nullptr) {}

template <class T>
DeviceBuffer<T>::DeviceBuffer(std::size_t n, std::string_view label)
    : impl_(n > 0 ? std::make_unique<Impl>(label, n) : nullptr) {}

template <class T>
DeviceBuffer<T>::DeviceBuffer(const DeviceBuffer& other) {
  if (other.impl_) {
    impl_ = std::make_unique<Impl>(other.impl_->label, 0);
    impl_->dev = DeviceView<T>(
        Kokkos::view_alloc(impl_->label, Kokkos::WithoutInitializing),
        other.impl_->dev.extent(0));
    Kokkos::deep_copy(impl_->dev, other.impl_->dev);
  }
}

template <class T>
DeviceBuffer<T>& DeviceBuffer<T>::operator=(const DeviceBuffer& other) {
  if (this != &other) {
    DeviceBuffer tmp(other);
    impl_ = std::move(tmp.impl_);
  }
  return *this;
}

template <class T>
DeviceBuffer<T>::DeviceBuffer(DeviceBuffer&& other) noexcept = default;

template <class T>
DeviceBuffer<T>& DeviceBuffer<T>::operator=(DeviceBuffer&& other) noexcept =
    default;

template <class T>
DeviceBuffer<T>::~DeviceBuffer() = default;

template <class T>
std::size_t DeviceBuffer<T>::size() const noexcept {
  return impl_ ? impl_->dev.extent(0) : 0;
}

template <class T>
bool DeviceBuffer<T>::empty() const noexcept {
  return size() == 0;
}

template <class T>
std::string_view DeviceBuffer<T>::label() const noexcept {
  return impl_ ? std::string_view(impl_->label) : detail::kDefaultLabel;
}

template <class T>
void DeviceBuffer<T>::resize(std::size_t n) {
  if (n == 0) {
    impl_.reset();  // empty buffers hold no allocation
    return;
  }
  if (!impl_) {
    impl_ = std::make_unique<Impl>(detail::kDefaultLabel, n);
    return;
  }
  const std::size_t keep = std::min(n, impl_->dev.extent(0));
  auto fresh = std::make_unique<Impl>(impl_->label, n);
  if (keep > 0) {
    const auto range = std::pair<std::size_t, std::size_t>(0, keep);
    Kokkos::deep_copy(Kokkos::subview(fresh->dev, range),
                      Kokkos::subview(impl_->dev, range));
  }
  impl_ = std::move(fresh);
}

template <class T>
void DeviceBuffer<T>::fill(T value) {
  if (size() > 0) {
    detail::FillView<T>(impl_->dev, value);
    // A separate host mirror gets the same value, so both images agree
    // without a sync (set_zero followed by host reads is common).
    if (impl_->has_host && !impl_->aliases()) {
      Kokkos::deep_copy(impl_->host, value);
    }
  }
}

template <class T>
void DeviceBuffer<T>::CopyFromHost(std::span<const T> src) {
  IMSRG_CHECK(src.size() == size(),
              "DeviceBuffer::copy_from_host: size mismatch");
  if (src.empty()) {
    return;
  }
  Kokkos::View<const T*, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      from(src.data(), src.size());
  Kokkos::deep_copy(impl_->dev, from);
}

template <class T>
void DeviceBuffer<T>::CopyToHost(std::span<T> dst) const {
  IMSRG_CHECK(dst.size() == size(),
              "DeviceBuffer::copy_to_host: size mismatch");
  if (dst.empty()) {
    return;
  }
  Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      to(dst.data(), dst.size());
  Kokkos::deep_copy(to, impl_->dev);
}

template <class T>
std::vector<T> DeviceBuffer<T>::ToVector() const {
  std::vector<T> out(size());
  CopyToHost(std::span<T>(out));
  return out;
}

template <class T>
std::span<T> DeviceBuffer<T>::host_span() {
  if (size() == 0) {
    return {};
  }
  impl_->EnsureHost();
  return {impl_->host.data(), impl_->host.extent(0)};
}

template <class T>
std::span<const T> DeviceBuffer<T>::host_span() const {
  if (size() == 0) {
    return {};
  }
  impl_->EnsureHost();
  return {impl_->host.data(), impl_->host.extent(0)};
}

template <class T>
void DeviceBuffer<T>::SyncToHost() const {
  if (size() == 0) {
    return;
  }
  if (!impl_->has_host) {
    Kokkos::fence("imsrg::DeviceBuffer::sync_to_host");
    impl_->EnsureHost();  // created from (or aliasing) the device data
    return;
  }
  if (impl_->aliases()) {
    Kokkos::fence("imsrg::DeviceBuffer::sync_to_host");
    return;
  }
  Kokkos::deep_copy(impl_->host, impl_->dev);
}

template <class T>
void DeviceBuffer<T>::SyncToDevice() {
  if (size() == 0) {
    return;
  }
  impl_->EnsureHost();
  if (impl_->aliases()) {
    Kokkos::fence("imsrg::DeviceBuffer::sync_to_device");
    return;
  }
  Kokkos::deep_copy(impl_->dev, impl_->host);
}

template <class T>
bool DeviceBuffer<T>::host_is_device() const noexcept {
  return detail::kMemoryIsHostAccessible;
}

template <class T>
T* DeviceBuffer<T>::device_data() noexcept {
  return impl_ ? impl_->dev.data() : nullptr;
}

template <class T>
const T* DeviceBuffer<T>::device_data() const noexcept {
  return impl_ ? impl_->dev.data() : nullptr;
}

}  // namespace imsrg::exec

#endif  // IMSRG_EXEC_DETAIL_DEVICE_BUFFER_IMPL_H_
