#pragma once
/// @file
/// `DeviceBuffer<T>`: an owning, resizable array of trivially copyable `T` in the default memory
/// space of the compute backend (host memory for Serial/OpenMP, device memory for CUDA/HIP), with
/// an optional host mirror (design/12 section 2, design D13). This header does not include Kokkos:
/// the implementation is a pimpl whose member definitions live in
/// `imsrg/exec/detail/device_buffer_impl.hpp`. Kernel translation units reach the data through
/// `exec::view(buf)` in `imsrg/exec/kokkos_backend.hpp`.
///
/// `DeviceBuffer<T>` is explicitly instantiated in `device_buffer.cpp` for double, float,
/// std::int32_t, std::int64_t, std::uint16_t and std::uint8_t. A kernel TU of another module can
/// instantiate it for its own POD type by including the impl header and writing
/// `template class imsrg::exec::DeviceBuffer<MyPod>;`.
///
/// Memory model: kernels run asynchronously on backends with asynchronous launches. Host access
/// to data a kernel wrote (via `host_span`) requires `sync_to_host()` first, which fences (also
/// when the mirror aliases the device data). `copy_to_host`, `copy_from_host`, `to_vector` fence
/// implicitly.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

namespace imsrg::exec {

template <class T>
class DeviceBuffer;

namespace detail {
/// Grants the facade (`exec::view`) access to the pimpl. Defined in the impl header.
template <class U>
struct DeviceBufferAccess;
}  // namespace detail

/// Owning array of `T` in the default memory space. Value semantics: copying deep-copies the data
/// on the device, moving is cheap and leaves the source empty. Requires an initialized
/// `exec::Runtime` for every operation except default construction, moves of empty buffers, and
/// destruction of empty buffers.
template <class T>
class DeviceBuffer {
  static_assert(std::is_trivially_copyable_v<T>, "DeviceBuffer<T> needs a trivially copyable T");

 public:
  using value_type = T;

  /// Empty buffer without allocation; `label()` is "imsrg::DeviceBuffer".
  DeviceBuffer() noexcept;

  /// `n` zero-initialized elements, allocated with `label` (shown by Kokkos tools).
  explicit DeviceBuffer(std::size_t n, std::string_view label = "imsrg::DeviceBuffer");

  /// Deep copy (device-to-device); the host mirror of the copy is created on demand.
  DeviceBuffer(const DeviceBuffer& other);
  /// Deep copy, see the copy constructor.
  DeviceBuffer& operator=(const DeviceBuffer& other);
  /// Cheap move; `other` becomes empty.
  DeviceBuffer(DeviceBuffer&& other) noexcept;
  /// Cheap move; `other` becomes empty.
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
  ~DeviceBuffer();

  /// Number of elements.
  std::size_t size() const noexcept;
  /// True if `size() == 0`.
  bool empty() const noexcept;
  /// The allocation label (kept by copies; the default label for a default-constructed buffer).
  std::string_view label() const noexcept;

  /// Changes the size to `n`: the common prefix `[0, min(size(), n))` of the DEVICE data is kept,
  /// new elements are zero. Host-mirror changes not yet synced are discarded and the mirror is
  /// recreated on demand. Cost: one allocation and one device copy of the prefix.
  void resize(std::size_t n);

  /// Sets every element to `value` with a device kernel (asynchronous on async backends).
  /// Complexity O(size()), embarrassingly parallel.
  void fill(T value);

  /// Copies `src` to the device data. Precondition: `src.size() == size()` (IMSRG_CHECK). The
  /// host mirror, if separate from the device data, becomes stale until `sync_to_host()`.
  void copy_from_host(std::span<const T> src);

  /// Copies the device data to `dst`. Precondition: `dst.size() == size()` (IMSRG_CHECK).
  void copy_to_host(std::span<T> dst) const;

  /// The device data as a new host vector.
  std::vector<T> to_vector() const;

  /// Host mirror, created on first use (initialized from the device data). When
  /// `host_is_device()` it aliases the device data. Writes become visible to kernels after
  /// `sync_to_device()`; kernel results become visible after `sync_to_host()`.
  std::span<T> host_span();
  /// Read-only host mirror; see the non-const overload.
  std::span<const T> host_span() const;

  /// Copies device data to the host mirror (creating it if needed); fences. A fence only when
  /// the mirror aliases the device data.
  void sync_to_host();
  /// Copies the host mirror to the device data (no-op apart from a fence when aliasing).
  void sync_to_device();

  /// True if device memory is host accessible, so mirror and device data are the same memory.
  bool host_is_device() const noexcept;

  /// Raw device pointer (nullptr when empty). Only dereference it in kernels.
  T* device_data() noexcept;
  /// Raw device pointer, read-only.
  const T* device_data() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  template <class U>
  friend struct detail::DeviceBufferAccess;
};

extern template class DeviceBuffer<double>;
extern template class DeviceBuffer<float>;
extern template class DeviceBuffer<std::int32_t>;
extern template class DeviceBuffer<std::int64_t>;
extern template class DeviceBuffer<std::uint16_t>;
extern template class DeviceBuffer<std::uint8_t>;

}  // namespace imsrg::exec
