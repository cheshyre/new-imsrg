#pragma once
// Test-only kernel entry points (Kokkos-free interface) that evaluate the device accessors of
// basis/device_tables.hpp; the kernels are in device_tables_kernels.cpp.
#include <cstdint>

#include "imsrg/basis/device_tables.hpp"
#include "imsrg/exec/device_buffer.hpp"

namespace imsrg::basis::test {

/// For every ket k and J = 0..max_J: local[k (max_J + 1) + J] = local_index(k, 2J),
/// channel[...] = channel_of(p_k, q_k, 2J).kind value; ket_idx[k] = ket_index(p_k, q_k).
/// All buffers must be pre-sized (local/channel: num_kets (max_J + 1), ket_idx: num_kets).
void eval_ket_tables(const TwoBodyTableView& tb, std::int32_t max_J,
                     exec::DeviceBuffer<std::int32_t>& local,
                     exec::DeviceBuffer<std::int32_t>& channel,
                     exec::DeviceBuffer<std::int32_t>& ket_idx);

/// Odd and negative two_j probes: out[k * 3 + i] = local_index(k, TwoJ{probe_i}) for
/// probes {-2, 1, 2 (max_J + 1)}. Buffer size num_kets * 3.
void eval_invalid_two_j(const TwoBodyTableView& tb, std::int32_t max_J,
                        exec::DeviceBuffer<std::int32_t>& out);

/// For every channel: out[ket_offset + i] = global_ket(ch, i) for i < num_kets of the channel;
/// the buffer has size channel_kets total.
void eval_global_kets(const TwoBodyTableView& tb, exec::DeviceBuffer<std::int32_t>& out);

/// Per (ch, class) subset: begin/end/size of the subset into the three buffers (size
/// num_channels * kNumKetClasses), and copies of the local kets and products over [begin, end)
/// into `kets`, `occ`, `vac` (size of the flat subset table).
void eval_subsets(const KetSubsetTableView& sv, exec::DeviceBuffer<std::int32_t>& begin,
                  exec::DeviceBuffer<std::int32_t>& end, exec::DeviceBuffer<std::int32_t>& size,
                  exec::DeviceBuffer<std::int32_t>& kets, exec::DeviceBuffer<double>& occ,
                  exec::DeviceBuffer<double>& vac);

/// Copies the orbit table to six buffers of size num_orbits (n, l, j2, tz2, cvq) and occ.
void eval_orbits(const OrbitTableView& ov, exec::DeviceBuffer<std::int32_t>& n,
                 exec::DeviceBuffer<std::int32_t>& l, exec::DeviceBuffer<std::int32_t>& j2,
                 exec::DeviceBuffer<std::int32_t>& tz2, exec::DeviceBuffer<std::int32_t>& cvq,
                 exec::DeviceBuffer<double>& occ);

}  // namespace imsrg::basis::test
