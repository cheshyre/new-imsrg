#pragma once

/// @file
/// Precomputed "lifted one-body" lists of the 2-body-reduced term comm122ss (no Kokkos).
///
/// For the row ket r = (i, j) of a channel with total angular momentum J, the bra part of
/// [X, Y]_{(ij),(kl)} is  sum_c x(i,c) Y_u(c j; kl) + sum_c x(j,c) Y_u(i c; kl) - (X <-> Y),
/// where Y_u is the unnormalized element. Each term maps to one stored (normalized) row m of the
/// same channel, up to a fixed factor. The lists store, per row, the entries
/// (m, from, to, coef) with coef = exchange phase * sqrt2^[c == j] / sqrt2^[i == j]; the
/// dependence on (k, l) cancels, so the term is x(from,to) * coef * Y_norm(m, col). Entries whose
/// ket does not exist for this J (triangle, Pauli) are omitted.

#include <cstdint>

#include "imsrg/basis/model_space.hpp"
#include "imsrg/exec/device_buffer.hpp"

namespace imsrg::commutator::detail {

/// CSR lists of the lifted one-body entries, indexed by the global row (channel row base + local
/// row). Device buffers (zero-initialized when empty).
struct LiftedData {
  /// First global row of each channel; size num_channels + 1 (the last is the total row count).
  exec::DeviceBuffer<std::int32_t> row_base;
  /// CSR offsets of the entries of each global row; size total rows + 1.
  exec::DeviceBuffer<std::int32_t> entry_offset;
  exec::DeviceBuffer<std::int32_t> m;     ///< Local row (in the same channel) of each entry.
  exec::DeviceBuffer<std::int32_t> from;  ///< One-body row index of the entry.
  exec::DeviceBuffer<std::int32_t> to;    ///< One-body column index of the entry.
  exec::DeviceBuffer<double> coef;        ///< Phase and sqrt(2) factor of the entry.
};

/// Builds the lifted lists of `ms` on the host and uploads them. Needs an initialized
/// `exec::Runtime`. Complexity O(number of kets x orbits per one-body channel).
LiftedData build_lifted_data(const basis::ModelSpace& ms);

}  // namespace imsrg::commutator::detail
