// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/operator/three_body.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/base/error.h"
#include "imsrg/basis/three_body_distribution.h"
#include "imsrg/basis/three_body_rows.h"
#include "imsrg/operator/detail/part_kernels.h"
#include "imsrg/operator/operator_spec.h"

namespace imsrg::op {

ThreeBody::ThreeBody(std::shared_ptr<const basis::ModelSpace> ms, Hermiticity h)
    : ms_(std::move(ms)), hermiticity_(h) {
  if (ms_ == nullptr) {
    throw InputError("ThreeBody: null model space");
  }
  if (ms_->three_body_space() == nullptr) {
    throw InputError(
        "ThreeBody: the model space has no three-body space (build it with "
        "ModelSpace::WithThreeBody)");
  }
  offsets_ = OffsetsFor(packed());
  const std::int64_t local_size = ms_->three_body_rows().StoredSize(packed());
  IMSRG_CHECK(distributed() || local_size == offsets_.back(),
              "ThreeBody: the whole row set does not match the block layout");
  data_ = exec::DeviceBuffer<double>(static_cast<std::size_t>(local_size),
                                     "imsrg::op::ThreeBody");
}

bool ThreeBody::distributed() const noexcept {
  return ms_ != nullptr && ms_->three_body_distribution() != nullptr;
}

std::span<const std::int64_t> ThreeBody::block_offsets() const {
  IMSRG_CHECK(!distributed(),
              "ThreeBody::block_offsets: a distributed part does not hold "
              "whole blocks (use op::GatherToRoot)");
  return offsets_;
}

std::int64_t ThreeBody::LocalRow(Channel3Index ch, std::int64_t k) const {
  const std::int64_t off = ms_->three_body_rows().LocalRowOffset(
      ch, basis::KeptRow3Index{static_cast<std::int32_t>(k)}, packed());
  IMSRG_CHECK(off >= 0,
              "ThreeBody: the stored row of the element is not local to this "
              "rank");
  return off;
}

const basis::ThreeBodySpace& ThreeBody::space() const {
  IMSRG_CHECK(ms_ != nullptr, "ThreeBody::space: empty part");
  return *ms_->three_body_space();
}

std::vector<std::int64_t> ThreeBody::OffsetsFor(bool packed_mode) const {
  const auto channels = space().channels();
  std::vector<std::int64_t> off(channels.size() + 1, 0);
  for (std::size_t ch = 0; ch < channels.size(); ++ch) {
    off[ch + 1] = off[ch] + BlockSize(channels[ch].dim, packed_mode);
  }
  return off;
}

std::int32_t ThreeBody::dim(Channel3Index ch) const { return space().dim(ch); }

ThreeBody::Location ThreeBody::Locate(Channel3Index ch, basis::KeptRow3Index i,
                                      basis::KeptRow3Index j) const {
  const std::int64_t n = dim(ch);
  IMSRG_ASSERT(i.value() >= 0 && i.value() < n && j.value() >= 0 &&
               j.value() < n);
  const std::int64_t a = i.value();
  const std::int64_t b = j.value();
  const int h = HermiticitySign(hermiticity_);
  if (distributed()) {
    // Element (a, b) of the stored triangle lives in row min(a, b) (packed,
    // a >= b: z(a, b) at row b offset a - b) or in column b (dense).
    if (!packed()) return {LocalRow(ch, b) + a, 1};
    if (a >= b) {
      return {LocalRow(ch, b) + (a - b), a == b && h < 0 ? 0 : 1};
    }
    return {LocalRow(ch, a) + (b - a), h};
  }
  const std::int64_t off = offsets_[ch.ToSize()];
  if (!packed()) return {off + a + b * n, 1};
  if (a >= b) {
    return {off + PackedIndex(n, a, b), a == b && h < 0 ? 0 : 1};
  }
  return {off + PackedIndex(n, b, a), h};
}

double ThreeBody::get(Channel3Index ch, basis::KeptRow3Index i,
                      basis::KeptRow3Index j) const {
  const Location loc = Locate(ch, i, j);
  if (loc.sign == 0) return 0.0;
  return loc.sign * data_.host_span()[static_cast<std::size_t>(loc.index)];
}

void ThreeBody::set(Channel3Index ch, basis::KeptRow3Index i,
                    basis::KeptRow3Index j, double v) {
  const Location loc = Locate(ch, i, j);
  if (loc.sign == 0) {
    IMSRG_CHECK(v == 0.0,
                "ThreeBody::set: the diagonal of an anti-Hermitian part is "
                "zero");
    return;
  }
  data_.host_span()[static_cast<std::size_t>(loc.index)] = loc.sign * v;
}

void ThreeBody::Add(Channel3Index ch, basis::KeptRow3Index i,
                    basis::KeptRow3Index j, double v) {
  const Location loc = Locate(ch, i, j);
  if (loc.sign == 0) {
    IMSRG_CHECK(v == 0.0,
                "ThreeBody::Add: the diagonal of an anti-Hermitian part is "
                "zero");
    return;
  }
  data_.host_span()[static_cast<std::size_t>(loc.index)] += loc.sign * v;
}

void ThreeBody::set_zero() {
  if (data_.size() > 0) data_.fill(0.0);
}

namespace {

/// Zeroes the diagonal entries of the local rows of a distributed packed part
/// `hs` holding the rows `rows` (row k of a block of dimension n has length
/// n - k and starts with the diagonal entry).
void ZeroLocalPackedDiagonals(std::span<double> hs, const basis::RowSet& rows,
                              const basis::ThreeBodySpace& space) {
  std::size_t pos = 0;
  for (const basis::RowRange& r : rows.ranges()) {
    const std::int64_t n = space.dim(r.channel);
    for (std::int64_t k = r.row_begin.value(); k < r.row_end.value(); ++k) {
      hs[pos] = 0.0;
      pos += static_cast<std::size_t>(n - k);
    }
  }
}

/// Zeroes the diagonal of every channel block of a packed part `hs`.
void ZeroPackedDiagonals(std::span<double> hs,
                         std::span<const basis::KeptChannel3> channels,
                         const std::vector<std::int64_t>& offsets) {
  for (std::size_t ch = 0; ch < channels.size(); ++ch) {
    const std::int64_t n = channels[ch].dim;
    for (std::int64_t i = 0; i < n; ++i) {
      hs[static_cast<std::size_t>(offsets[ch] +
                                  ThreeBody::PackedIndex(n, i, i))] = 0.0;
    }
  }
}

/// Converts `src` (layout `offsets`) to `dst` (layout `new_offsets`):
/// packed -> dense when `!to_packed` (the old symmetry sign `old_h` fills the
/// upper triangle), dense -> packed otherwise (target hermiticity `h`).
void ConvertThreeBodyLayout(std::span<const double> src, std::span<double> dst,
                            std::span<const basis::KeptChannel3> channels,
                            const std::vector<std::int64_t>& offsets,
                            const std::vector<std::int64_t>& new_offsets,
                            bool to_packed, Hermiticity h, std::int32_t old_h) {
  for (std::size_t ch = 0; ch < channels.size(); ++ch) {
    const std::int64_t n = channels[ch].dim;
    const std::int64_t so = offsets[ch];
    const std::int64_t d0 = new_offsets[ch];
    for (std::int64_t j = 0; j < n; ++j) {
      for (std::int64_t i = 0; i < n; ++i) {
        if (!to_packed) {
          // Packed -> dense: the old symmetry fills the upper triangle.
          const double v =
              i >= j ? src[static_cast<std::size_t>(
                           so + ThreeBody::PackedIndex(n, i, j))]
                     : old_h * src[static_cast<std::size_t>(
                                   so + ThreeBody::PackedIndex(n, j, i))];
          dst[static_cast<std::size_t>(d0 + i + j * n)] = v;
        } else if (i >= j) {
          const bool zero_diag = i == j && h == Hermiticity::kAntiHermitian;
          dst[static_cast<std::size_t>(d0 + ThreeBody::PackedIndex(n, i, j))] =
              zero_diag ? 0.0 : src[static_cast<std::size_t>(so + i + j * n)];
        }
      }
    }
  }
}

}  // namespace

void ThreeBody::set_hermiticity(Hermiticity h) {
  if (h == hermiticity_) return;
  if (ms_ == nullptr) {
    hermiticity_ = h;
    return;
  }
  const bool to_packed = h != Hermiticity::kNonHermitian;
  if (packed() && to_packed) {
    // H <-> A: only the flag changes; an anti-Hermitian part has a zero
    // diagonal.
    hermiticity_ = h;
    if (h == Hermiticity::kAntiHermitian) {
      data_.SyncToHost();
      auto hs = data_.host_span();
      if (distributed()) {
        ZeroLocalPackedDiagonals(hs, ms_->three_body_rows(), space());
      } else {
        ZeroPackedDiagonals(hs, space().channels(), offsets_);
      }
      data_.SyncToDevice();
    }
    return;
  }
  if (distributed()) {
    throw NotImplementedError(
        "ThreeBody::set_hermiticity: converting a distributed part to or from "
        "non-Hermitian (dense) storage is not implemented (M12b-Q8)");
  }
  // Packed <-> dense: convert on the host.
  data_.SyncToHost();
  const std::int32_t old_h = HermiticitySign(hermiticity_);
  std::vector<std::int64_t> new_offsets = OffsetsFor(to_packed);
  exec::DeviceBuffer<double> out(static_cast<std::size_t>(new_offsets.back()),
                                 "imsrg::op::ThreeBody");
  const auto src = std::as_const(data_).host_span();
  auto dst = out.host_span();
  ConvertThreeBodyLayout(src, dst, space().channels(), offsets_, new_offsets,
                         to_packed, h, old_h);
  out.SyncToDevice();
  data_ = std::move(out);
  offsets_ = std::move(new_offsets);
  hermiticity_ = h;
}

namespace {

/// Sum of squares of the lower triangle (i >= j) of the local rows of a
/// distributed dense part: column k holds z(m, k), m = 0..n-1; m >= k counts.
double DistributedDenseSumOfSquares(const ThreeBody& t) {
  const std::vector<double> data = t.buffer().ToVector();
  double sum = 0.0;
  std::size_t pos = 0;
  for (const basis::RowRange& r :
       t.model_space_ptr()->three_body_rows().ranges()) {
    const std::int64_t n = t.dim(r.channel);
    for (std::int64_t k = r.row_begin.value(); k < r.row_end.value(); ++k) {
      for (std::int64_t m = k; m < n; ++m) {
        const double v = data[pos + static_cast<std::size_t>(m)];
        sum += v * v;
      }
      pos += static_cast<std::size_t>(n);
    }
  }
  return sum;
}

}  // namespace

ThreeBody GatherToRoot(const ThreeBody& t) {
  if (t.empty() || !t.distributed()) return t;
  const basis::ThreeBodyDistribution& dist =
      *t.model_space_ptr()->three_body_distribution();
  const comm::Communicator& comm = dist.comm;
  const int size = comm.size();
  const bool packed = t.packed();
  t.SyncToHost();
  const std::span<const double> local = t.host_span();

  // The partition is replicated: every rank checks every part and the total
  // against the int counts/displacements of the collective, so that all ranks
  // throw together (no rank is left waiting in Alltoallv).
  std::vector<std::int64_t> part_sizes(static_cast<std::size_t>(size));
  std::int64_t total = 0;
  for (int r = 0; r < size; ++r) {
    const std::int64_t cnt = dist.partition->part(r).StoredSize(packed);
    part_sizes[static_cast<std::size_t>(r)] = cnt;
    total += cnt;
    if (cnt > std::numeric_limits<int>::max() ||
        total > std::numeric_limits<int>::max()) {
      throw RuntimeError(
          "GatherToRoot: the three-body part has more than 2^31 - 1 stored "
          "doubles (part " +
          std::to_string(r) + " or the total); too large for one gather");
    }
  }
  std::vector<int> send_counts(static_cast<std::size_t>(size), 0);
  std::vector<int> send_displs(static_cast<std::size_t>(size), 0);
  std::vector<int> recv_counts(static_cast<std::size_t>(size), 0);
  std::vector<int> recv_displs(static_cast<std::size_t>(size), 0);
  send_counts[0] = static_cast<int>(local.size());
  if (comm.is_root()) {
    std::int64_t displ = 0;
    for (int r = 0; r < size; ++r) {
      recv_displs[static_cast<std::size_t>(r)] = static_cast<int>(displ);
      recv_counts[static_cast<std::size_t>(r)] =
          static_cast<int>(part_sizes[static_cast<std::size_t>(r)]);
      displ += part_sizes[static_cast<std::size_t>(r)];
    }
  }
  std::vector<double> recv(comm.is_root() ? static_cast<std::size_t>(total)
                                          : std::size_t{0});
  comm.Alltoallv(local, send_counts, send_displs, recv, recv_counts,
                 recv_displs);
  if (!comm.is_root()) return ThreeBody();

  ThreeBody out(t.model_space_ptr()->WithoutThreeBodyDistribution(),
                t.hermiticity());
  const auto offsets = out.block_offsets();
  const std::span<double> dst = out.host_span();
  for (int r = 0; r < size; ++r) {
    std::size_t pos =
        static_cast<std::size_t>(recv_displs[static_cast<std::size_t>(r)]);
    for (const basis::RowRange& rg : dist.partition->part(r).ranges()) {
      const std::int64_t n = t.dim(rg.channel);
      // Start of row k in its block: PackedIndex(n, k, k) packed, k n dense
      // (valid for k = n, the end of the block).
      const auto row_start = [&](std::int64_t k) {
        return packed ? ThreeBody::PackedIndex(n, k, k) : k * n;
      };
      const std::int64_t first = row_start(rg.row_begin.value());
      const std::int64_t len = row_start(rg.row_end.value()) - first;
      std::copy_n(recv.begin() + static_cast<std::ptrdiff_t>(pos), len,
                  dst.begin() + (offsets[rg.channel.ToSize()] + first));
      pos += static_cast<std::size_t>(len);
    }
  }
  out.SyncToDevice();
  return out;
}

double ImsrgppNorm(const ThreeBody& t) {
  if (t.empty()) return 0.0;
  if (t.distributed()) {
    // Collective: every rank takes part, also one without rows.
    double local = 0.0;
    if (t.size() > 0) {
      local = t.packed()
                  ? detail::SumOfSquares(t.buffer(), 0,
                                         static_cast<std::int64_t>(t.size()))
                  : DistributedDenseSumOfSquares(t);
    }
    const comm::Communicator& comm =
        t.model_space_ptr()->three_body_distribution()->comm;
    return std::sqrt(comm.ReduceBcastSum(local));
  }
  if (t.size() == 0) return 0.0;
  if (t.packed()) {
    return std::sqrt(detail::SumOfSquares(t.buffer(), 0,
                                          static_cast<std::int64_t>(t.size())));
  }
  const std::vector<double> data = t.buffer().ToVector();
  const auto offsets = t.block_offsets();
  double sum = 0.0;
  for (std::size_t ch = 0; ch + 1 < offsets.size(); ++ch) {
    const std::int64_t n = t.dim(Channel3Index{static_cast<std::int32_t>(ch)});
    for (std::int64_t j = 0; j < n; ++j) {
      for (std::int64_t i = j; i < n; ++i) {
        const double v =
            data[static_cast<std::size_t>(offsets[ch] + i + j * n)];
        sum += v * v;
      }
    }
  }
  return std::sqrt(sum);
}

}  // namespace imsrg::op
