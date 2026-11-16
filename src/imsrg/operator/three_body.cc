// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/operator/three_body.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/base/error.h"
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
  data_ = exec::DeviceBuffer<double>(static_cast<std::size_t>(offsets_.back()),
                                     "imsrg::op::ThreeBody");
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
  const std::int64_t off = offsets_[ch.ToSize()];
  const std::int64_t a = i.value();
  const std::int64_t b = j.value();
  if (!packed()) return {off + a + b * n, 1};
  const int h = HermiticitySign(hermiticity_);
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
      ZeroPackedDiagonals(hs, space().channels(), offsets_);
      data_.SyncToDevice();
    }
    return;
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

double ImsrgppNorm(const ThreeBody& t) {
  if (t.empty() || t.size() == 0) return 0.0;
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
