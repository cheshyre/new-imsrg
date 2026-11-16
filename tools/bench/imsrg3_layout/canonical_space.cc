// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "tools/bench/imsrg3_layout/canonical_space.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/basis/three_body_channels.h"

namespace imsrg::layout3 {

namespace {

// Field order of the int32 tables in CanonicalSpace::i32_.
enum Field : int {
  kChTwoJ,
  kChParity,
  kChTwoTz,
  kChDim,
  kChRowOffset,
  kChTripleBegin,
  kChTripleEnd,
  kTrA,
  kTrB,
  kTrC,
  kTrChannel,
  kTrRowBegin,
  kTrRowEnd,
  kTrSym,
  kRowTwoJab,
  kRowTriple,
  kNumFields
};

}  // namespace

CanonicalSpace::CanonicalSpace(std::shared_ptr<const basis::ModelSpace> ms,
                               double de3max, double occnat3_cut)
    : ms_(std::move(ms)) {
  IMSRG_CHECK(ms_ != nullptr && ms_->three_body() != nullptr,
              "CanonicalSpace: the model space has no three-body channels");
  const basis::ThreeBodyChannels& tbc = *ms_->three_body();
  const basis::SingleParticleBasis& basis = ms_->basis();
  const basis::Reference& ref = ms_->reference();

  std::vector<std::vector<std::int32_t>> f(kNumFields);
  std::vector<std::int64_t> block_offset{0};
  for (std::int32_t bc = 0; bc < tbc.num_channels(); ++bc) {
    const basis::ThreeBodyChannel& chan = tbc.channel(Channel3Index{bc});
    const auto first_triple = static_cast<std::int32_t>(f[kTrA].size());
    const std::int32_t canonical_channel = num_channels_;
    std::int32_t dim = 0;
    std::int32_t cur_a = -1;
    std::int32_t cur_b = -1;
    std::int32_t cur_c = -1;
    for (const Ket3Index k : tbc.channel_kets(Channel3Index{bc})) {
      if (!tbc.PassesConfigurationCut(k, basis, ref, de3max, occnat3_cut)) {
        continue;
      }
      const basis::Ket3& ket = tbc.ket(k);
      const std::int32_t a = ket.p.value();
      const std::int32_t b = ket.q.value();
      const std::int32_t c = ket.r.value();
      if (a != cur_a || b != cur_b || c != cur_c) {
        if (cur_a >= 0) f[kTrRowEnd].back() = dim;
        f[kTrA].push_back(a);
        f[kTrB].push_back(b);
        f[kTrC].push_back(c);
        f[kTrChannel].push_back(canonical_channel);
        f[kTrRowBegin].push_back(dim);
        f[kTrRowEnd].push_back(dim);
        f[kTrSym].push_back((a == b && b == c)   ? 1
                            : (a == b || b == c) ? 3
                                                 : 6);
        cur_a = a;
        cur_b = b;
        cur_c = c;
      }
      f[kRowTwoJab].push_back(ket.two_jpq.twice);
      f[kRowTriple].push_back(static_cast<std::int32_t>(f[kTrA].size()) - 1);
      ++dim;
    }
    if (dim == 0) continue;
    f[kTrRowEnd].back() = dim;
    f[kChTwoJ].push_back(chan.two_j.twice);
    f[kChParity].push_back(static_cast<std::int32_t>(chan.parity));
    f[kChTwoTz].push_back(chan.two_tz.twice);
    f[kChDim].push_back(dim);
    f[kChRowOffset].push_back(static_cast<std::int32_t>(num_rows_));
    f[kChTripleBegin].push_back(first_triple);
    f[kChTripleEnd].push_back(static_cast<std::int32_t>(f[kTrA].size()));
    block_offset.push_back(block_offset.back() +
                           static_cast<std::int64_t>(dim) * dim);
    num_rows_ += dim;
    ++num_channels_;
  }
  num_triples_ = static_cast<std::int32_t>(f[kTrA].size());

  std::size_t total = 0;
  for (int i = 0; i < kNumFields; ++i) {
    field_offset_[i] = static_cast<std::int64_t>(total);
    total += f[static_cast<std::size_t>(i)].size();
  }
  i32_ = exec::DeviceBuffer<std::int32_t>(std::max<std::size_t>(total, 1),
                                          "imsrg3_layout::CanonicalSpace");
  std::span<std::int32_t> out = i32_.host_span();
  for (int i = 0; i < kNumFields; ++i) {
    std::copy(f[static_cast<std::size_t>(i)].begin(),
              f[static_cast<std::size_t>(i)].end(),
              out.begin() + field_offset_[i]);
  }
  i32_.SyncToDevice();
  block_offset_ = exec::DeviceBuffer<std::int64_t>(
      block_offset.size(), "imsrg3_layout::CanonicalSpace::block_offset");
  block_offset_.CopyFromHost(block_offset);
}

std::int64_t CanonicalSpace::total_size() const noexcept {
  return block_offset_.host_span()[static_cast<std::size_t>(num_channels_)];
}

SpaceCounts CanonicalSpace::Counts() const {
  const CanonicalView v = host_view();
  SpaceCounts s;
  for (std::int32_t c = 0; c < num_channels_; ++c) {
    const std::int64_t n = v.ch_dim[c];
    s.sum_n += n;
    s.sum_n2 += n * n;
    s.sum_packed += n * (n + 1) / 2;
    s.max_n = std::max(s.max_n, static_cast<std::int32_t>(n));
  }
  s.num_channels = num_channels_;
  s.num_triples = num_triples_;
  return s;
}

std::int64_t CanonicalSpace::MetadataBytes() const noexcept {
  return static_cast<std::int64_t>(i32_.size() * sizeof(std::int32_t) +
                                   block_offset_.size() * sizeof(std::int64_t));
}

namespace {

CanonicalView MakeView(const std::int32_t* base, const std::int64_t* offsets,
                       const std::int64_t* field_offset,
                       std::int32_t num_channels, std::int32_t num_triples,
                       std::int64_t num_rows) {
  CanonicalView v;
  v.ch_two_j = base + field_offset[kChTwoJ];
  v.ch_parity = base + field_offset[kChParity];
  v.ch_two_tz = base + field_offset[kChTwoTz];
  v.ch_dim = base + field_offset[kChDim];
  v.ch_row_offset = base + field_offset[kChRowOffset];
  v.ch_triple_begin = base + field_offset[kChTripleBegin];
  v.ch_triple_end = base + field_offset[kChTripleEnd];
  v.ch_block_offset = offsets;
  v.tr_a = base + field_offset[kTrA];
  v.tr_b = base + field_offset[kTrB];
  v.tr_c = base + field_offset[kTrC];
  v.tr_channel = base + field_offset[kTrChannel];
  v.tr_row_begin = base + field_offset[kTrRowBegin];
  v.tr_row_end = base + field_offset[kTrRowEnd];
  v.tr_sym = base + field_offset[kTrSym];
  v.row_two_jab = base + field_offset[kRowTwoJab];
  v.row_triple = base + field_offset[kRowTriple];
  v.num_channels = num_channels;
  v.num_triples = num_triples;
  v.num_rows = num_rows;
  return v;
}

}  // namespace

CanonicalView CanonicalSpace::host_view() const noexcept {
  return MakeView(i32_.host_span().data(), block_offset_.host_span().data(),
                  field_offset_, num_channels_, num_triples_, num_rows_);
}

CanonicalView CanonicalSpace::device_view() const noexcept {
  return MakeView(i32_.device_data(), block_offset_.device_data(),
                  field_offset_, num_channels_, num_triples_, num_rows_);
}

double ImsrgppNorm(const CanonicalSpace& space,
                   std::span<const double> canonical) {
  IMSRG_CHECK(static_cast<std::int64_t>(canonical.size()) == space.total_size(),
              "ImsrgppNorm: buffer size does not match the canonical space");
  const CanonicalView v = space.host_view();
  double sum = 0.0;
  for (std::int32_t c = 0; c < v.num_channels; ++c) {
    const std::int64_t n = v.ch_dim[c];
    const double* z = canonical.data() + v.ch_block_offset[c];
    for (std::int64_t j = 0; j < n; ++j) {
      for (std::int64_t i = 0; i <= j; ++i) {
        const double x = z[i + j * n];
        sum += x * x;
      }
    }
  }
  return std::sqrt(sum);
}

}  // namespace imsrg::layout3
