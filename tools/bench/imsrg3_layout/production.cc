// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include "tools/bench/imsrg3_layout/production.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/base/types.h"
#include "imsrg/basis/three_body_channels.h"
#include "imsrg/basis/three_body_space.h"

namespace imsrg::layout3 {

std::shared_ptr<const basis::ModelSpace> ProductionModelSpace(
    const std::shared_ptr<const basis::ModelSpace>& ms, double de3max,
    double occnat3) {
  IMSRG_CHECK(ms != nullptr && ms->three_body() != nullptr,
              "ProductionModelSpace: needs a model space with 3B channels");
  basis::ThreeBodyCuts cuts = ms->three_body()->cuts();
  cuts.de3max = de3max;
  cuts.occnat3_cut = occnat3;
  cuts.natural_occupations.clear();
  return ms->WithThreeBody(cuts);
}

std::vector<double> ToCanonicalDense(const op::ThreeBody& z,
                                     const CanonicalSpace& space) {
  const CanonicalView cv = space.host_view();
  const basis::ThreeBodySpace& ks = z.space();
  std::vector<double> out(static_cast<std::size_t>(space.total_size()), 0.0);
  z.SyncToHost();
  std::int32_t c = 0;  // benchmark channel
  for (std::int32_t ch = 0; ch < ks.num_channels(); ++ch) {
    const basis::KeptChannel3& kc = ks.channel(Channel3Index{ch});
    if (kc.dim == 0) continue;
    IMSRG_CHECK(c < cv.num_channels, "ToCanonicalDense: too many channels");
    IMSRG_CHECK(cv.ch_two_j[c] == kc.two_j.twice &&
                    cv.ch_parity[c] == static_cast<std::int32_t>(kc.parity) &&
                    cv.ch_two_tz[c] == kc.two_tz.twice &&
                    cv.ch_dim[c] == kc.dim,
                "ToCanonicalDense: channel mismatch");
    const std::int64_t n = kc.dim;
    for (std::int64_t i = 0; i < n; ++i) {
      const std::int64_t pr = cv.ch_row_offset[c] + i;
      const basis::OrbitTriple& t = ks.triple(ks.row_triple(kc.row_offset + i));
      const std::int32_t pt = cv.row_triple[pr];
      IMSRG_CHECK(
          cv.row_two_jab[pr] == ks.row_two_jab(kc.row_offset + i).twice &&
              cv.tr_a[pt] == t.a.value() && cv.tr_b[pt] == t.b.value() &&
              cv.tr_c[pt] == t.c.value(),
          "ToCanonicalDense: row mismatch");
    }
    double* blk = out.data() + cv.ch_block_offset[c];
    for (std::int64_t j = 0; j < n; ++j) {
      for (std::int64_t i = 0; i < n; ++i) {
        blk[i + j * n] =
            z.get(Channel3Index{ch},
                  basis::KeptRow3Index{static_cast<std::int32_t>(i)},
                  basis::KeptRow3Index{static_cast<std::int32_t>(j)});
      }
    }
    ++c;
  }
  IMSRG_CHECK(c == cv.num_channels, "ToCanonicalDense: too few channels");
  return out;
}

}  // namespace imsrg::layout3
