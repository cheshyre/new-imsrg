// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "tools/bench/imsrg3_layout/layouts.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "imsrg/angmom/wigner.h"
#include "imsrg/base/assert.h"
#include "imsrg/base/types.h"
#include "imsrg/linalg/blas.h"
#include "imsrg/linalg/matrix_view.h"
#include "tools/bench/imsrg3_layout/recoupling.h"

namespace imsrg::layout3 {
namespace {

constexpr double kDropBelow = 1.0e-14;
constexpr double kSymmetryTol = 1.0e-12;
constexpr double kProjectorTol = 1.0e-10;

constexpr std::size_t U(std::int64_t i) { return static_cast<std::size_t>(i); }

template <class T>
exec::DeviceBuffer<T> ToBuffer(const std::vector<T>& v, const char* label) {
  exec::DeviceBuffer<T> buf(v.size(), label);
  buf.CopyFromHost(std::span<const T>(v));
  return buf;
}

template <class T>
std::int64_t BufferBytes(const exec::DeviceBuffer<T>& b) {
  return static_cast<std::int64_t>(b.size() * sizeof(T));
}

template <class T>
std::int64_t VectorBytes(const std::vector<T>& v) {
  return static_cast<std::int64_t>(v.size() * sizeof(T));
}

SpaceCounts CountsFromDims(std::span<const std::int32_t> dims,
                           std::int64_t num_triples) {
  SpaceCounts s;
  for (const std::int32_t d : dims) {
    const std::int64_t n = d;
    s.sum_n += n;
    s.sum_n2 += n * n;
    s.sum_packed += n * (n + 1) / 2;
    s.max_n = std::max(s.max_n, d);
  }
  s.num_channels = static_cast<std::int32_t>(dims.size());
  s.num_triples = num_triples;
  return s;
}

std::vector<linalg::BlockShape> ShapesFromDims(
    std::span<const std::int32_t> dims) {
  std::vector<linalg::BlockShape> shapes;
  shapes.reserve(dims.size());
  for (std::size_t c = 0; c < dims.size(); ++c) {
    const linalg::BlockIndex idx(static_cast<std::int32_t>(c));
    shapes.push_back({idx, idx, dims[c], dims[c]});
  }
  return shapes;
}

/// Fills `t` from the per-channel dims and the per-triple (channel-local) row
/// ranges, the weights per flat layout row and the outer map.
void FillTables(const std::vector<std::int32_t>& dim,
                const std::vector<std::int32_t>& tr_begin,
                const std::vector<std::int32_t>& tr_end,
                const std::vector<double>& weight, Csr outer, LayoutTables* t) {
  const std::size_t nc = dim.size();
  std::vector<std::int32_t> row_offset(nc);
  std::vector<std::int64_t> block_offset(nc + 1, 0);
  std::int32_t rows = 0;
  for (std::size_t c = 0; c < nc; ++c) {
    row_offset[c] = rows;
    rows += dim[c];
    block_offset[c + 1] =
        block_offset[c] + static_cast<std::int64_t>(dim[c]) * dim[c];
  }
  IMSRG_CHECK(static_cast<std::size_t>(rows) == weight.size(),
              "layout: weight table does not match the row count");
  t->dim = dim;
  t->block_offset = block_offset;
  t->ch_dim = ToBuffer(dim, "imsrg3_layout::ch_dim");
  t->ch_row_offset = ToBuffer(row_offset, "imsrg3_layout::ch_row_offset");
  t->ch_block_offset = ToBuffer(block_offset, "imsrg3_layout::ch_block_offset");
  t->tr_row_begin = ToBuffer(tr_begin, "imsrg3_layout::tr_row_begin");
  t->tr_row_end = ToBuffer(tr_end, "imsrg3_layout::tr_row_end");
  t->energy_weight = ToBuffer(weight, "imsrg3_layout::energy_weight");
  t->outer = std::move(outer);
}

/// Accumulates the rows of an outer map: each row is a linear combination of
/// rows of a source CSR matrix; the columns come out sorted and summed.
class OuterBuilder {
 public:
  /// Adds `coef` times row `row` of `m` to the row under construction.
  void Add(const CsrView& m, std::int64_t row, double coef) {
    for (std::int64_t k = m.ptr[row]; k < m.ptr[row + 1]; ++k) {
      entries_.emplace_back(m.col[k], coef * m.val[k]);
    }
  }

  /// Closes the row: sorts by column, sums duplicates, drops |sum| < 1e-14.
  void EndRow() {
    std::sort(entries_.begin(), entries_.end(),
              [](const auto& x, const auto& y) { return x.first < y.first; });
    std::size_t i = 0;
    while (i < entries_.size()) {
      const std::int32_t c = entries_[i].first;
      double sum = 0.0;
      for (; i < entries_.size() && entries_[i].first == c; ++i) {
        sum += entries_[i].second;
      }
      if (std::abs(sum) >= kDropBelow) {
        col_.push_back(c);
        val_.push_back(sum);
      }
    }
    entries_.clear();
    ptr_.push_back(static_cast<std::int64_t>(col_.size()));
  }

  Csr Finish() const { return Csr::FromHost(ptr_, col_, val_); }

 private:
  std::vector<std::int64_t> ptr_{0};
  std::vector<std::int32_t> col_;
  std::vector<double> val_;
  std::vector<std::pair<std::int32_t, double>> entries_;
};

/// s_T = sqrt(sym_T / 6) of flat triple `t`.
double TripleScale(const CanonicalView& cv, std::int32_t t) {
  return std::sqrt(static_cast<double>(cv.tr_sym[t]) / 6.0);
}

/// 6j provider calling `angmom::Wigner6j` (no table; large-emax counts).
struct WignerSixJ {
  double operator()(TwoJ j1, TwoJ j2, TwoJ j12, TwoJ j3, TwoJ j,
                    TwoJ j23) const {
    return angmom::Wigner6j(j1, j2, j12, j3, j, j23);
  }
  double Hat(int j) const { return std::sqrt(2.0 * j + 1.0); }
};

/// Whether the antisymmetric subspace of a triple is a proper subspace.
enum class Projected : std::int8_t { kNone, kBcEqual, kAllEqual };

Projected ProjectionKind(const CanonicalView& cv, std::int32_t t) {
  if (cv.tr_b[t] == cv.tr_c[t]) {
    return cv.tr_a[t] == cv.tr_b[t] ? Projected::kAllEqual
                                    : Projected::kBcEqual;
  }
  return Projected::kNone;
}

/// Adds `scale * M_p` (M_p[J', J] over the kept rows of triple `t`,
/// column-major n x n) to `q`.
template <class SixJ>
void AddRecoupling(const CanonicalView& cv, std::span<const basis::Orbit> orbs,
                   std::int32_t t, Perm p, const SixJ& sixj, double scale,
                   std::vector<double>* q) {
  const std::int32_t ch = cv.tr_channel[t];
  const std::int32_t n = cv.tr_row_end[t] - cv.tr_row_begin[t];
  const TwoJ ja = orbs[U(cv.tr_a[t])].j2;
  const TwoJ jb = orbs[U(cv.tr_b[t])].j2;
  const TwoJ jc = orbs[U(cv.tr_c[t])].j2;
  const TwoJ two_j{cv.ch_two_j[ch]};
  const std::int32_t* jab =
      cv.row_two_jab + cv.ch_row_offset[ch] + cv.tr_row_begin[t];
  for (std::int32_t j = 0; j < n; ++j) {
    for (std::int32_t jp = 0; jp < n; ++jp) {
      (*q)[U(jp + static_cast<std::int64_t>(j) * n)] +=
          scale * PermutationPhase(p) *
          RecouplingCoefficient(p, ja, jb, jc, jab[jp] / 2, jab[j] / 2, two_j,
                                sixj);
    }
  }
}

/// Q = (I + sum M_p) / m for the projected triples (m = 2 or 3).
template <class SixJ>
std::vector<double> BuildQ(const CanonicalView& cv,
                           std::span<const basis::Orbit> orbs, std::int32_t t,
                           Projected kind, const SixJ& sixj) {
  const std::int32_t n = cv.tr_row_end[t] - cv.tr_row_begin[t];
  std::vector<double> q(U(static_cast<std::int64_t>(n) * n), 0.0);
  const double scale = kind == Projected::kBcEqual ? 0.5 : 1.0 / 3.0;
  for (std::int32_t i = 0; i < n; ++i) {
    q[U(i + static_cast<std::int64_t>(i) * n)] = scale;
  }
  AddRecoupling(cv, orbs, t, Perm::kACB, sixj, scale, &q);
  if (kind == Projected::kAllEqual) {
    AddRecoupling(cv, orbs, t, Perm::kCBA, sixj, scale, &q);
  }
  return q;
}

/// Dense block-diagonal conversion matrices of layout B for channel `ch`:
/// S (dB x nA) = blockdiag(s_T V_T^T) and T (nA x dB) = blockdiag(V_T / s_T).
void BuildConversion(const CanonicalSpace& space, const LayoutTables& tables,
                     const std::vector<std::int64_t>& v_offset,
                     const std::vector<std::int32_t>& v_dim,
                     const std::vector<double>& v_all, std::int32_t ch,
                     std::vector<double>* s_mat, std::vector<double>* t_mat) {
  const CanonicalView cv = space.host_view();
  const std::int64_t na = cv.ch_dim[ch];
  const std::int64_t nb = tables.dim[U(ch)];
  s_mat->assign(U(nb * na), 0.0);
  t_mat->assign(U(na * nb), 0.0);
  const std::int32_t* tbb = tables.tr_row_begin.host_span().data();
  for (std::int32_t t = cv.ch_triple_begin[ch]; t < cv.ch_triple_end[ch]; ++t) {
    const std::int64_t n = cv.tr_row_end[t] - cv.tr_row_begin[t];
    const std::int64_t a0 = cv.tr_row_begin[t];
    const std::int64_t b0 = tbb[t];
    const std::int64_t d = v_dim[U(t)];
    const double s = TripleScale(cv, t);
    for (std::int64_t k = 0; k < d; ++k) {
      for (std::int64_t j = 0; j < n; ++j) {
        const double v = v_offset[U(t)] >= 0
                             ? v_all[U(v_offset[U(t)] + j + k * n)]
                             : (j == k ? 1.0 : 0.0);
        (*s_mat)[U((b0 + k) + (a0 + j) * nb)] = s * v;
        (*t_mat)[U((a0 + j) + (b0 + k) * na)] = v / s;
      }
    }
  }
}

/// out = X * in * X^T for X (m x n), in (n x n), out (m x m), column-major.
void Sandwich(const std::vector<double>& x, std::int64_t m, std::int64_t n,
              std::span<const double> in, std::span<double> out) {
  std::vector<double> tmp(U(m * n), 0.0);
  linalg::Gemm(1.0, linalg::MatrixView<const double>{x.data(), m, n, m},
               linalg::Op::kNoTrans,
               linalg::MatrixView<const double>{in.data(), n, n, n},
               linalg::Op::kNoTrans, 0.0,
               linalg::MatrixView<double>{tmp.data(), m, n, m});
  linalg::Gemm(
      1.0, linalg::MatrixView<const double>{tmp.data(), m, n, m},
      linalg::Op::kNoTrans, linalg::MatrixView<const double>{x.data(), m, n, m},
      linalg::Op::kTrans, 0.0, linalg::MatrixView<double>{out.data(), m, m, m});
}

}  // namespace

LayoutView LayoutTables::device_view() const noexcept {
  LayoutView v;
  v.ch_dim = ch_dim.device_data();
  v.ch_row_offset = ch_row_offset.device_data();
  v.ch_block_offset = ch_block_offset.device_data();
  v.tr_row_begin = tr_row_begin.device_data();
  v.tr_row_end = tr_row_end.device_data();
  v.outer = outer.device_view();
  v.energy_weight = energy_weight.device_data();
  v.num_channels = static_cast<std::int32_t>(dim.size());
  return v;
}

std::int64_t LayoutTables::Bytes() const noexcept {
  return BufferBytes(ch_dim) + BufferBytes(ch_row_offset) +
         BufferBytes(ch_block_offset) + BufferBytes(tr_row_begin) +
         BufferBytes(tr_row_end) + BufferBytes(energy_weight);
}

// ---------------------------------------------------------------- Layout A

LayoutA::LayoutA(const CanonicalSpace& space, const PairThirdSpace& pt)
    : space_(&space) {
  const CanonicalView cv = space.host_view();
  std::vector<std::int32_t> dim(cv.ch_dim, cv.ch_dim + cv.num_channels);
  std::vector<std::int32_t> tb(cv.tr_row_begin,
                               cv.tr_row_begin + cv.num_triples);
  std::vector<std::int32_t> te(cv.tr_row_end, cv.tr_row_end + cv.num_triples);
  std::vector<double> weight(U(cv.num_rows));
  for (std::int64_t r = 0; r < cv.num_rows; ++r) {
    weight[U(r)] = TripleScale(cv, cv.row_triple[r]);
  }
  FillTables(dim, tb, te, weight, pt.RMap(), &t_);
}

SpaceCounts LayoutA::Counts() const {
  return CountsFromDims(t_.dim, space_->NumTriples());
}

std::int64_t LayoutA::MetadataBytes() const noexcept {
  return t_.Bytes() + t_.outer.Bytes();
}

std::vector<linalg::BlockShape> LayoutA::Shapes() const {
  return ShapesFromDims(t_.dim);
}

void LayoutA::ToCanonical(std::int32_t ch, std::span<const double> in,
                          std::span<double> out) const {
  const std::int64_t n = t_.dim[U(ch)];
  IMSRG_CHECK(static_cast<std::int64_t>(in.size()) == n * n &&
                  static_cast<std::int64_t>(out.size()) == n * n,
              "LayoutA::ToCanonical: block size mismatch");
  std::copy(in.begin(), in.end(), out.begin());
}

void LayoutA::FromCanonical(std::int32_t ch, std::span<const double> in,
                            std::span<double> out) const {
  const std::int64_t n = t_.dim[U(ch)];
  IMSRG_CHECK(static_cast<std::int64_t>(in.size()) == n * n &&
                  static_cast<std::int64_t>(out.size()) == n * n,
              "LayoutA::FromCanonical: block size mismatch");
  std::copy(in.begin(), in.end(), out.begin());
}

// ---------------------------------------------------------------- Layout B

LayoutB::LayoutB(const CanonicalSpace& space, const PairThirdSpace& pt,
                 const SixJ3Table& sixj)
    : space_(&space) {
  const CanonicalView cv = space.host_view();
  const SixJ3View sj = sixj.host_view();
  const CsrView rm = pt.RMap().host_view();
  const auto orbs = space.model_space().basis().orbits();

  std::vector<std::int32_t> dim(U(cv.num_channels), 0);
  std::vector<std::int32_t> tb(U(cv.num_triples));
  std::vector<std::int32_t> te(U(cv.num_triples));
  std::vector<double> weight;
  v_offset_.assign(U(cv.num_triples), -1);
  v_dim_.assign(U(cv.num_triples), 0);
  OuterBuilder outer;

  for (std::int32_t ch = 0; ch < cv.num_channels; ++ch) {
    std::int32_t local = 0;
    for (std::int32_t t = cv.ch_triple_begin[ch]; t < cv.ch_triple_end[ch];
         ++t) {
      const std::int32_t n = cv.tr_row_end[t] - cv.tr_row_begin[t];
      const std::int64_t row0 = cv.ch_row_offset[ch] + cv.tr_row_begin[t];
      const Projected kind = ProjectionKind(cv, t);
      std::int32_t d = n;
      std::int64_t voff = -1;
      if (kind != Projected::kNone) {
        const std::vector<double> q = BuildQ(cv, orbs, t, kind, sj);
        double asym = 0.0;
        for (std::int32_t i = 0; i < n; ++i) {
          for (std::int32_t j = 0; j < i; ++j) {
            asym = std::max(asym, std::abs(q[U(i + std::int64_t{j} * n)] -
                                           q[U(j + std::int64_t{i} * n)]));
          }
        }
        IMSRG_CHECK(asym <= kSymmetryTol,
                    "LayoutB: the antisymmetrizer Q is not symmetric");
        std::vector<double> evals(U(n));
        std::vector<double> evecs(U(std::int64_t{n} * n));
        if (n > 0) {
          linalg::Eigh(linalg::MatrixView<const double>{q.data(), n, n, n},
                       evals,
                       linalg::MatrixView<double>{evecs.data(), n, n, n});
        }
        std::int32_t first = n;
        for (std::int32_t k = 0; k < n; ++k) {
          const double e = evals[U(k)];
          IMSRG_CHECK(std::abs(e) <= kProjectorTol ||
                          std::abs(e - 1.0) <= kProjectorTol,
                      "LayoutB: eigenvalue of Q is neither 0 nor 1");
          if (e > 0.5 && first == n) first = k;
        }
        d = n - first;
        voff = static_cast<std::int64_t>(v_.size());
        v_.insert(v_.end(), evecs.begin() + std::int64_t{first} * n,
                  evecs.end());
      }
      v_offset_[U(t)] = voff;
      v_dim_[U(t)] = d;
      tb[U(t)] = local;
      local += d;
      te[U(t)] = local;
      const double s = TripleScale(cv, t);
      for (std::int32_t k = 0; k < d; ++k) {
        for (std::int32_t j = 0; j < n; ++j) {
          const double coef = voff >= 0 ? v_[U(voff + j + std::int64_t{k} * n)]
                                        : (j == k ? 1.0 : 0.0);
          if (coef != 0.0) outer.Add(rm, row0 + j, s * coef);
        }
        outer.EndRow();
        weight.push_back(1.0);
      }
    }
    dim[U(ch)] = local;
  }
  FillTables(dim, tb, te, weight, outer.Finish(), &t_);
}

SpaceCounts LayoutB::Counts() const {
  return CountsFromDims(t_.dim, space_->NumTriples());
}

std::int64_t LayoutB::MetadataBytes() const noexcept {
  return t_.Bytes() + t_.outer.Bytes() + VectorBytes(v_offset_) +
         VectorBytes(v_dim_) + VectorBytes(v_);
}

std::vector<linalg::BlockShape> LayoutB::Shapes() const {
  return ShapesFromDims(t_.dim);
}

void LayoutB::FromCanonical(std::int32_t ch, std::span<const double> in,
                            std::span<double> out) const {
  const std::int64_t na = space_->host_view().ch_dim[ch];
  const std::int64_t nb = t_.dim[U(ch)];
  IMSRG_CHECK(static_cast<std::int64_t>(in.size()) == na * na &&
                  static_cast<std::int64_t>(out.size()) == nb * nb,
              "LayoutB::FromCanonical: block size mismatch");
  std::vector<double> s_mat;
  std::vector<double> t_mat;
  BuildConversion(*space_, t_, v_offset_, v_dim_, v_, ch, &s_mat, &t_mat);
  Sandwich(s_mat, nb, na, in, out);
}

void LayoutB::ToCanonical(std::int32_t ch, std::span<const double> in,
                          std::span<double> out) const {
  const std::int64_t na = space_->host_view().ch_dim[ch];
  const std::int64_t nb = t_.dim[U(ch)];
  IMSRG_CHECK(static_cast<std::int64_t>(in.size()) == nb * nb &&
                  static_cast<std::int64_t>(out.size()) == na * na,
              "LayoutB::ToCanonical: block size mismatch");
  std::vector<double> s_mat;
  std::vector<double> t_mat;
  BuildConversion(*space_, t_, v_offset_, v_dim_, v_, ch, &s_mat, &t_mat);
  Sandwich(t_mat, na, nb, in, out);
}

SpaceCounts LayoutB::CountOnly(const CanonicalSpace& space) {
  const CanonicalView cv = space.host_view();
  const auto orbs = space.model_space().basis().orbits();
  const WignerSixJ sixj;
  std::vector<std::int32_t> dim(U(cv.num_channels), 0);
  for (std::int32_t ch = 0; ch < cv.num_channels; ++ch) {
    for (std::int32_t t = cv.ch_triple_begin[ch]; t < cv.ch_triple_end[ch];
         ++t) {
      const std::int32_t n = cv.tr_row_end[t] - cv.tr_row_begin[t];
      const Projected kind = ProjectionKind(cv, t);
      std::int32_t d = n;
      if (kind != Projected::kNone) {
        const std::vector<double> q = BuildQ(cv, orbs, t, kind, sixj);
        double trace = 0.0;
        for (std::int32_t i = 0; i < n; ++i) {
          trace += q[U(i + std::int64_t{i} * n)];
        }
        d = static_cast<std::int32_t>(std::llround(trace));
        IMSRG_CHECK(std::abs(trace - d) <= 1.0e-6,
                    "LayoutB::CountOnly: trace of Q is not an integer");
      }
      dim[U(ch)] += d;
    }
  }
  return CountsFromDims(dim, cv.num_triples);
}

// ---------------------------------------------------------------- Layout C

LayoutC::LayoutC(const CanonicalSpace& space, const PairThirdSpace& pt)
    : space_(&space), pt_(&pt) {
  const CanonicalView cv = space.host_view();
  const PairThirdView pv = pt.host_view();
  const CsrView rm = pv.r;
  const CsrView em = pt.EMap().host_view();

  std::vector<std::int32_t> dim(pv.ch_num_pt, pv.ch_num_pt + pv.num_channels);
  std::vector<std::int32_t> tb(pv.tr_pt_begin, pv.tr_pt_begin + cv.num_triples);
  std::vector<std::int32_t> te(pv.tr_pt_end, pv.tr_pt_end + cv.num_triples);
  std::vector<double> weight(U(pv.num_states), 0.0);
  canonical_row_.assign(U(cv.num_rows), -1);
  OuterBuilder outer;

  // Layout rows are in flat state order, so the outer map is built channel by
  // channel, triple by triple, state by state.
  for (std::int32_t ch = 0; ch < cv.num_channels; ++ch) {
    const std::int64_t state0 = pv.ch_pt_offset[ch];
    for (std::int32_t t = cv.ch_triple_begin[ch]; t < cv.ch_triple_end[ch];
         ++t) {
      const std::int64_t row0 = cv.ch_row_offset[ch];
      const double s = TripleScale(cv, t);
      for (std::int32_t st = pv.tr_pt_begin[t]; st < pv.tr_pt_end[t]; ++st) {
        const std::int64_t flat = state0 + st;
        for (std::int64_t k = em.ptr[flat]; k < em.ptr[flat + 1]; ++k) {
          outer.Add(rm, row0 + em.col[k], em.val[k]);
        }
        outer.EndRow();
      }
      // The ABC state of each canonical row (p = a, q = b, r = c).
      for (std::int32_t lr = cv.tr_row_begin[t]; lr < cv.tr_row_end[t]; ++lr) {
        const std::int64_t flat_row = row0 + lr;
        const std::int32_t two_jab = cv.row_two_jab[flat_row];
        std::int32_t found = -1;
        for (std::int32_t st = pv.tr_pt_begin[t]; st < pv.tr_pt_end[t]; ++st) {
          const std::int64_t f = state0 + st;
          if (pv.pt_p[f] == cv.tr_a[t] && pv.pt_q[f] == cv.tr_b[t] &&
              pv.pt_r[f] == cv.tr_c[t] && pv.pt_two_jpq[f] == two_jab) {
            found = st;
            break;
          }
        }
        IMSRG_CHECK(found >= 0, "LayoutC: canonical row has no ABC state");
        canonical_row_[U(flat_row)] = found;
        weight[U(state0 + found)] = s;
      }
    }
  }
  FillTables(dim, tb, te, weight, outer.Finish(), &t_);
}

SpaceCounts LayoutC::Counts() const {
  return CountsFromDims(t_.dim, space_->NumTriples());
}

std::int64_t LayoutC::MetadataBytes() const noexcept {
  return t_.Bytes() + t_.outer.Bytes() + VectorBytes(canonical_row_);
}

std::vector<linalg::BlockShape> LayoutC::Shapes() const {
  return ShapesFromDims(t_.dim);
}

void LayoutC::FromCanonical(std::int32_t ch, std::span<const double> in,
                            std::span<double> out) const {
  const CanonicalView cv = space_->host_view();
  const PairThirdView pv = pt_->host_view();
  const CsrView em = pt_->EMap().host_view();
  const std::int64_t na = cv.ch_dim[ch];
  const std::int64_t nc = t_.dim[U(ch)];
  IMSRG_CHECK(static_cast<std::int64_t>(in.size()) == na * na &&
                  static_cast<std::int64_t>(out.size()) == nc * nc,
              "LayoutC::FromCanonical: block size mismatch");
  std::vector<double> e(U(nc * na), 0.0);
  for (std::int64_t s = 0; s < nc; ++s) {
    const std::int64_t flat = pv.ch_pt_offset[ch] + s;
    for (std::int64_t k = em.ptr[flat]; k < em.ptr[flat + 1]; ++k) {
      e[U(s + em.col[k] * nc)] = em.val[k];
    }
  }
  Sandwich(e, nc, na, in, out);
}

void LayoutC::ToCanonical(std::int32_t ch, std::span<const double> in,
                          std::span<double> out) const {
  const CanonicalView cv = space_->host_view();
  const std::int64_t na = cv.ch_dim[ch];
  const std::int64_t nc = t_.dim[U(ch)];
  IMSRG_CHECK(static_cast<std::int64_t>(in.size()) == nc * nc &&
                  static_cast<std::int64_t>(out.size()) == na * na,
              "LayoutC::ToCanonical: block size mismatch");
  const std::int32_t* g = canonical_row_.data() + cv.ch_row_offset[ch];
  for (std::int64_t j = 0; j < na; ++j) {
    for (std::int64_t i = 0; i < na; ++i) {
      out[U(i + j * na)] = in[U(g[i] + std::int64_t{g[j]} * nc)];
    }
  }
}

}  // namespace imsrg::layout3
