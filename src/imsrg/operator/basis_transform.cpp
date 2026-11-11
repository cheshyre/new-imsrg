#include "imsrg/operator/basis_transform.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "imsrg/base/assert.hpp"
#include "imsrg/base/error.hpp"
#include "imsrg/basis/two_body_channels.hpp"
#include "imsrg/linalg/blas.hpp"

namespace imsrg::op {

namespace {

/// The two-body transformation matrix D of channel `ch` (imsrg++ HartreeFock.cc:938-956): rows
/// are the old kets, columns the new kets, both in local order. D is n x n, column-major.
linalg::Matrix<double> channel_transform(const basis::TwoBodyChannels& chans, ChannelIndex ch,
                                         linalg::MatrixView<const double> C) {
  const basis::TwoBodyChannel& info = chans.channel(ch);
  const auto kets = chans.channel_kets(ch);
  const std::int64_t n = info.num_kets;
  linalg::Matrix<double> D(n, n);
  for (std::int64_t i = 0; i < n; ++i) {
    const basis::Ket& old_ket = chans.ket(kets[static_cast<std::size_t>(i)]);
    const std::int64_t p = old_ket.p.value();
    const std::int64_t q = old_ket.q.value();
    const double phase = static_cast<double>(
        basis::exchange_phase(chans.orbits()[static_cast<std::size_t>(p)].j2,
                              chans.orbits()[static_cast<std::size_t>(q)].j2, info.two_j));
    for (std::int64_t j = 0; j < n; ++j) {
      const basis::Ket& new_ket = chans.ket(kets[static_cast<std::size_t>(j)]);
      const std::int64_t pp = new_ket.p.value();
      const std::int64_t qq = new_ket.q.value();
      double d = C(p, pp) * C(q, qq);
      if (p != q) d += C(q, pp) * C(p, qq) * phase;
      if (p == q) d *= std::sqrt(2.0);
      if (pp == qq) d /= std::sqrt(2.0);
      D(i, j) = d;
    }
  }
  return D;
}

}  // namespace

Operator transform_basis(const Operator& op, linalg::MatrixView<const double> C) {
  IMSRG_CHECK(!op.empty());
  const std::int64_t norb = op.model_space().num_orbits();
  if (C.rows != norb || C.cols != norb) {
    throw InvariantError("transform_basis: C must be norb x norb");
  }
  // The input is const: work on a copy whose host mirror is current.
  Operator src(op);
  src.sync_to_host();
  const Operator& csrc = src;
  Operator out(src.model_space_ptr(), src.spec());
  out.zero_body() = src.zero_body();

  // One-body: O' = C^T O C.
  {
    linalg::Matrix<double> tmp(norb, norb);
    linalg::gemm(1.0, csrc.one_body().host_view(), linalg::Op::NoTrans, C, linalg::Op::NoTrans, 0.0,
                 tmp.view());
    linalg::gemm(1.0, C, linalg::Op::Trans, tmp.const_view(), linalg::Op::NoTrans, 0.0,
                 out.one_body().host_view());
  }

  // Two-body: V' = D^T V D per channel.
  if (csrc.two_body().allocated()) {
    const basis::TwoBodyChannels& chans = csrc.two_body().channels();
    for (std::int32_t c = 0; c < chans.num_channels(); ++c) {
      const ChannelIndex ch{c};
      const std::int64_t n = chans.channel(ch).num_kets;
      if (n == 0) continue;
      const linalg::Matrix<double> D = channel_transform(chans, ch, C);
      linalg::Matrix<double> tmp(n, n);
      linalg::gemm(1.0, csrc.two_body().host_block(ch), linalg::Op::NoTrans, D.const_view(),
                   linalg::Op::NoTrans, 0.0, tmp.view());
      linalg::gemm(1.0, D.const_view(), linalg::Op::Trans, tmp.const_view(), linalg::Op::NoTrans,
                   0.0, out.two_body().host_block(ch));
    }
  }
  out.sync_to_device();
  return out;
}

Operator with_model_space(const Operator& op, std::shared_ptr<const basis::ModelSpace> ms) {
  if (op.empty()) throw InvariantError("with_model_space: empty operator");
  if (!ms) throw InvariantError("with_model_space: null model space");
  const auto a = op.model_space().two_body().orbits();
  const auto b = ms->two_body().orbits();
  if (a.size() != b.size() || !std::equal(a.begin(), a.end(), b.begin())) {
    throw InvariantError("with_model_space: the single-particle bases differ");
  }
  Operator src(op);
  src.sync_to_host();
  Operator out(std::move(ms), src.spec());
  out.zero_body() = src.zero_body();
  out.one_body().buffer().copy_from_host(src.one_body().buffer().host_span());
  if (src.two_body().allocated()) {
    out.two_body().matrix().buffer().copy_from_host(src.two_body().matrix().buffer().host_span());
  }
  out.sync_to_host();
  return out;
}

}  // namespace imsrg::op
