#include "imsrg/operator/two_body.hpp"

#include <cmath>
#include <utility>
#include <vector>

#include "imsrg/base/assert.hpp"
#include "imsrg/base/error.hpp"

namespace imsrg::op {

namespace {

/// One side (a bra or a ket) of an element: channel, local ket index and exchange phase.
struct Side {
  ChannelIndex ch;
  LocalKetIndex idx;
  std::int32_t phase;
};

/// Location of a normalized matrix element in the stored block.
struct Location {
  ChannelIndex ch;
  LocalKetIndex i;  // bra
  LocalKetIndex j;  // ket
  std::int32_t phase;
};

std::optional<Side> locate_side(const basis::TwoBodyChannels& tb, OrbitIndex a, OrbitIndex b,
                                TwoJ two_j) {
  const auto orbits = tb.orbits();
  const basis::Orbit& oa = orbits[a.to_size()];
  const basis::Orbit& ob = orbits[b.to_size()];
  const Parity par = (oa.parity() == ob.parity()) ? Parity::Even : Parity::Odd;
  const auto ch = tb.find_channel(two_j, par, TwoTz{oa.tz2.twice + ob.tz2.twice});
  if (!ch) return std::nullopt;
  const bool swap = a.value() > b.value();
  const OrbitIndex p = swap ? b : a;
  const OrbitIndex q = swap ? a : b;
  const auto local = tb.local_index(*ch, tb.ket_index(p, q));
  if (!local) return std::nullopt;
  return Side{*ch, *local, swap ? basis::exchange_phase(oa.j2, ob.j2, two_j) : 1};
}

std::optional<Location> locate(const basis::TwoBodyChannels& tb, OrbitIndex a, OrbitIndex b,
                               OrbitIndex c, OrbitIndex d, TwoJ two_j) {
  const auto bra = locate_side(tb, a, b, two_j);
  if (!bra) return std::nullopt;
  const auto ket = locate_side(tb, c, d, two_j);
  if (!ket || ket->ch != bra->ch) return std::nullopt;
  return Location{bra->ch, bra->idx, ket->idx, bra->phase * ket->phase};
}

}  // namespace

TwoBody::TwoBody(std::shared_ptr<const basis::ModelSpace> ms, const OperatorSpec& spec)
    : ms_(std::move(ms)), hermiticity_(spec.hermiticity) {
  IMSRG_CHECK(ms_ != nullptr);
  if (spec.particle_rank < 2) return;
  const auto& tb = ms_->two_body();
  std::vector<linalg::BlockShape> shapes;
  shapes.reserve(static_cast<std::size_t>(tb.num_channels()));
  for (std::int32_t ch = 0; ch < tb.num_channels(); ++ch) {
    const std::int64_t nk = tb.channel(ChannelIndex{ch}).num_kets;
    shapes.push_back({linalg::BlockIndex{ch}, linalg::BlockIndex{ch}, nk, nk});
  }
  matrix_ = linalg::BlockSparseMatrix<double>(shapes, "imsrg::op::TwoBody");
}

const basis::TwoBodyChannels& TwoBody::channels() const {
  IMSRG_CHECK(ms_ != nullptr);
  return ms_->two_body();
}

std::optional<std::size_t> TwoBody::block_index(ChannelIndex ch_bra,
                                                ChannelIndex ch_ket) const noexcept {
  if (ch_bra != ch_ket || ch_bra.value() < 0 || ch_bra.to_size() >= matrix_.num_blocks()) {
    return std::nullopt;
  }
  return ch_bra.to_size();
}

double TwoBody::get(ChannelIndex ch_bra, ChannelIndex ch_ket, LocalKetIndex i,
                    LocalKetIndex j) const {
  const auto k = block_index(ch_bra, ch_ket);
  if (!k) return 0.0;
  const auto blk = matrix_.host_block(*k);
  IMSRG_ASSERT(i.value() >= 0 && i.value() < blk.rows && j.value() >= 0 && j.value() < blk.cols);
  return blk(i.value(), j.value());
}

void TwoBody::set(ChannelIndex ch_bra, ChannelIndex ch_ket, LocalKetIndex i, LocalKetIndex j,
                  double v) {
  const auto k = block_index(ch_bra, ch_ket);
  IMSRG_CHECK(k.has_value());
  const auto blk = matrix_.host_block(*k);
  IMSRG_CHECK(i.value() >= 0 && i.value() < blk.rows && j.value() >= 0 && j.value() < blk.cols);
  blk(i.value(), j.value()) = v;
}

double TwoBody::get_normalized(OrbitIndex a, OrbitIndex b, OrbitIndex c, OrbitIndex d,
                               TwoJ J) const {
  if (!allocated()) return 0.0;
  const auto loc = locate(channels(), a, b, c, d, J);
  if (!loc) return 0.0;
  return loc->phase * get(loc->ch, loc->ch, loc->i, loc->j);
}

double TwoBody::get_unnormalized(OrbitIndex a, OrbitIndex b, OrbitIndex c, OrbitIndex d,
                                 TwoJ J) const {
  const double f = std::sqrt((a == b ? 2.0 : 1.0) * (c == d ? 2.0 : 1.0));
  return f * get_normalized(a, b, c, d, J);
}

void TwoBody::set_normalized(OrbitIndex a, OrbitIndex b, OrbitIndex c, OrbitIndex d, TwoJ J,
                             double v) {
  IMSRG_CHECK(allocated());
  const auto loc = locate(channels(), a, b, c, d, J);
  IMSRG_CHECK(loc.has_value());
  const double stored = loc->phase * v;
  set(loc->ch, loc->ch, loc->i, loc->j, stored);
  const std::int32_t h = hermiticity_sign(hermiticity_);
  if (loc->i != loc->j && h != 0) set(loc->ch, loc->ch, loc->j, loc->i, h * stored);
}

void TwoBody::add_normalized(OrbitIndex a, OrbitIndex b, OrbitIndex c, OrbitIndex d, TwoJ J,
                             double v) {
  IMSRG_CHECK(allocated());
  const auto loc = locate(channels(), a, b, c, d, J);
  IMSRG_CHECK(loc.has_value());
  const double stored = loc->phase * v;
  // The partner receives h * stored. imsrg++ AddToTBME omits the sign h for anti-Hermitian parts;
  // this is fixed here (design/30 section 6).
  set(loc->ch, loc->ch, loc->i, loc->j, get(loc->ch, loc->ch, loc->i, loc->j) + stored);
  const std::int32_t h = hermiticity_sign(hermiticity_);
  if (loc->i != loc->j && h != 0) {
    set(loc->ch, loc->ch, loc->j, loc->i, get(loc->ch, loc->ch, loc->j, loc->i) + h * stored);
  }
}

void TwoBody::set_zero() {
  if (matrix_.total_size() > 0) matrix_.buffer().fill(0.0);
}

linalg::MatrixView<double> TwoBody::host_block(ChannelIndex ch) {
  const auto k = block_index(ch, ch);
  IMSRG_CHECK(k.has_value());
  return matrix_.host_block(*k);
}
linalg::MatrixView<const double> TwoBody::host_block(ChannelIndex ch) const {
  const auto k = block_index(ch, ch);
  IMSRG_CHECK(k.has_value());
  return std::as_const(matrix_).host_block(*k);
}
linalg::MatrixView<double> TwoBody::device_block(ChannelIndex ch) {
  const auto k = block_index(ch, ch);
  IMSRG_CHECK(k.has_value());
  return matrix_.device_block(*k);
}
linalg::MatrixView<const double> TwoBody::device_block(ChannelIndex ch) const {
  const auto k = block_index(ch, ch);
  IMSRG_CHECK(k.has_value());
  return std::as_const(matrix_).device_block(*k);
}

}  // namespace imsrg::op
