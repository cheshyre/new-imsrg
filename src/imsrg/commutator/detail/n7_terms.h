// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_COMMUTATOR_DETAIL_N7_TERMS_H_
#define IMSRG_COMMUTATOR_DETAIL_N7_TERMS_H_

/// @file
/// The batch interface of the row-tiled IMSRG(3)-N7 terms (design/14 section
/// 6.3): the inputs of one batch (operand views, tables, the batch's tiles and
/// the H and W intermediates of the three-body operands), the batch loop, and
/// the per-term batch kernels with their element-wise references. Internal to
/// the commutator module and its tests; no Kokkos.
///
/// Contract of every batch kernel: it adds the contribution of exactly the
/// lines of its batch (design/14 section 6.3 attribution) to the output, reads
/// three-body operand entries of these lines only (through
/// `ThreeBodyLocal` or the batch's H/W), and writes nothing else. Summed over
/// the batches of any `basis::RowPartition` of the space, the contributions
/// give the whole term (partition invariance).

#include <cstdint>
#include <functional>
#include <type_traits>

#include "imsrg/basis/three_body_rows.h"
#include "imsrg/commutator/detail/comm232.h"
#include "imsrg/commutator/detail/n7_batch.h"
#include "imsrg/commutator/detail/n7_common.h"
#include "imsrg/commutator/detail/n7_views.h"
#include "imsrg/commutator/three_body.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/pair_third.h"

namespace imsrg::commutator::detail {

/// Default byte budget of the per-batch intermediates (H and W of all operand
/// blocks together): 512 MiB, as comm232ss before the row tiling (M10-Q74).
inline constexpr std::int64_t kN7BatchBytes = std::int64_t{1} << 29;

/// Which intermediates the batch loop builds for the three-body operands.
struct N7Needs {
  bool h = false;  ///< H = E B on the batch lines (331, 133).
  bool w = false;  ///< W = E B E^T for the batch's states (232, 231, 132);
                   ///< implies H.
};
static_assert(std::is_trivially_copyable_v<N7Needs>);

/// Everything a batch kernel reads (raw device pointers, captured by value).
///
/// H and W blocks (`detail/n7_batch.h`: rows in the sorted order of
/// `Comm232Tables`, columns per tile, `LineTile::h_offset`/`w_offset`) of X3
/// and Y3: `hx`/`wx` are those of X3; `hxt`/`wxt` those of X3^T, a separate
/// block only for a dense (non-Hermitian) X3, otherwise the block of X3
/// itself with `sxt` = h_X3 (B^T = h_B B for a packed part), so H(X3^T) =
/// `sxt` * hxt and W(X3^T) = `sxt` * wxt in every case. Likewise for Y3. A
/// pointer is null if the part is missing or the intermediate was not
/// requested (`N7Needs`).
struct N7BatchInputs {
  N7Views v;                    ///< Operands and canonical-space tables.
  op::PairThirdView pt;         ///< Pair-third states, R and E.
  Comm232TablesView tab;        ///< Sorted pair-third tables.
  LineBatchView batch;          ///< The batch's tiles.
  const double* hx = nullptr;   ///< H(X3).
  const double* hxt = nullptr;  ///< H(X3^T) up to `sxt`.
  const double* hy = nullptr;   ///< H(Y3).
  const double* hyt = nullptr;  ///< H(Y3^T) up to `syt`.
  const double* wx = nullptr;   ///< W(X3).
  const double* wxt = nullptr;  ///< W(X3^T) up to `sxt`.
  const double* wy = nullptr;   ///< W(Y3).
  const double* wyt = nullptr;  ///< W(Y3^T) up to `syt`.
  double sxt = 1.0;             ///< Sign of the X3^T blocks.
  double syt = 1.0;             ///< Sign of the Y3^T blocks.
};
static_assert(std::is_trivially_copyable_v<N7BatchInputs>);

/// Runs `body` once per batch of `rows` (`PlanBatches` with `budget_bytes`),
/// in plan order, after building the requested intermediates of the
/// three-body parts in `v` (device work fenced before `body`). The buffers
/// are allocated once per call (largest batch) and reused, so `body` must
/// finish its device work (fence) before it returns. Operand blocks for the
/// budget: one per present three-body part, two for a dense one. Host only.
/// Preconditions: `v` from `PrepareN7` (or with masked operand data of the
/// same layout) for the context `ctx`; `rows` built on the context's
/// three-body space. Cost: the H/W builds (`BuildH`, `BuildW`) plus `body`.
void ForEachN7Batch(const N7Views& v, const ThreeBodyContext& ctx,
                    const basis::RowSet& rows, std::int64_t budget_bytes,
                    N7Needs needs,
                    const std::function<void(const N7BatchInputs&)>& body);

/// The per-batch ket-side output of 133 (design/14 section 6.3): for batch
/// line l (the batch's lines numbered in tile order, rows ascending within a
/// tile) with channel-local row m, the values D(r, m), r = 0..m, at
/// `data[line_off[l] + r]`; they belong to the stored elements z(m, r) of
/// rows r (r = m is skipped for an anti-Hermitian Z) and are added there by
/// `AddTransposeOut`. Device pointers; trivially copyable.
struct TransposeOut {
  double* data = nullptr;                  ///< Sum over lines of (m + 1).
  const std::int64_t* line_off = nullptr;  ///< Per batch line.
};
static_assert(std::is_trivially_copyable_v<TransposeOut>);

// ---------------------------------------------------------------------------
// Per-term batch kernels (defined in kernels/comm<term>.cc). Each adds the
// contribution of the batch's lines; blocking (fenced) unless stated.

/// [X2, Y3] + [X3, Y2] -> Z2 for the W columns of the batch (needs W).
void Comm232Batch(const N7BatchInputs& in, const TwoBodyOutView& z2);

/// [X2, Y3] + [X3, Y2] -> Z1 for the W columns of the batch (needs W).
/// `z1`: Z's one-body data (norb x norb, column-major, device).
void Comm231Batch(const N7BatchInputs& in, double* z1);

/// [X3, Y3] -> Z1 for the lines of the batch (needs H).
void Comm331Batch(const N7BatchInputs& in, double* z1);

/// [X1, Y3] + [X3, Y1] -> Z2 for the W columns of the batch (needs W).
void Comm132Batch(const N7BatchInputs& in, const TwoBodyOutView& z2);

/// [X3, Y3] -> Z0: the sum over the lines of the batch (needs nothing).
double Comm330Batch(const N7BatchInputs& in);

/// [X1, Y3] + [X3, Y1] -> Z3: the bra side into the Z3 rows of the batch's
/// lines (`z3`), the ket side into `tout` (needs H). Packed Z only; a dense
/// Z uses `Comm133ssReference`.
void Comm133Batch(const N7BatchInputs& in, const ThreeBodyOutView& z3,
                  const TransposeOut& tout);

/// Adds the ket-side values of `tout` (written by `Comm133Batch` for the
/// batch `in.batch`) into the stored elements z(m, r) of `z3`. Blocking.
void AddTransposeOut(const N7BatchInputs& in, const TransposeOut& tout,
                     const ThreeBodyOutView& z3);

// ---------------------------------------------------------------------------
// Element-wise references (the kernels before the row tiling, kept for tests
// and as the dense-Z path of 133; same contract as the `terms::` functions).

void Comm133ssReference(const op::Operator& X, const op::Operator& Y,
                        op::Operator& Z, const ThreeBodyContext& ctx);
void Comm330ssReference(const op::Operator& X, const op::Operator& Y,
                        op::Operator& Z, const ThreeBodyContext& ctx);
void Comm331ssReference(const op::Operator& X, const op::Operator& Y,
                        op::Operator& Z, const ThreeBodyContext& ctx);
void Comm231ssReference(const op::Operator& X, const op::Operator& Y,
                        op::Operator& Z, const ThreeBodyContext& ctx);
void Comm132ssReference(const op::Operator& X, const op::Operator& Y,
                        op::Operator& Z, const ThreeBodyContext& ctx);

}  // namespace imsrg::commutator::detail

#endif  // IMSRG_COMMUTATOR_DETAIL_N7_TERMS_H_
