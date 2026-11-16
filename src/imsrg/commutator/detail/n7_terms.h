// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_COMMUTATOR_DETAIL_N7_TERMS_H_
#define IMSRG_COMMUTATOR_DETAIL_N7_TERMS_H_

/// @file
/// The batch interface of the row-tiled IMSRG(3)-N7 terms (design/14 section
/// 6.3): the inputs of one batch (operand views, tables, the batch's tiles and
/// the H and W intermediates of the three-body operands), the batch loop, the
/// per-term batch kernels with their element-wise references, and the fused
/// driver of all N7 terms (`RunN7`). Internal to the commutator module and its
/// tests; no Kokkos.
///
/// Contract of every batch kernel: it adds the contribution of exactly the
/// lines of its batch (design/14 section 6.3 attribution) to the output, reads
/// three-body operand entries of these lines only (through
/// `ThreeBodyLocal` or the batch's H/W), and writes nothing else. Summed over
/// the batches of any `basis::RowPartition` of the space, the contributions
/// give the whole term (partition invariance).

#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>

#include "imsrg/basis/three_body_rows.h"
#include "imsrg/commutator/detail/comm232.h"
#include "imsrg/commutator/detail/n7_batch.h"
#include "imsrg/commutator/detail/n7_common.h"
#include "imsrg/commutator/detail/n7_views.h"
#include "imsrg/commutator/options.h"
#include "imsrg/commutator/three_body.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/pair_third.h"

namespace imsrg::commutator::detail {

/// Default byte budget of the per-batch intermediates (H and W of all operand
/// blocks together): 512 MiB, as comm232ss before the row tiling (M10-Q74).
/// It bounds the H + W of ONE batch (`PlanBatches`); the allocation of
/// `ForEachN7Batch` is the max over batches of H plus the max of W (at most
/// twice the budget), and the 133 `TransposeOut` (~ sum over lines of (m + 1)
/// doubles) and slot tables (36 bytes per entry, n 3 n1max entries per tile
/// channel) come on top; see `PlanBatches`.
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
  /// HOST pointers for the host part of a batch kernel (per-batch work
  /// lists); never dereferenced in device code.
  const ThreeBodyContext* ctx = nullptr;  ///< The context of the loop.
  const LineBatch* plan = nullptr;        ///< Host copy of the batch.
};
static_assert(std::is_trivially_copyable_v<N7BatchInputs>);

/// Runs `body` once per batch of `rows` (`PlanBatches` with `budget_bytes`),
/// in plan order, after building the requested intermediates of the
/// three-body parts in `v` (device work fenced before `body`). The buffers
/// are allocated once per call (largest batch) and reused, so `body` must
/// finish its device work (fence) before it returns. Operand blocks for the
/// budget: one per present three-body part, two for a dense one. Device memory
/// of the H/W buffers: 8 blocks (h_max + w_max) bytes, with h_max / w_max the
/// largest H / W of any batch (so up to twice `budget_bytes`, see
/// `PlanBatches`; W only if requested), plus what `body` allocates itself
/// (e.g. the 133 TransposeOut and slot tables). Host only.
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

/// Per-call state of the 133 batch loop: the largest one-body channel and the
/// slot table of the channels of the last batch (a table entry depends on its
/// channel only, not on the batch's row range, so consecutive batches of the
/// same channels reuse it). Create one per `Comm133Rows`/`RunN7` call; the
/// operands, the context and the set of present H blocks must stay the same
/// for all batches that share it. Move-only value.
class Comm133Cache {
 public:
  Comm133Cache();
  ~Comm133Cache();
  Comm133Cache(const Comm133Cache&) = delete;
  Comm133Cache& operator=(const Comm133Cache&) = delete;
  Comm133Cache(Comm133Cache&&) noexcept;
  Comm133Cache& operator=(Comm133Cache&&) noexcept;

  struct Impl;  // Defined in kernels/comm133.cc.
  Impl& impl() { return *impl_; }

 private:
  std::unique_ptr<Impl> impl_;
};

/// [X1, Y3] + [X3, Y1] -> Z3: the bra side into the Z3 rows of the batch's
/// lines (`z3`), the ket side into `tout` (needs H). Packed Z only; a dense
/// Z uses `Comm133ssReference`. `cache` is the per-call state (see
/// `Comm133Cache`): the host and device work per batch is proportional to the
/// batch's tiles, except when the list of tile channels differs from the
/// previous batch's (then the slot tables of those channels are rebuilt).
void Comm133Batch(const N7BatchInputs& in, const ThreeBodyOutView& z3,
                  const TransposeOut& tout, Comm133Cache& cache);

/// Adds the ket-side values of `tout` (written by `Comm133Batch` for the
/// batch `in.batch`) into the stored elements z(m, r) of `z3`. Blocking.
void AddTransposeOut(const N7BatchInputs& in, const TransposeOut& tout,
                     const ThreeBodyOutView& z3);

// ---------------------------------------------------------------------------
// Single-term batch loops (defined next to the batch kernels): the term
// restricted to the lines of `rows`, batches planned with `budget_bytes`,
// added into Z; the public `terms::` function of the term is the call with
// `RowSet::Whole` and `kN7BatchBytes`. Same checks and contract as the
// `terms::` functions (`PrepareN7`). For 133 the transpose outputs are
// placed into Z's rows (all rows are local in serial code).

void Comm232Rows(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
                 const ThreeBodyContext& ctx, const basis::RowSet& rows,
                 std::int64_t budget_bytes);
void Comm231Rows(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
                 const ThreeBodyContext& ctx, const basis::RowSet& rows,
                 std::int64_t budget_bytes);
void Comm331Rows(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
                 const ThreeBodyContext& ctx, const basis::RowSet& rows,
                 std::int64_t budget_bytes);
void Comm132Rows(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
                 const ThreeBodyContext& ctx, const basis::RowSet& rows,
                 std::int64_t budget_bytes);
void Comm330Rows(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
                 const ThreeBodyContext& ctx, const basis::RowSet& rows,
                 std::int64_t budget_bytes);
void Comm133Rows(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
                 const ThreeBodyContext& ctx, const basis::RowSet& rows,
                 std::int64_t budget_bytes);

// ---------------------------------------------------------------------------
// The fused batch driver (defined in n7_batches.cc).

/// Adds the N7 terms of `terms` of [X, Y] (any subset of the seven,
/// kComm133ss..kComm132ss; IMSRG(2) terms in the set are ignored, so the
/// mask of a `Commutator` call can be passed as is), restricted to the lines
/// of `rows`, into Z (design/14 section 6.3): the batch driver of
/// `Commutator()`.
///
/// ONE `ForEachN7Batch` whose `N7Needs` is the union of the needs of the
/// selected terms that contribute (H for 133 and 331, W for 232, 231 and
/// 132), so H and W are built once per batch and shared by the terms. Per
/// batch, in this order: 133 (`Comm133Batch`, then `AddTransposeOut` into
/// Z's rows: all rows are local in serial code), 232, 132, 331, 231, 330
/// (the partial Z0 of the batches summed on the host in batch order). After
/// the loop: 223 (`AddComm223Rows` on `rows`, so every Z3 element gets 133
/// before 223, as in imsrg++), then Z0 += the 330 sum. Per output part this
/// keeps imsrg++'s relative order (Z3: 133 then 223; Z2: 232 then 132; Z1:
/// 331 then 231), so the result equals the sequence of the single-term
/// `terms::` calls (imsrg++ order) bitwise if both plan the same single
/// batch, and up to rounding (the interleaving of the per-batch partial
/// sums of different terms, and a different batching) otherwise (M10-Q79).
/// Each kernel sees the inputs its single-term loop would give it (232 and
/// 132 without the three-body part whose two- or one-body partner is
/// missing); the batch plan is that of the full operands.
///
/// A dense (non-Hermitian) Z3 with 133 selected: 133 through
/// `Comm133ssReference` before the loop, which needs `rows` to cover the
/// whole kept space (M12b never distributes dense operands). An operand
/// without a three-body part skips, term by term, what the single-term
/// functions skip (their early returns): 330 and 331 need both three-body
/// parts, 133 and 231 at least one, 232 and 132 a three-body part whose
/// two- (232) or one-body (132) partner in the other operand is present, 223
/// both two-body parts; 330 adds nothing to an anti-Hermitian Z.
///
/// Checks once, before any work, what the single-term functions check
/// (`PrepareN7`; Z's three-body part if 133 or 223 is selected, its two-body
/// part if 232 or 132 is), with the same exceptions; a message names the
/// first selected term (imsrg++ order) that makes the check. An empty
/// selection does nothing (no checks).
/// @throws NotImplementedError for an N8/N9 term in `terms`, and as
///         `PrepareN7`.
/// @throws InputError as `PrepareN7`.
/// @throws InvariantError as `PrepareN7`, `ThreeBodyOut` and `TwoBodyOut`,
///         and for a dense Z3 with 133 selected if `rows` does not cover the
///         whole kept space.
/// Preconditions: `rows` built on the context's three-body space;
/// `budget_bytes` >= 0 (`PlanBatches`). Host only. Cost: the H/W builds once
/// per batch, the selected batch kernels, and the 223 work items of `rows`.
void RunN7(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
           const ThreeBodyContext& ctx, TermMask terms,
           const basis::RowSet& rows, std::int64_t budget_bytes);

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
