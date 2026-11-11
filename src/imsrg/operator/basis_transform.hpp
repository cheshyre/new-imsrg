#pragma once
/// @file
/// Unitary single-particle basis transformations of operators (e.g. HO -> HF basis), design/10
/// section 3.3, imsrg++ `HartreeFock::TransformToHFBasis` (HartreeFock.cc:914-1000).
#include <memory>

#include "imsrg/basis/model_space.hpp"
#include "imsrg/linalg/matrix_view.hpp"
#include "imsrg/operator/operator.hpp"

namespace imsrg::op {

/// The operator in the new single-particle basis |p'> = sum_p C_{p p'} |p>, where `C` is the
/// norb x norb matrix of imsrg++ `HartreeFock::C` (row = old orbit, column = new orbit, host
/// memory). C must be orthogonal and may only mix orbits with equal (l, 2j, tz2), so that every
/// two-body channel is mapped onto itself; this is the caller's responsibility (not checked).
/// * zero-body: unchanged;
/// * one-body: O' = C^T O C;
/// * two-body, per channel J: V' = D^T V D with, for old kets (pq) (rows) and new kets (p'q')
///   (columns), D = C_{pp'} C_{qq'} + [p != q] C_{qp'} C_{pq'} (-1)^(j_p + j_q - J + 1), times
///   sqrt(2) if p == q and divided by sqrt(2) if p' == q' (normalized kets).
/// The result is on the same model space with the same spec (use `with_model_space` to attach
/// other occupations, e.g. after a HF re-filling).
/// @throws InvariantError if C is not norb x norb.
/// Complexity: O(norb^3) for the one-body part plus O(sum_ch n_ch^3) (two gemms per channel).
Operator transform_basis(const Operator& op, linalg::MatrixView<const double> C);

/// The same matrix elements on another model space with the same single-particle basis (only
/// the reference occupations and/or the partition differ), for example to attach the HF
/// occupations to an operator transformed to the HF basis. The data are copied unchanged.
/// @throws InvariantError if `op` is empty, `ms` is null, or the orbits of the two bases differ.
Operator with_model_space(const Operator& op, std::shared_ptr<const basis::ModelSpace> ms);

}  // namespace imsrg::op
