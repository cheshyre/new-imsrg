// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef TESTS_UNIT_COMMUTATOR_N7_TEST_SUPPORT_H_
#define TESTS_UNIT_COMMUTATOR_N7_TEST_SUPPORT_H_

/// @file
/// Shared fixtures of the IMSRG(3)-N7 term tests: the golden inputs of
/// tests/golden/n7 (regenerated in process from their seeds), golden term
/// outputs, and the m-scheme brute-force reference of a term
/// (`commutator_ref::mscheme`, the D34 oracle).

#include <memory>
#include <string_view>

#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/commutator/options.h"
#include "imsrg/commutator/three_body.h"
#include "imsrg/operator/operator.h"

namespace imsrg::testing {

/// Operands of an N7 test: the model space with its three-body space, the
/// context, and X, Y.
struct N7Inputs {
  std::shared_ptr<const basis::ModelSpace> ms3;
  std::unique_ptr<commutator::ThreeBodyContext> ctx;
  op::Operator x;
  op::Operator y;
};

/// The inputs of the goldens tests/golden/n7/<reference>_emax<emax>: on
/// `MakeThreeBodySpace(emax, reference, {e3max})`, X =
/// RandomThreeBodyOperator(ctx, anti-Hermitian, 11), Y = (Hermitian, 12).
N7Inputs GoldenN7Inputs(int emax, std::string_view reference, int e3max);

/// Inputs on an arbitrary model space with a three-body space: X =
/// RandomThreeBodyOperator(ctx, hx, seed), Y = (hy, seed + 1).
N7Inputs RandomN7Inputs(std::shared_ptr<const basis::ModelSpace> ms3,
                        Hermiticity hx, Hermiticity hy, std::uint64_t seed);

/// A zero Z of rank 3 with the hermiticity of [X, Y] on `ms3`.
op::Operator ZeroN7Output(const op::Operator& X, const op::Operator& Y,
                          std::shared_ptr<const basis::ModelSpace> ms3);

/// Z = term(X, Y) alone, through the public term function of imsrg3.h on a
/// fresh `ZeroN7Output`. Precondition: `t` is one of the seven N7 terms.
op::Operator EvaluateN7Term(commutator::Term t, const op::Operator& X,
                            const op::Operator& Y,
                            const commutator::ThreeBodyContext& ctx);

/// The m-scheme brute force of the N7 term `t` (the parts [X^a, Y^b]^c of the
/// term, e.g. [X2, Y3]^1 + [X3, Y2]^1 for comm231ss), contracted to a rank-3
/// J-scheme operator on `ms3` with the hermiticity of [X, Y]. Cost: seconds
/// at emax 1; use emax 1 only.
op::Operator BruteForceN7Term(commutator::Term t, const op::Operator& X,
                              const op::Operator& Y,
                              std::shared_ptr<const basis::ModelSpace> ms3);

/// The golden output tests/golden/n7/<case_dir>/<file> read on `ms3`.
op::Operator ReadN7Golden(std::string_view case_dir, std::string_view file,
                          std::shared_ptr<const basis::ModelSpace> ms3);

/// A value of tests/golden/n7/<case_dir>/values.json (parsed from its %.17g
/// string): the field `field` ("zero_body", "one_body_norm", "two_body_norm",
/// "three_body_norm") of `name` (a term name such as "comm330ss", "n7",
/// "generator", or an input "X", "Y", "H") in the case with E3max `e3max`.
/// Fails the test (ADD_FAILURE) and returns NaN if the entry is missing.
double N7GoldenValue(std::string_view case_dir, int e3max,
                     std::string_view name, std::string_view field);

}  // namespace imsrg::testing

#endif  // TESTS_UNIT_COMMUTATOR_N7_TEST_SUPPORT_H_
