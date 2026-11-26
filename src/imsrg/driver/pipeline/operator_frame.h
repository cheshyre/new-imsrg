// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_DRIVER_PIPELINE_OPERATOR_FRAME_H_
#define IMSRG_DRIVER_PIPELINE_OPERATOR_FRAME_H_
/// @file
/// The per-operator pipeline of `driver::Run` as reusable functions
/// (design/50 section 2, design/reference/06 section 1.8-1.10): an operator
/// is built in the bare oscillator space, transformed to the HF/NAT basis,
/// normal ordered w.r.t. the reference, truncated to the IMSRG space
/// (`PrepareOperator`), transformed with the flow's Omegas by the caller, and
/// finished (`FinishOperator`: two-body part, re-normal ordering to the core).
/// Scalars and tensor operators (rank L > 0, NREFT) take the same path. The
/// `OperatorFrame` holds everything these functions need, so a caller without
/// a `PreparedHamiltonian` (C and occupations read from files, the transform
/// task) can build one by hand and gets bit-identical results.
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "imsrg/basis/model_space.h"
#include "imsrg/driver/config/schema.h"
#include "imsrg/driver/operators/operator_request.h"
#include "imsrg/driver/pipeline/hamiltonian.h"
#include "imsrg/io/valence_writers.h"
#include "imsrg/linalg/matrix_view.h"
#include "imsrg/operator/normal_ordering.h"
#include "imsrg/operator/operator.h"

namespace imsrg::driver::pipeline {

/// Spaces, basis change and options an operator passes through.
struct OperatorFrame {
  /// Bare (oscillator) model space the operators are built on.
  std::shared_ptr<const basis::ModelSpace> ms_bare;
  /// Coefficients C(ho orbit, HF/NAT orbit) of the basis change (as
  /// `hf::HFResult::C`); empty for the oscillator basis.
  std::optional<linalg::Matrix<double>> c;
  /// Model space (basis.emax, occupations of the reference) in which the
  /// transformed operator is normal ordered.
  std::shared_ptr<const basis::ModelSpace> ms_normal_ordering;
  /// The IMSRG model space (the one of H); the normal-ordered operator is
  /// truncated to it if it differs from `ms_normal_ordering`.
  std::shared_ptr<const basis::ModelSpace> ms_imsrg;
  /// Same basis as `ms_imsrg`, reference = core occupations; null if no
  /// re-normal-ordering is needed (single reference, or no valence space).
  std::shared_ptr<const basis::ModelSpace> ms_core;
  /// Mass number of the kinetic energy and the point-nucleon operators.
  int mass_number = 0;
  /// Proton number of the target nucleus (Rp2 and friends).
  int proton_number = 0;
  /// Options of the tensor normal ordering (compat: fork diagonal bug).
  op::TensorNormalOrderOptions tensor_no;
};

/// Occupations of the core of `ms` (1 for core orbits, 0 otherwise).
std::vector<double> CoreOccupations(const basis::ModelSpace& ms);

/// imsrg++.cc:1221-1277: true if, with a valence space, the reference used for
/// normal ordering is not the core (ensemble / targeted normal ordering), so
/// the final operators are re-normal ordered w.r.t. the core.
bool NeedsRenormalOrdering(const basis::ModelSpace& ms);

/// The frame of a prepared Hamiltonian: C from `ph.nat` (else `ph.hf`),
/// `ms_normal_ordering` as in `PrepareHamiltonian`, `ms_imsrg` the model space
/// of `ph.hno`, `ms_core` set iff `NeedsRenormalOrdering`, and the compat
/// options of `config.run.compat`.
OperatorFrame MakeOperatorFrame(const PreparedHamiltonian& ph,
                                const RunConfig& config);

/// The bare operator of `request` on `frame.ms_bare` (oscillator basis,
/// vacuum normal ordered). NREFT shorthands give scalars for L = 0 and
/// Hermitian tensor operators for L > 0.
/// @throws ConfigError ("E-PLANNED ...") for kinds that are not implemented.
op::Operator BuildBareOperator(const OperatorRequest& request,
                               const OperatorFrame& frame);

/// `BuildBareOperator` then `PrepareBareOperator`.
op::Operator PrepareOperator(const OperatorRequest& request,
                             const OperatorFrame& frame);

/// A bare operator (on `frame.ms_bare`) -> basis change with `frame.c` ->
/// normal ordering w.r.t. the reference of `frame.ms_normal_ordering`
/// (`op::NormalOrder`, tensors `op::NormalOrderTensor(frame.tensor_no)`) ->
/// `op::Truncate` to `frame.ms_imsrg` if that differs (imsrg++.cc:1386-1475).
op::Operator PrepareBareOperator(const op::Operator& bare,
                                 const OperatorFrame& frame);

/// The transformed operator after the flow: the zero- to two-body part, then
/// `op::ReNormalOrder` to `frame.ms_core` (w.r.t. the core) when set, with
/// `frame.tensor_no`. The result lives on `frame.ms_core` in that case.
op::Operator FinishOperator(const op::Operator& transformed,
                            const OperatorFrame& frame);

/// Header and format options of the valence-space writers. With
/// `compat.tokyo_writer_fork_79e9330` the fork's number format and its header values
/// (Aref/Zref and A_for_kinetic_energy are those of the core, reference/06
/// section 1.10).
io::ValenceWriteOptions MakeValenceWriteOptions(const RunConfig& config,
                                                const OperatorFrame& frame,
                                                io::ValenceWriteMode mode);

/// File name of an operator's valence file: `<stem>_<name>.snt` for scalars,
/// `<stem>_<name>_2b.snt` for tensors (the fork's naming).
std::string OperatorFileName(const std::string& stem, const std::string& name,
                             const op::OperatorSpec& spec);

/// Writes a TENSOR operator in the (reduced matrix element) tokyo format of
/// `io::WriteTensorTokyo`; format and header per `MakeValenceWriteOptions`.
/// Syncs `O` to the host.
void WriteTensorSnt(const std::filesystem::path& path, op::Operator O,
                    const RunConfig& config, const OperatorFrame& frame);

}  // namespace imsrg::driver::pipeline

#endif  // IMSRG_DRIVER_PIPELINE_OPERATOR_FRAME_H_
