// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_DRIVER_CONFIG_COMPAT_H_
#define IMSRG_DRIVER_CONFIG_COMPAT_H_
/// @file
/// Compatibility profiles (`run.compat`, design/50 section 5): one config key
/// switches on a set of named internal options that reproduce the behaviour
/// of another code version. The default profile (`none`) is the fixed
/// behaviour. Each option is a plain bool with its own test against the
/// reference build it emulates; add new options here, with a line in the
/// list below and in design/16 (`run.compat`).
///
/// Profile `imsrg_fork_79e9330` (the fork imsrg++ 79e9330 of the 27Al
/// archive, reference/06 section 4.1):
///   * `nat_sort_by_abs_occupation` (lead C): natural orbitals re-sorted by
///     |occupation| inside each channel (HFMBPT.cc:131-182).
///   * `fork_step_defaults` (lead C): the fork's Parameters.hh default
///     domega = 0.2 (imsrg-devel: 0.1) when imsrg.step.domega is not given.
///   * `tokyo_writer_fork_79e9330` (lead B): the tokyo writers (scalar and
///     tensor) in the fork format (7 significant digits, absolute 1e-7
///     thresholds, core reference in the header).
///   * `tensor_no_fork_diagonal_bug`: tensor normal ordering omits, for
///     a == b, the channel pairs with ch(a h, J_bra) > ch(a h, J_ket)
///     (also in the re-normal-ordering).
///   * `fork_operator_file_names` (lead B, AL-F2): operator files are named
///     `<stem><name>.snt` / `<stem><name>_2b.snt` (no separator, the fork's
///     `<intfile><op>`) instead of `<stem>_<name>.snt`.
///   * `tensor_ninej_threshold_1e8`: 9j skip threshold 1e-8 (not 1e-10) in
///     the tensor commutators of the flow.
///   * `tensor_fork_ph_missing_rank_phase`: the fork's particle-hole tensor
///     commutator omits (-1)^L (devel has it) when it reads Zbar through a
///     swapped pair; affects odd L only (Sigma', Sigma'', Delta').

#include "imsrg/driver/config/schema.h"

namespace imsrg::driver {

/// The internal options a compatibility profile switches on.
struct CompatOptions {
  /// Sort natural orbitals by descending |occupation| (fork 79e9330), instead
  /// of descending signed occupation (imsrg-devel).
  bool nat_sort_by_abs_occupation = false;
  /// Default imsrg.step.domega = 0.2 (fork Parameters.hh:117) instead of 0.1.
  bool fork_step_defaults = false;
  /// Tokyo (.snt) writer in the fork 79e9330 format: 7 significant
  /// digits (scientific, precision 6, width 18), one-body |v| < 1e-7 dropped
  /// (absolute), two-body without threshold, zero body with 6 significant
  /// digits, header Aref/Zref/A_for_kinetic_energy of the core reference (the
  /// fork writes after SetReference(core): 16 8 16 for an sd-shell run).
  /// The tensor writer uses `io::TokyoFormat::kFork79e9330` and the same
  /// header.
  bool tokyo_writer_fork_79e9330 = false;
  /// Fork tensor normal-ordering bug
  /// (`op::TensorNormalOrderOptions::fork79e9330_diagonal_bug`).
  bool tensor_no_fork_diagonal_bug = false;
  /// `commutator::CommutatorOptions::tensor_ninej_threshold = 1e-8`.
  bool tensor_ninej_threshold_1e8 = false;
  /// Fork particle-hole tensor commutator without the (-1)^L of devel for odd
  /// L (`CommutatorOptions::tensor_fork_ph_missing_rank_phase`).
  bool tensor_fork_ph_missing_rank_phase = false;
  /// Operator valence files `<stem><name>[_2b].snt` (fork) instead of
  /// `<stem>_<name>[_2b].snt`; the Hamiltonian file `<stem>.snt` is unchanged.
  bool fork_operator_file_names = false;
  friend bool operator==(const CompatOptions&, const CompatOptions&) = default;
};

/// The options of profile `c`.
constexpr CompatOptions CompatOptionsFor(Compat c) noexcept {
  CompatOptions o;
  switch (c) {
    case Compat::kNone:
      break;
    case Compat::kImsrgFork79e9330:
      o.nat_sort_by_abs_occupation = true;
      o.fork_step_defaults = true;
      o.tokyo_writer_fork_79e9330 = true;
      o.fork_operator_file_names = true;
      o.tensor_no_fork_diagonal_bug = true;
      o.tensor_ninej_threshold_1e8 = true;
      o.tensor_fork_ph_missing_rank_phase = true;
      break;
  }
  return o;
}

}  // namespace imsrg::driver

#endif  // IMSRG_DRIVER_CONFIG_COMPAT_H_
