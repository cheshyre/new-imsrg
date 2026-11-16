// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_DRIVER_RUN_H_
#define IMSRG_DRIVER_RUN_H_
/// @file
/// The pipeline entry point of the `imsrg` executable and the Python bindings.

#include "imsrg/driver/config/schema.h"

namespace imsrg::driver {

/// Runs the calculation described by the resolved `config` (design/13 section
/// 9, design/16 section 6.1). First writes
/// `<output.directory>/resolved_config.toml` (directory created if missing), so
/// that a run that fails later still leaves it (design/16 section 1.7); then:
/// Hamiltonian preparation (model space, interaction, HF, normal ordering), the
/// IMSRG(2) Magnus flow (stage 1 with `imsrg.generator.core`; with a valence
/// space and `imsrg.steps = 2` stage 2 with `imsrg.generator.valence`, a new
/// Omega, integrating [s1, s1 + smax], D47), re-normal ordering w.r.t. the core
/// for ensemble references, the transformation of `operators.list`, and the
/// outputs in `output.directory`: `results.json` (E_ref, HF parts, per-stage
/// step counts and final values, E0, operator values), the flow CSV at
/// `output.flow_file` (one %.17g line per step: stage, istep, s, ds, E0, norms
/// of H, eta and the current Omega, Omega count, wall time), the valence-space
/// interaction `<stem>.snt` (tokyo; `.int`/`.sp` for nushellx) and one
/// `<stem>_<operator>.snt` per operator. Needs an initialized exec::Runtime.
/// @throws ConfigError ("E-PLANNED ...") for options not implemented yet
///         (checked before any work: methods other than magnus, IMSRG(3), NAT
///         basis, ...); InputError for unreadable inputs or unwritable outputs;
///         NumericalError if the flow diverged (after writing results.json).
void Run(const RunConfig& config);

}  // namespace imsrg::driver

#endif  // IMSRG_DRIVER_RUN_H_
