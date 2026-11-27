// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_DRIVER_PIPELINE_PREPARED_STATE_H_
#define IMSRG_DRIVER_PIPELINE_PREPARED_STATE_H_
/// @file
/// The prepared Hamiltonian in a checkpoint directory (design/00 D75,
/// design/13 section 7.3): `<checkpoint dir>/prepared/` holds what
/// `driver::Run` needs of a `PreparedHamiltonian` after stage 1 (readers, HF,
/// NAT), so a restart (`checkpoint.restart`) or another flow with the same
/// Hamiltonian (`checkpoint.prepared_from`) skips the ~16 min of preparation
/// of a production run.
///
/// Files (all native float64, independent of `checkpoint.format`):
///   * `hno.opb`: `PreparedHamiltonian::hno` (normal ordered, IMSRG model
///     space, Lawson constant subtracted), `io::WriteOperatorBinary`;
///   * `c_hf.bin`, `c_nat.bin`: C of `hf` / `nat` (u64 rows, u64 cols,
///     column-major f64, as `<stem>_basis.bin` of `output.write_omega`);
///   * `prepared.toml`, written last: version, `preparation_hash`, mass and
///     proton number, e_reference, e_mp2 (hexfloat strings), the occupations
///     and reference names of hno's model space and of `ms_normal_ordering`,
///     the HF/NAT scalars (e_hf, e1, e2, e3, iterations, converged) and
///     occupations, the natural occupations.
/// The directory is written as `prepared.tmp/` and renamed, so `prepared/`
/// is complete whenever it exists.
///
/// Restored: every field of `PreparedHamiltonian` that later stages read
/// (`ms_bare` is rebuilt from the configuration). NOT restored (left empty):
/// `HFResult::spe`, `rho`, `F` and `ediff_history` of `hf` and `nat`.
#include <cstdint>
#include <filesystem>

#include "imsrg/driver/config/schema.h"
#include "imsrg/driver/pipeline/hamiltonian.h"

namespace imsrg::driver::pipeline {

/// FNV-1a hash of what determines the prepared Hamiltonian H_0
/// (`ConfigTablesHash`): the tables basis, reference, valence, hamiltonian,
/// interaction.two_body, interaction.three_body, hartree_fock,
/// natural_orbitals, imsrg.space, imsrg3 and the key run.compat. Everything
/// else (step, omega, generator, operators, output, checkpoint, transform,
/// tables added later) does not enter (WE-A-4).
std::uint64_t PreparationHash(const RunConfig& config);

/// `<checkpoint_dir>/prepared`.
std::filesystem::path PreparedStateDirectory(
    const std::filesystem::path& checkpoint_dir);

/// True if `<checkpoint_dir>/prepared/prepared.toml` exists.
bool HasPreparedState(const std::filesystem::path& checkpoint_dir);

/// Writes `ph` into `<checkpoint_dir>/prepared/` (replacing an earlier one),
/// with `PreparationHash(config)`. Call on one rank only. Precondition:
/// `ph` was prepared for `config`.
/// @throws InputError / std::filesystem::filesystem_error if writing fails.
void WritePreparedState(const std::filesystem::path& checkpoint_dir,
                        const PreparedHamiltonian& ph, const RunConfig& config);

/// Reads the prepared Hamiltonian of `<checkpoint_dir>/prepared/` for
/// `config`: bit-identical `hno`, C matrices, scalars and occupations as
/// written; the model spaces are rebuilt from the configuration and the
/// stored occupations exactly as `PrepareHamiltonian` builds them.
/// @throws InputError if the directory is missing or incomplete, or was
///   written for another Hamiltonian (`PreparationHash` differs).
PreparedHamiltonian ReadPreparedState(
    const std::filesystem::path& checkpoint_dir, const RunConfig& config);

}  // namespace imsrg::driver::pipeline

#endif  // IMSRG_DRIVER_PIPELINE_PREPARED_STATE_H_
