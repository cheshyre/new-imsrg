// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Writers of valence-space interaction files: KSHELL/tokyo (.snt) and NuShellX
// (.int, .sp).
#ifndef IMSRG_IO_VALENCE_WRITERS_H_
#define IMSRG_IO_VALENCE_WRITERS_H_

#include <cstdint>
#include <filesystem>
#include <string>

#include "imsrg/operator/operator.h"

namespace imsrg::io {

/// How the two-body matrix elements are written (imsrg++ `mode` argument of
/// `WriteTokyo` and `WriteNuShellX_intfile`).
enum class ValenceWriteMode {
  /// "" / "int": the matrix elements as stored (a Hamiltonian).
  kHamiltonian,
  /// "op": for proton-neutron orbit pairs, <ab|V|cd> is replaced by the average
  /// of itself and the matrix element with every orbit replaced by its isospin
  /// partner (a - tz2 in imsrg++ indexing, i.e. proton <-> neutron). NuShellX
  /// requires Vpnpn = Vnpnp for operators other than the Hamiltonian, which
  /// only holds approximately for IMSRG-evolved operators.
  kIsospinAveragedOperator
};

/// Number formatting and thresholds of `WriteTokyo`.
enum class TokyoNumberFormat {
  /// imsrg-devel: fixed, 16 decimals, relative thresholds, zero body with 9
  /// digits.
  kDevel,
  /// The owner's imsrg++ fork at 79e9330 (compat "imsrg_fork_79e9330"), byte
  /// for byte: one-body rows kept iff |v| >= 1e-7 (absolute); no two-body
  /// threshold (zeros are written); integers width 4, values scientific with
  /// 6 digits, width 18, no extra column spacing; zero body in the default
  /// stream format (6 significant digits); the stream-state quirk that the
  /// two-body count line prints hw as scientific (e.g. 1.600000e+01) iff at
  /// least one one-body row was written.
  kFork79e9330
};

/// Everything the header comments of the files record that the operator itself
/// does not know (imsrg++ takes these from `ReadWrite` members and the model
/// space). Only used for the comment lines; they never influence the numbers.
struct ValenceWriteOptions {
  ValenceWriteMode mode =
      ValenceWriteMode::kHamiltonian;  ///< Two-body averaging mode.
  std::string file_2n;      ///< Path of the 2N input file; only the last path
                            ///< component is written.
  std::string file_3n;      ///< Path of the 3N input file; only the last path
                            ///< component is written.
  std::int32_t e1max = 0;   ///< Single-particle truncation of the calculation.
  std::int32_t e2max = 0;   ///< Two-body truncation of the calculation.
  std::int32_t e3max = 0;   ///< Three-body truncation of the calculation.
  double hbar_omega = 0.0;  ///< Oscillator frequency in MeV; also written to
                            ///< the interaction line of the .snt file.
  std::int32_t a_ref =
      0;  ///< Mass number of the reference state (comment only).
  std::int32_t z_ref =
      0;  ///< Proton number of the reference state (comment only).
  double a_target =
      0.0;  ///< Target mass A used for the kinetic energy (comment only).
  TokyoNumberFormat tokyo_format =
      TokyoNumberFormat::kDevel;  ///< `WriteTokyo` only.
  std::string
      version;  ///< Code version string written to the first comment line.
};

/// Options of `WriteTokyo` (same struct as the NuShellX writers).
using TokyoWriteOptions = ValenceWriteOptions;
/// Options of `WriteNushellxInt` and `WriteNushellxSps`.
using NushellxWriteOptions = ValenceWriteOptions;

/// Writes the valence-space interaction of `op` in the KSHELL/tokyo `.snt`
/// format (imsrg++ `ReadWrite::WriteTokyo`, ReadWrite.cc:5841).
///
/// The valence space is the valence part of the model space's partition.
/// Valence orbits are numbered 1, 2, ... protons first, then neutrons, each in
/// ascending orbit index. Layout: five comment lines (version, 2N and 3N files,
/// truncations / hw / Aref / Zref / A, the zero-body term with 9 significant
/// digits), the model-space block (counts and core Z, N; one line n l 2j 2tz
/// per valence orbit), the one-body block (count, 0, hw; then rows a b value
/// for a >= b in orbit order) and the two-body block (count, 0, hw; then rows a
/// b c d J value) with the matrix elements <ab;J|V|cd;J> of the *normalized*
/// (unsymmetrized) storage, for channels in channel order and, within a
/// channel, valence-valence bras and kets with ket >= bra. Thresholds are
/// relative: a one-body element is dropped if |v| < 1e-7 ||V1|| or ||V1|| <
/// 1e-8, a two-body element if |v| < 1e-7 ||V2|| or ||V2|| < 1e-7 (norms as
/// `op::OneBodyNorm`, `op::TwoBodyNorm`). Numbers are fixed with 16
/// decimals, width 16. The stream state of imsrg++ is reproduced, including its
/// quirk that hbar_omega is printed with 9 significant digits in the one-body
/// count line and with 16 decimals in the two-body count line.
///
/// `op` must have the host mirrors of the one- and two-body parts up to date.
/// Cost: O(number of valence two-body matrix elements).
/// @throws RuntimeError if the file cannot be written.
void WriteTokyo(const std::filesystem::path& path, const op::Operator& op,
                const TokyoWriteOptions& options);

/// Writes the valence-space interaction of `op` in the NuShellX `.int` format
/// (imsrg++ `ReadWrite::WriteNuShellX_intfile`, ReadWrite.cc:3610; mode "int"
/// for `Hamiltonian`, "op" for `IsospinAveragedOperator`).
///
/// Orbit numbering as `WriteTokyo`. Layout: comment lines (version, input
/// files, truncations, zero-body term, orbit index table), then the single line
/// `-999`, the diagonal one-body elements, A_core, A_core + 2 and 0.00000 (no
/// mass dependence), then one row `a b c d J T value` per two-body element with
/// |v| >= 1e-6 (absolute threshold), in the order of `WriteTokyo`. T = |Tz| of
/// the channel; for Tz = 0 the element is multiplied by sqrt(2) for each pair
/// of distinct orbits (pn matrix elements are unnormalized in NuShellX), T
/// becomes (J + 1) mod 2, indices are swapped (with the exchange phase) so that
/// a <= b and c <= d, and the pairs are swapped so that (a, b) <= (c, d).
/// Numbers: fixed, 6 decimals, width 12; the zero-body term and the diagonal
/// one-body elements use the default stream precision (6 significant digits).
///
/// `op` must have the host mirrors up to date. Cost: O(number of valence
/// two-body matrix elements).
/// @throws RuntimeError if the file cannot be written.
void WriteNushellxInt(const std::filesystem::path& path, const op::Operator& op,
                      const NushellxWriteOptions& options);

/// Writes the valence model space in the NuShellX `.sp` format (imsrg++
/// `ReadWrite::WriteNuShellX_sps`, ReadWrite.cc:3753): a comment line, "pn",
/// A_core and Z_core, the number of valence orbits, "2 <number of proton
/// orbits> <number of neutron orbits>", one line `index n+1 l 2j` per orbit
/// (radial quantum number counted from 1) and an empty line. Only the
/// operator's model space is used (`options` is accepted for symmetry and
/// ignored).
/// @throws RuntimeError if the file cannot be written.
void WriteNushellxSps(const std::filesystem::path& path, const op::Operator& op,
                      const NushellxWriteOptions& options = {});

}  // namespace imsrg::io

#endif  // IMSRG_IO_VALENCE_WRITERS_H_
