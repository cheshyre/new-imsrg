// Reader for the Darmstadt (me2j) two-body matrix element format.
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>

#include "imsrg/basis/model_space.hpp"
#include "imsrg/operator/operator.hpp"

namespace imsrg::io {

/// Truncation of a me2j file (imsrg++ `file2e1max`, `file2e2max`, `file2lmax`). The file contains
/// no header information about its own truncation, so it must be supplied to interpret the token
/// stream. A non-positive `e2max` or `lmax` selects the imsrg++ defaults (`2 * e1max` and
/// `e1max`; imsrg++ tests `> 0`, not `>= 0`).
struct Me2jTruncation {
  std::int32_t e1max = 0;   ///< Maximum single-particle oscillator shell e = 2n + l in the file.
  std::int32_t e2max = -1;  ///< Maximum e_a + e_b (and e_c + e_d) of a pair; -1: 2 * e1max.
  std::int32_t lmax = -1;   ///< Maximum single-particle l in the file; -1: e1max.
};

/// The number of floating-point values (4 per (abcd, J) record: T=0 pn, nn, T=1 pn, pp) that a
/// me2j file of truncation `file` holds when only the first `emax_read` shells of the first
/// orbit of the bra are read (the prefix of the file that a model space with `emax = emax_read`
/// needs). `emax_read < 0` or `> file.e1max` counts the whole file. The pair-energy cut always
/// uses the file's own `e2max`, as in imsrg++. Pure arithmetic on the loop structure of the
/// format; O(number of records).
/// Reference value: `{14, 28, 14}` gives 168066656.
std::int64_t me2j_value_count(const Me2jTruncation& file, std::int32_t emax_read = -1);

/// Options for `read_me2j`.
struct Me2jReadOptions {
  Me2jTruncation file;           ///< Truncation of the file being read.
  bool read_whole_file = false;  ///< Consume the entire file and verify its value count.
};

/// What `read_me2j` did.
struct Me2jReadReport {
  std::int64_t values_read = 0;  ///< Number of floats consumed (prefix, or whole file).
  double seconds = 0.0;          ///< Wall time of the read, including file open.
};

/// Reads a Darmstadt me2j file (plain or gzip, detected by content) into a new Hermitian scalar
/// two-body operator on `ms` (`particle_rank` 2, zero- and one-body parts zero).
///
/// Exactly imsrg++ `ReadBareTBME_Darmstadt` plus `Set_pn_TBME_from_iso`: the first line is
/// skipped; then 4 floats per (abcd, J) record in the order T=0 pn, nn, T=1 pn, pp, with orbits in
/// *file order* (e ascending, then l, then 2j; not the model-space order) are parsed as `float`
/// and widened to double. File values are unnormalized, antisymmetrized JT-coupled matrix
/// elements in MeV; they are stored in this library's normalized antisymmetrized convention
/// (multiplied by 1/sqrt(2) for each identical pair, with the imsrg++ Pauli selection). pp and nn
/// elements are set directly and the isospin elements are converted to pn matrix elements
/// V_pnpn = (V_T1 + V_T0)/2 and V_pnnp = (V_T1 - V_T0)/2 (with the 2T-1 phase). Proton and neutron
/// orbits are looked up by (n, l, 2j, tz2) in the model-space basis. Records whose orbits are not
/// all in `ms` are consumed and skipped.
///
/// By default only the prefix of the file needed by `min(file.e1max, ms->basis().emax())` shells
/// is read (the file is ordered by the first bra orbit), which is a small fraction of a big file.
/// With `read_whole_file` the whole file is consumed and its value count checked.
/// Complexity: O(values read) with ~1 ns/byte parsing.
/// @throws InputError if the file cannot be opened; if the data ends before the required values
///         are complete; if a token is not a number; or, with `read_whole_file`, if the number of
///         values differs from `me2j_value_count(options.file)` (the message states expected and
///         found counts and hints at file2e1max, file2e2max and file2lmax).
op::Operator read_me2j(const std::filesystem::path& path, const Me2jReadOptions& options,
                       std::shared_ptr<const basis::ModelSpace> ms,
                       Me2jReadReport* report = nullptr);

}  // namespace imsrg::io
