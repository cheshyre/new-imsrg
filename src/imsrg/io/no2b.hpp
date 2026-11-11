// Reader for the NO2B (normal-ordered two-body approximation) three-nucleon matrix-element files.
#pragma once

#include <cstdint>
#include <filesystem>

#include "imsrg/operator/no2b.hpp"

namespace imsrg::io {

/// Value type of a binary NO2B file.
enum class No2bPrecision {
  Single,  ///< IEEE binary32 (`float`), 4 bytes per value.
  Half,    ///< IEEE binary16, 2 bytes per value (converted exactly to float).
};

/// Truncation of a NO2B file (imsrg++ `file3e1max`, `file3e2max`, `file3e3max`, `file3lmax`). The
/// file has no header describing its truncation, so it must be supplied to interpret the stream.
struct No2bFileTruncation {
  std::int32_t e1max = 0;  ///< Largest single-particle shell e = 2n + l in the file.
  std::int32_t e2max = 0;  ///< Largest pair energy e_a + e_b of any pair of the three orbits.
  std::int32_t e3max = 0;  ///< Largest three-body energy e_a + e_b + e_c.
  std::int32_t lmax = -1;  ///< Largest single-particle l in the file; -1 selects `e1max`.
};

/// Options for `read_no2b`.
struct No2bReadOptions {
  No2bFileTruncation file;                          ///< Truncation of the file being read.
  No2bPrecision precision = No2bPrecision::Single;  ///< Value type; used for binary files only.
};

/// What `read_no2b` did.
struct No2bReadReport {
  std::int64_t values_read = 0;    ///< Values consumed from the file (stored or skipped).
  std::int64_t values_stored = 0;  ///< Values stored in the container (|v| > 1e-8, in range).
  double seconds = 0.0;            ///< Wall time of the read, including file open.
};

/// The number of values in a whole NO2B file of truncation `file`: for every file-orbit sextuple
/// (a >= b, d <= a, e <= d; c, f with j_c = j_f, l_c = l_f; energy cuts; parity) and every J from
/// max(|ja-jb|, |jd-je|)/2 to min(ja+jb, jd+je)/2, five isospin combinations
/// ((T12, T45, 2T3) = (0,0,1), (0,1,1), (1,0,1), (1,1,1), (1,1,3)). Pure arithmetic on the loop
/// structure of imsrg++ `CountME`, without the model-space cut on the first orbit. Reference value:
/// `{6, 12, 18}` gives 32685840. Complexity O(number of orbit sextuples).
/// @throws InputError if a truncation is negative.
std::int64_t no2b_value_count(const No2bFileTruncation& file);

/// Reads a NO2B file into `v3`, filling exactly the elements imsrg++ `ThreeBodyStorage_no2b::
/// ReadFile` would store, with the container's own truncations (`v3.truncation()`) acting as the
/// model-space cuts.
///
/// Format selection follows imsrg++: if the file NAME (final path component) contains
/// "stream.bin" the file is binary, raw little-endian values without header, of type
/// `options.precision`; otherwise it is text (plain or gzip, detected by content, see
/// `TextInput`) whose first line is a header that is skipped, followed by whitespace-separated
/// decimal values. Values are consumed in imsrg++ order (first orbit a over file orbits in
/// ascending e, l, 2j; b <= a; c; d <= a; e <= d; f; J; T12; T45; T3) and are stored (unnormalized,
/// as in the file; `v3` holds floats) when all six shells satisfy the container's emax, all pair
/// sums its e2max, both triples its e3max, all orbits exist in the container (lmax), the state is
/// not Pauli-forbidden (a == b with J + T12 even, or d == e with J + T45 even) and |v| > 1e-8.
/// Reading stops after the last first-orbit shell the container can hold (a prefix of the file),
/// and blocks that cannot be stored are skipped without conversion (binary) or parsed for
/// validation only (text). Binary files are read in 8 MiB chunks. Complexity O(values read).
/// @throws InputError if the file cannot be opened; for a binary file, if its size differs from
///         `no2b_value_count(file) * sizeof(value)` (the message states both and hints at
///         file3e1max, file3e2max, file3e3max); for a text file, if the data end before the
///         required values or a token is not a number; or if a file truncation is invalid.
void read_no2b(const std::filesystem::path& path, const No2bReadOptions& options,
               op::No2bThreeBody& v3, No2bReadReport* report = nullptr);

}  // namespace imsrg::io
