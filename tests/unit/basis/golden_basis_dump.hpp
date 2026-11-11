#pragma once
/// @file
/// Test helper: strict parser of the "imsrg-basis-dump v1" golden files of tests/golden/basis/
/// (format: tests/golden/README.md). The files hold the orbit list, occupations, partitions,
/// two-body kets/channels/subsets, ph (Pandya) channels and three-body channel sizes of the
/// reference code (imsrg++), against which the basis module is checked.

#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace imsrg::basis::testing {

/// One `orbit` record. `cvq`: 0 core, 1 valence, 2 q-space; `tz2` = -1 proton, +1 neutron.
struct GoldenOrbit {
  int index = 0, n = 0, l = 0, j2 = 0, tz2 = 0;
  double occupation = 0.0;
  int cvq = 0;
};

/// One two-body channel: `channel` + `channel_kets` + nine `channel_subset` records. `kets` are
/// global ket indices in local order; `subsets` maps each of pp hh ph cc vc qc vv qv qq to the
/// local indices (positions in `kets`) it contains.
struct GoldenChannel {
  int ch = 0, J = 0, parity = 0, Tz = 0, nkets = 0;
  std::vector<int> kets;
  std::map<std::string, std::vector<int>> subsets;
};

/// One cross-coupled (Pandya) channel: `ph_channel` + `ph_channel_kets`.
struct GoldenPhChannel {
  int ch = 0, J = 0, parity = 0, Tz = 0, nkets = 0;
  std::vector<int> kets;
};

/// One `threebody_channel` record (only the sizes are stored).
struct GoldenChannel3 {
  int ch = 0, twoJ = 0, parity = 0, twoTz = 0, nkets = 0;
};

/// The content of one golden basis file.
struct GoldenBasisDump {
  std::string source;
  int emax = 0, e2max = 0, e3max = 0, lmax = 0;
  double hw = 0.0;
  std::string reference, valence_space;
  double Aref = 0, Zref = 0, Acore = 0, Zcore = 0;
  int target_mass = 0, target_Z = 0;
  std::vector<GoldenOrbit> orbits;
  std::vector<int> holes, particles, core, valence, qspace;
  std::vector<std::pair<int, int>> kets;  ///< (p, q); the position is the ket index.
  std::vector<GoldenChannel> channels;
  std::vector<GoldenPhChannel> ph_channels;
  std::vector<GoldenChannel3> channels3;
  long long nkets3 = 0, sum_channel_kets3 = 0;
};

/// Parses `file` strictly: mandatory first line, records in the documented order, every count
/// equal to the number of records/indices that follow, `end` last (nothing after it).
/// @throws std::runtime_error naming the file, the line number and the reason on any deviation
///         (including an unreadable file).
GoldenBasisDump read_golden_basis_dump(const std::filesystem::path& file);

/// The directory of the golden basis files, `IMSRG_GOLDEN_DIR "/basis"` (compile definition).
std::filesystem::path golden_basis_dir();

/// File names (not paths) of the 12 golden basis files, sorted.
std::vector<std::string> golden_basis_files();

}  // namespace imsrg::basis::testing
