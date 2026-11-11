// rh_common.hh -- shared helpers of the golden-data harness `refharness`.
//
// refharness links against the OLD imsrg++ library (imsrg-devel, read-only reference) and is
// never part of the new build. See design/30-testing-and-validation.md section 3 and
// design/15-operator-dump-format.md. Every floating-point number is printed with %.17g.
#pragma once

#include <cstdio>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ModelSpace.hh"
#include "Operator.hh"

namespace rh {

/// Command line: `refharness <subcommand> [key=value ...] [positional ...]`.
/// Keys use the imsrg++ parameter names where one exists (emax, reference, 2bme, ...).
struct Args {
  std::string subcommand;
  std::map<std::string, std::string> kv;
  std::vector<std::string> positional;

  bool has(const std::string& k) const { return kv.count(k) > 0; }
  std::string str(const std::string& k, const std::string& def) const;
  int integer(const std::string& k, int def) const;
  double real(const std::string& k, double def) const;
  bool flag(const std::string& k) const;  ///< key=true, or bare `--key` / `key`
};

Args parse_args(int argc, char** argv);

/// Formats a double with %.17g (round-trips exactly).
std::string fmt17(double x);

/// Reference version string: sha256 over imsrg-devel src/ (no .git in the reference tree),
/// read from $IMSRG_REFERENCE_ROOT/REFERENCE_SRC_SHA256 (written by build_imsrg_devel.sh).
std::string reference_version();

/// Model-space parameters, defaults as in imsrg++ Parameters.hh except where noted.
struct SpaceParams {
  int emax = 2;
  int e3max = 12;            // imsrg++ default e3max
  int lmax = 99999;          // imsrg++ default lmax (no cut)
  double hw = 16.0;          // NOTE: imsrg++ default is 20; all goldens use 16
  std::string reference = "default";
  std::string valence_space = "";
  int A = -1;                // target mass for T_int; -1 = A of the reference
};

SpaceParams space_params(const Args& a);

/// Builds the ModelSpace exactly as imsrg++.cc (lines 331-363) does. The returned object must
/// not be moved or copied while operators refer to it (Ket holds Orbit pointers).
std::unique_ptr<ModelSpace> make_modelspace(const SpaceParams& p);

/// Name used in dumps for the reference: the reference string, or the valence-space core if
/// reference == "default".
std::string reference_name(const SpaceParams& p);

/// Interaction-file parameters (imsrg++ names). Defaults: the project's test files.
struct InteractionParams {
  std::string file2 = "/Users/cheshyre/Downloads/new-imsrg/VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz";
  int file2e1max = 14, file2e2max = 28, file2lmax = 14;
  std::string file3 = "none";
  std::string type3 = "no2b";
  int file3e1max = 6, file3e2max = 12, file3e3max = 18;
  std::string no2b_precision = "single";
};

InteractionParams interaction_params(const Args& a);

/// Bare Hamiltonian pieces as imsrg++.cc builds them (lines 430-541): V2 from the me2j file,
/// optional 3N (no2b storage, particle rank 3), and T_int = imsrg_util::Trel_Op.
struct BareH {
  Operator T;      ///< imsrg_util::Trel_Op (1B + 2B)
  Operator V2;     ///< 2N interaction, normalized antisymmetrized J-coupled
  Operator V3;     ///< rank-3 operator holding only the 3N (no2b mode); empty if no 3N file
  Operator H;      ///< T + V2 (+ V3), particle rank 2 or 3: the imsrg++ Hbare
  bool has3n = false;
};

BareH make_bare_hamiltonian(ModelSpace& ms, const InteractionParams& ip);

// ---------------------------------------------------------------- OperatorDump v1 (spec 15)
struct DumpMeta {
  std::string source;      ///< free text
  std::string reference;   ///< reference name or "custom"
};

/// Writes `op` in OperatorDump v1 format. The 3B section is written as `three_body none`
/// (reserved for M10) unless with_3b is set and the operator has pn-mode 3B storage.
void write_opdump(const std::string& path, const Operator& op, const DumpMeta& meta,
                  bool with_3b = false);

/// Header data of an OperatorDump, needed to build a matching ModelSpace before reading.
struct DumpHeader {
  int emax = -1, lmax = -1;
  double hw = 0;
  std::string reference;
  int particle_rank = 2, rank_J = 0, parity = 0, delta_Tz = 0;
  char hermiticity = 'H';
};

DumpHeader read_opdump_header(const std::string& path);

/// Reads an OperatorDump into a new Operator on `ms` (orbits matched by n, l, 2j, 2tz;
/// occupations must agree with `ms` within 1e-12). Throws std::runtime_error with the line
/// number on malformed input.
Operator read_opdump(const std::string& path, ModelSpace& ms);

/// Builds the ModelSpace described by the header of an OperatorDump (emax, hw, reference;
/// valence_space = `valence_space` key or the dump's reference; e3max key or min(12, 3 emax)).
std::unique_ptr<ModelSpace> modelspace_for_dump(const std::string& path, const Args& a);

// ---------------------------------------------------------------- tiny JSON writer
/// Minimal ordered JSON object writer (numbers with %.17g).
class Json {
 public:
  Json& num(const std::string& k, double v);
  Json& integer(const std::string& k, long long v);
  Json& str(const std::string& k, const std::string& v);
  Json& boolean(const std::string& k, bool v);
  Json& nums(const std::string& k, const std::vector<double>& v);
  Json& ints(const std::string& k, const std::vector<long long>& v);
  Json& raw(const std::string& k, const std::string& json_value);
  std::string dump(int indent = 2) const;
  void write(const std::string& path) const;

 private:
  std::vector<std::pair<std::string, std::string>> items_;
};

std::string json_escape(const std::string& s);

/// Norms of an operator as imsrg++ computes them (Operator::Norm etc.), as a JSON object.
std::string norms_json(const Operator& op);

// ---------------------------------------------------------------- subcommands
int cmd_dump_basis(const Args& a);
int cmd_dump_hamiltonian(const Args& a);
int cmd_hf(const Args& a);
int cmd_roundtrip(const Args& a);
int cmd_norms(const Args& a);
int cmd_commutator(const Args& a);
int cmd_bench_commutator(const Args& a);
int cmd_generator(const Args& a);
int cmd_flow(const Args& a);
int cmd_normal_order(const Args& a);
int cmd_basis_transform(const Args& a);

}  // namespace rh
