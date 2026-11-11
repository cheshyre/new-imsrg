#pragma once
/// @file
/// OperatorDump v1 reader and writer: the plain-text exchange format with imsrg-devel
/// (`tools/reference/refharness`), specified in design/15-operator-dump-format.md.
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "imsrg/basis/model_space.hpp"
#include "imsrg/basis/orbit.hpp"
#include "imsrg/operator/operator.hpp"
#include "imsrg/operator/operator_spec.hpp"

namespace imsrg::op {

/// One `orbit` record of a dump: quantum numbers and reference occupation.
struct DumpOrbit {
  basis::Orbit orbit;       ///< n, l, 2j, 2tz (2tz = -1 proton).
  double occupation = 0.0;  ///< Reference occupation in [0, 1].
};

/// Everything before the `zero_body` record of a dump.
struct OperatorDumpHeader {
  std::string source;            ///< Free text of the `source` record.
  std::int32_t emax = 0;         ///< `basis emax`.
  std::int32_t lmax = 0;         ///< `basis lmax` (min(lmax, emax) as written by imsrg-devel).
  double hw = 0.0;               ///< `basis hw` (hbar omega, MeV).
  std::string reference;         ///< `reference` record (a nucleus name or "custom").
  std::vector<DumpOrbit> orbits; ///< The `orbit` records in file order (index = position).
  OperatorSpec spec;             ///< The `spec` record.
};

/// Options of the readers.
struct OperatorDumpReadOptions {
  /// Reject the file (InputError) if an orbit occupation differs from the model-space reference
  /// by more than `occupation_tolerance`. Disable to read an operator into a model space with
  /// another reference (the matrix elements do not depend on it).
  bool check_occupations = true;
  /// Tolerance of the occupation check.
  double occupation_tolerance = 1e-12;
};

/// Options of the writers.
struct OperatorDumpWriteOptions {
  /// Text of the `source` record (code, version, what the operator is). Must not contain a
  /// newline.
  std::string source = "imsrg";
  /// Text of the `reference` record; empty means the model-space reference name.
  std::string reference;
};

/// Reads the header records (up to and including `spec`) of the dump at `path`.
/// @throws InputError ("<path>:<line>: <message>") if the file cannot be read, the version line
///         is not `# imsrg-operator-dump v1`, or a record is missing or malformed.
OperatorDumpHeader read_operator_dump_header(const std::filesystem::path& path);

/// A model space matching a dump header: the harmonic-oscillator basis with the header's emax,
/// lmax and hw; the reference occupations of the orbit records (named `header.reference`); the
/// partition with the holes (occupation >= kOccupationCut) as core and all other orbits as
/// q-space (a dump does not record a valence space).
/// @throws InputError if an orbit of the header is not in that basis.
std::shared_ptr<const basis::ModelSpace> model_space_for_dump(const OperatorDumpHeader& header);

/// Reads the dump at `path` into an operator on `ms`. Orbits are mapped to `ms` by their quantum
/// numbers (n, l, 2j, 2tz), so `ms` may be larger than the dump's basis. Every record is
/// validated (design/15): version, section order, counts, index ranges, p <= q and r <= s,
/// kets that exist in their channel and channel pairs allowed by the spec; hw equal to the
/// model space's (1e-12 relative); occupations (see the options); and the hermiticity declared
/// in `spec` within 1e-12 * max|V| (one-body and two-body parts).
/// @throws InputError ("<path>:<line>: <message>") on any violation.
/// @throws NotImplementedError for specs `validate` rejects and for a non-empty `three_body`
///         section (reserved for M10).
Operator read_operator_dump(const std::filesystem::path& path,
                            std::shared_ptr<const basis::ModelSpace> ms,
                            const OperatorDumpReadOptions& options = {});

/// Stream version of `read_operator_dump`; `name` labels error messages.
Operator read_operator_dump(std::istream& in, std::string_view name,
                            std::shared_ptr<const basis::ModelSpace> ms,
                            const OperatorDumpReadOptions& options = {});

/// Writes `op` as an OperatorDump v1, in the record order and number format of the imsrg-devel
/// harness writer, so that harness dumps round-trip byte-identically: values with `%.17g`; the
/// one-body entries row by row (a, then b) and the two-body entries channel by channel, bra
/// (local index), then ket, both only where the value is nonzero (so both (pq, rs) and (rs, pq)
/// appear); `three_body none`. The `basis` record holds the model space's emax, effective lmax
/// and hw; the orbit records its orbits and reference occupations.
/// @throws InputError if the file cannot be written.
void write_operator_dump(const std::filesystem::path& path, const Operator& op,
                         const OperatorDumpWriteOptions& options = {});

/// Stream version of `write_operator_dump`.
void write_operator_dump(std::ostream& out, const Operator& op,
                         const OperatorDumpWriteOptions& options = {});

}  // namespace imsrg::op
