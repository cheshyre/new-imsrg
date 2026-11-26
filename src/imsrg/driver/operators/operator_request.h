// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_DRIVER_OPERATORS_OPERATOR_REQUEST_H_
#define IMSRG_DRIVER_OPERATORS_OPERATOR_REQUEST_H_
/// @file
/// Operator requests: the table-driven operator-name grammar and catalog of
/// design/16 section 4, which replaces imsrg++'s `OperatorFromString` if/else
/// chain.
///
/// A spec is either a string in the legacy grammar (`"Rp2"`, `"rhop_3.0"`,
/// `"OBD_p0s1_p0s1"`) or an inline table with the catalog's parameter names (`{
/// name = "rhop", r = 3.0 }`); both normalise to the same `OperatorRequest`.
/// Grammar: `spec := name ("_" param)*`; the catalog, not the grammar, decides
/// how many leading `_`-tokens form the name (longest catalog prefix, so
/// "R2_p1" and "rhop_all" are names). Parameters are typed: integer (fully
/// consumed), number (C-locale floating point, fully consumed,
/// std::from_chars), orbit label (imsrg++ form, "p0s1", within the model space
/// emax) or word.
///
/// Phase-1 operators are scalar (J = 0, dTz = 0, even parity, particle rank <=
/// 2). Planned names are recognised and rejected with `E-PLANNED`; atomic names
/// with `E-OP-NS`.

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "imsrg/driver/config/config_issue.h"
#include "imsrg/driver/config/expected.h"
#include "imsrg/driver/config/phase.h"
#include "imsrg/driver/config/raw_tree.h"

namespace imsrg::driver {

/// The phase-1 operator families (one per P1 catalog entry; macros expand to
/// these).
enum class OperatorKind {
  kTrel,                ///< intrinsic kinetic energy, "Trel"
  kTcm,                 ///< center-of-mass kinetic energy, "TCM"
  kTlab,                ///< lab-frame kinetic energy, "Tlab"
  kTrelMassCorrection,  ///< proton-neutron mass splitting correction,
                        ///< "TrelMassCorrection"
  kHcm,                 ///< center-of-mass Hamiltonian, "HCM" or "HCM_<hw>"
  kVcm,                 ///< center-of-mass potential, "VCM_<hw>"
  kR2cm,                ///< center-of-mass radius squared, "R2CM"
  kRp2,                 ///< intrinsic point-proton radius squared, "Rp2"
  kRn2,                 ///< intrinsic point-neutron radius squared, "Rn2"
  kRm2,                 ///< intrinsic matter radius squared, "Rm2"
  kRm2lab,              ///< lab-frame matter radius squared, "Rm2lab"
  kRp2Z,    ///< point-proton radius squared for proton number Z, "Rp2Z_<Z>"
  kRp2Az,   ///< point-proton radius squared for A and Z, "Rp2AZ_<A>_<Z>"
  kRn2Z,    ///< point-neutron radius squared for proton number Z, "Rn2Z_<Z>"
  kR2p1,    ///< one-body proton r^2, "R2_p1"
  kR2n1,    ///< one-body neutron r^2, "R2_n1"
  kR2p2,    ///< two-body proton r^2 part, "R2_p2"
  kR2n2,    ///< two-body neutron r^2 part, "R2_n2"
  kRso,     ///< spin-orbit radius correction, "Rso"
  kE0,      ///< monopole operator, "E0"
  kOccRef,  ///< occupation number of the reference orbits, "OccRef"
  kOneOcc,  ///< number operator of one orbit, "OneOcc_<orb>"
  kAllOcc,  ///< number operator of all orbits with the (l, j, tz) of <orb>,
            ///< "AllOcc_<orb>"
  kObd,     ///< one-body density matrix element, "OBD_<orb>_<orb>"
  kRhoP,    ///< point-proton density at radius r (fm), "rhop_<r>"
  kRhoN,    ///< point-neutron density at radius r (fm), "rhon_<r>"
  kRhoCentralP,  ///< central proton density within rmax (fm),
                 ///< "rhocentralp_<rmax>"
  kRhoCentralN,  ///< central neutron density within rmax (fm),
                 ///< "rhocentraln_<rmax>"
  kIso2,         ///< total isospin squared, "Iso2"
  kTz2,          ///< isospin projection squared, "Tz2"
  kLdotS,        ///< one-body l.s, "LdotS"
  kNreft,        ///< scalar dark-matter NREFT shorthand, "FM0p_<q>",
                 ///< "FPhipp0n_<q>" (see opslib/nreft.h)
};

/// A typed operator parameter: integer, number (double), or a string (orbit
/// label in canonical imsrg++ form, or an enumerated word).
using OpParam = std::variant<std::int64_t, double, std::string>;

/// Tensor structure of an operator, as recorded in the catalog. Plain integers
/// because this is user-facing catalog data; the pipeline converts it to the
/// operator module's strong types.
struct OperatorShape {
  int rank_j = 0;  ///< spherical tensor rank J (not doubled)
  int dtz = 0;     ///< change of twice the total isospin projection (charge
                   ///< changing != 0)
  int parity = 0;  ///< 0 = even, 1 = odd
  int particle_rank = 2;  ///< highest particle rank (1, 2 or 3)

  friend bool operator==(const OperatorShape&, const OperatorShape&) = default;
};

/// One validated operator to transform with the flow.
struct OperatorRequest {
  OperatorKind kind = OperatorKind::kTrel;
  std::string canonical_name;  ///< e.g. "rhop_3" for "rhop_3.0"; re-parses to
                               ///< the same request
  std::vector<OpParam>
      params;                ///< in catalog order, typed per the catalog entry
  OperatorShape shape;       ///< from the catalog
  Phase phase = Phase::kP1;  ///< always P1 for a request that passed validation

  friend bool operator==(const OperatorRequest&,
                         const OperatorRequest&) = default;
};

/// Type of one catalog parameter.
enum class OpParamType {
  kInteger,  ///< decimal integer, fully consumed
  kNumber,   ///< C-locale floating-point number, fully consumed
  kOrbit,    ///< imsrg++ orbit label inside the model space ("p0s1")
  kWord,     ///< enumerated token [A-Za-z0-9+-]+
};

/// Schema of one catalog parameter.
struct OpParamSpec {
  std::string_view
      name;  ///< inline-table key, e.g. "r", "hw", "z", "a", "orbit1"
  OpParamType type = OpParamType::kNumber;
  bool non_negative = false;  ///< numbers: reject values < 0
  std::string_view doc;       ///< e.g. "radius in fm"
};

/// One catalog row (one operator family). The same table feeds
/// `--list-operators`.
struct OperatorCatalogEntry {
  std::string_view
      name;  ///< dispatch name, may contain '_' ("R2_p1", "rhop_all")
  std::span<const OpParamSpec> params;  ///< parameter schema (P1 entries)
  int min_params = 0;     ///< usually params.size(); "HCM" accepts 0 or 1
  bool variadic = false;  ///< PL/NS rows: parameters are not checked
  bool macro =
      false;  ///< expands to several requests (rhop_all, rhon_all, ...)
  OperatorShape shape;  ///< tensor structure
  Phase phase = Phase::kP1;
  std::optional<OperatorKind>
      kind;              ///< set for P1 rows (and P1 macros: the expanded kind)
  std::string_view doc;  ///< one line
};

/// The whole catalog: P1 rows, then planned rows, then not-supported rows
/// (design/16 4.5).
std::span<const OperatorCatalogEntry> OperatorCatalog() noexcept;

/// What the parser needs to know about the model space.
struct ModelSpaceInfo {
  int emax = 0;  ///< orbit labels must have e = 2n + l <= emax
};

/// Parses a list of operator specs (each a string or an inline table, as in
/// `operators.list`). Macros are expanded (`rhop_all` -> 51 requests r = 0.0,
/// 0.2, ..., 10.0), duplicates (same canonical name after expansion) are
/// `E-OP-DUPLICATE`. All errors are collected; `key_path` (e.g.
/// "operators.list") is put into every issue, `where` is taken from each spec.
Expected<std::vector<OperatorRequest>, ConfigIssues> ParseOperatorList(
    std::span<const RawValue> specs, const ModelSpaceInfo& info,
    std::string_view key_path);

/// Parses one spec string (convenience for tests and the legacy layer); may
/// yield several requests for a macro.
Expected<std::vector<OperatorRequest>, ConfigIssues> ParseOperatorSpec(
    std::string_view spec, const ModelSpaceInfo& info);

/// The catalog as text for `imsrg --list-operators`: one line per row with
/// name, parameters, shape, phase and doc, grouped by phase.
std::string OperatorCatalogText();

}  // namespace imsrg::driver

#endif  // IMSRG_DRIVER_OPERATORS_OPERATOR_REQUEST_H_
