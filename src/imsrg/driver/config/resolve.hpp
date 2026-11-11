#pragma once
/// @file
/// Raw tree -> `RunConfig` (design/16 sections 1.3, 5 and 6.1): typed reading, defaults, derived
/// values, then validation (rules V-xx), all issues accumulated.

#include "imsrg/driver/config/config_issue.hpp"
#include "imsrg/driver/config/raw_tree.hpp"
#include "imsrg/driver/config/schema.hpp"

namespace imsrg::driver {

/// Options of `resolve` / `validate`.
struct ResolveOptions {
  /// The raw tree came from the legacy layer: rules that design/16 downgrades for legacy input
  /// (V-21, V-40, V-41, V-42, ...) report warnings instead of errors.
  bool legacy = false;
  /// Check that input files exist and are readable and that output/scratch directories are
  /// creatable (V-14). Tests of pure configuration logic turn this off.
  bool check_files = true;
};

/// Result of `resolve`. `config` is meaningful only if `issues` has no error.
struct ResolveResult {
  RunConfig config;
  ConfigIssues issues;  ///< errors and warnings, in the order found
};

/// Converts `tree` into a resolved `RunConfig`:
/// 1. typed reading of every schema key present (E-TYPE, V-01 enum spellings with suggestion,
///    V-03 required keys, operator specs via `parse_operator_list`);
/// 2. derived values: effective basis.e3max = min(e3max, 3*emax); two_body e2max = 2*e1max and
///    lmax = e1max if absent; mass_number from reference.nucleus or the valence core (V-23);
///    hamiltonian.beta_cm_hw = basis.hw; kinetic = none for nushellx files and lab with trap_hw
///    if not given; imsrg.steps = 2 iff a valence space is defined; imsrg.space defaults;
///    output.stem and output.flow_file; generator.valence reset to its default if steps == 1;
/// 3. `validate`.
/// Does not touch `provenance`.
ResolveResult resolve(const RawTree& tree, const ResolveOptions& options);

/// The validation rules of design/16 section 5 (V-02, V-10 ... V-63, E-PLANNED) on a resolved
/// config. `tree` tells which keys were given explicitly (V-41, Q2). Returns errors and warnings.
ConfigIssues validate(const RunConfig& config, const RawTree& tree, const ResolveOptions& options);

}  // namespace imsrg::driver
