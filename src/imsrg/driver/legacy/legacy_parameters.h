// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_DRIVER_LEGACY_LEGACY_PARAMETERS_H_
#define IMSRG_DRIVER_LEGACY_LEGACY_PARAMETERS_H_
/// @file
/// The legacy layer (design/16 section 3): imsrg++'s `key=value` command line
/// (the 85 keys of `Parameters.hh` plus the fork key `use_HF_reference_in_NAT`) translated into the raw tree of the new
/// schema, with the shorthand methods, sentinel conversions, quirk decisions
/// Q1-Q16 and warning rules L1-L3 of design/16 3.5-3.6.
///
/// The translator never parses operator names (that is
/// `operators/operator_request`), never resolves derived values (that is
/// `resolve`), and supplies the old defaults only where the new schema has no
/// default or a different one (emax 6, hw 20, file2e1max 12, file3e1max 12,
/// file3e3max 12), so that a legacy command and the equivalent TOML resolve to
/// equal configs.

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "imsrg/driver/config/config_issue.h"
#include "imsrg/driver/config/raw_tree.h"

namespace imsrg::driver {

/// Value kind of an old key (the four maps of `Parameters.hh`).
enum class LegacyKind { kString, kDouble, kInt, kVector };

/// Status of an old key (design/16 section 3): S, C, NS, PL.
enum class LegacyStatus { kSupported, kChanged, kNotSupported, kPlanned };

/// One row of the legacy table.
struct LegacyKeyInfo {
  std::string_view key;  ///< old key, e.g. "2bme"
  LegacyKind kind = LegacyKind::kString;
  std::string_view old_default;  ///< as in Parameters.hh, e.g. "none", "20",
                                 ///< "-1", "" (vectors)
  LegacyStatus status = LegacyStatus::kSupported;
  std::string_view
      new_key;  ///< new path(s), e.g. "interaction.two_body.file"; "" for NS
  std::string_view note;  ///< one line (sentinels, shorthands, messages)
};

/// All 85 keys of `Parameters.hh` in its order (plus use_HF_reference_in_NAT) (strings, doubles, ints,
/// vectors).
std::span<const LegacyKeyInfo> LegacyKeys() noexcept;

/// The row of `key` (exact, case sensitive), or nullptr.
const LegacyKeyInfo* FindLegacyKey(std::string_view key) noexcept;

/// Result of translating a legacy command line.
struct LegacyTranslation {
  RawTree tree;         ///< raw tree in new-schema paths
  ConfigIssues issues;  ///< errors and warnings
  std::vector<std::string>
      applied_rules;  ///< e.g. "method=NSmagnus -> imsrg.method=magnus, ..."
};

/// Translates legacy tokens (`key=value` each, without the program name).
/// Errors: token without '=' or with an empty value (`E-LEGACY-SYNTAX`),
/// unknown key (`E-KEY-UNKNOWN` with did-you-mean over the 85 keys), unparsable
/// int/double/bool (`E-LEGACY-VALUE`), unknown enum spelling (`V-01` with
/// suggestion), not supported (`E-LEGACY-NS`). Scalars given twice: last wins
/// plus `W-DUPLICATE`; vector keys append. Every value keeps its token as
/// `SourceRef::LegacyArg`.
LegacyTranslation TranslateLegacy(std::span<const std::string> args);

/// The legacy table as text for `imsrg --legacy-keys`: old key, kind, old
/// default, status, new key and note, one row per key.
std::string LegacyKeysText();

}  // namespace imsrg::driver

#endif  // IMSRG_DRIVER_LEGACY_LEGACY_PARAMETERS_H_
