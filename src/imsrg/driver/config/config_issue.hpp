#pragma once
/// @file
/// Configuration diagnostics (design/16 sections 1.4 and 6.2): where a value came from
/// (`SourceRef`), one error or warning (`ConfigIssue`), and the did-you-mean helper.
///
/// Issue codes are stable and tested. The families are:
/// * `E-TOML-SYNTAX` (TOML parse error), `E-FILE` (config file unreadable), `E-KEY-UNKNOWN`
///   (unknown key or table, TOML / `--set` / legacy, with a suggestion), `E-TYPE` (wrong value
///   type), `E-SET-SYNTAX` (`--set` without `=`), `E-CLI-MIX` (`--config` plus legacy tokens);
/// * legacy layer: `E-LEGACY-SYNTAX` (token without `=`, empty value), `E-LEGACY-VALUE`
///   (unparsable number or boolean, malformed tag), `E-LEGACY-NS` (not supported key or value);
/// * `E-PLANNED` (planned-later value, key or operator), operator grammar `E-OP-UNKNOWN`,
///   `E-OP-PARAMS`, `E-OP-ORBIT`, `E-OP-NS`, `E-OP-DUPLICATE`;
/// * validation rules `V-01` ... `V-63` (design/16 section 5); `V-14` (input files) maps to exit
///   code 3, every other error to exit code 2;
/// * warnings: a downgraded rule keeps its `V-xx` code with `Severity::Warning`; legacy-only
///   warnings are `W-DUPLICATE` (scalar key given twice), `W-NSTEPS1` (Q1), `W-L1`, `W-L2`
///   (design/16 section 3.6 rules) and `W-Qn` for quirk n of design/16 section 3.6.

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace imsrg::driver {

/// Severity of a configuration issue. Errors stop the run before any computation; warnings are
/// printed and recorded in the provenance.
enum class Severity { Warning, Error };

/// Where a configuration value (or an issue) comes from.
struct SourceRef {
  /// Kind of source.
  enum class Kind {
    None,           ///< unknown / not applicable
    TomlFile,       ///< `name` = file path; `line`, `column` 1-based (0 = unknown)
    TomlString,     ///< `name` = a label of the TOML text (e.g. "<string>")
    LegacyArg,      ///< `name` = the verbatim legacy token, e.g. "emax=abc"
    LegacyDefault,  ///< `name` = the legacy key whose old default was supplied
    SetOption,      ///< `name` = the verbatim `--set` assignment, e.g. "basis.emax=4"
    Derived,        ///< computed by the resolver; `name` says from what
  };
  Kind kind = Kind::None;
  std::string name;
  int line = 0;
  int column = 0;

  /// Human-readable form: "run.toml:12:3", "legacy arg 'emax=abc'", "--set basis.emax=4",
  /// "legacy default of 'emax'", "derived (...)", or "" for `Kind::None`.
  std::string to_string() const;

  friend bool operator==(const SourceRef&, const SourceRef&) = default;
};

/// One configuration error or warning.
struct ConfigIssue {
  Severity severity = Severity::Error;
  std::string code;      ///< stable id, see the file comment
  std::string key_path;  ///< new-schema path ("basis.emax"); legacy issues may name the old key
  std::string message;   ///< what is wrong and what was expected
  std::optional<std::string> suggestion;  ///< did-you-mean candidate (unknown keys / values)
  SourceRef where;                        ///< origin of the offending value

  /// One line: "error V-10 [basis.emax] (run.toml:3:1): <message> (did you mean 'x'?)".
  /// Empty parts are left out.
  std::string to_string() const;

  friend bool operator==(const ConfigIssue&, const ConfigIssue&) = default;
};

/// A list of issues, in the order they were found.
using ConfigIssues = std::vector<ConfigIssue>;

/// True if any issue in `issues` is an error.
bool has_errors(const ConfigIssues& issues) noexcept;

/// An error issue.
ConfigIssue make_error(std::string code, std::string key_path, std::string message,
                       SourceRef where = {}, std::optional<std::string> suggestion = std::nullopt);

/// A warning issue.
ConfigIssue make_warning(std::string code, std::string key_path, std::string message,
                         SourceRef where = {},
                         std::optional<std::string> suggestion = std::nullopt);

/// Damerau-Levenshtein (optimal string alignment) distance between `a` and `b`, case sensitive.
int edit_distance(std::string_view a, std::string_view b);

/// The best did-you-mean candidate for `word` (design/16 V-01): the candidate with the smallest
/// edit distance if that distance is <= 2 (ties: the first in `candidates`); otherwise the
/// candidate with the longest common prefix with `word` if that prefix has >= 4 characters;
/// otherwise nullopt. A candidate equal to `word` is never suggested.
std::optional<std::string> did_you_mean(std::string_view word,
                                        std::span<const std::string_view> candidates);

/// Overload for owned candidate strings.
std::optional<std::string> did_you_mean(std::string_view word,
                                        const std::vector<std::string>& candidates);

}  // namespace imsrg::driver
