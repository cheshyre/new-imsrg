// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// RunConfig -> TOML (resolved_config.toml, design/16 sections 1.3, 1.7 and
// 6.1).
#include "imsrg/driver/config/writer.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include "imsrg/base/error.h"
#include "imsrg/driver/config/schema_keys.h"

namespace imsrg::driver {
namespace {

/// A TOML basic string: escapes `"`, `\`, the named control characters, and
/// other control characters (< 0x20, 0x7F) as \u00XX; all other bytes are
/// copied verbatim.
std::string Quote(std::string_view s) {
  std::string out = "\"";
  for (const char ch : s) {
    const auto c = static_cast<unsigned char>(ch);
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\t':
        out += "\\t";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\f':
        out += "\\f";
        break;
      case '\r':
        out += "\\r";
        break;
      default:
        if (c < 0x20 || c == 0x7F) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04X", static_cast<unsigned>(c));
          out += buf;
        } else {
          out += ch;
        }
    }
  }
  out += '"';
  return out;
}

/// A double with 17 significant digits that reads back as a TOML float.
std::string FormatDouble(double v) {
  if (std::isnan(v)) return "nan";
  if (std::isinf(v)) return v < 0 ? "-inf" : "inf";
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.17g", v);
  std::string text = buf;
  if (text.find_first_of(".eEni") == std::string::npos) text += ".0";
  return text;
}

std::string StringArray(const std::vector<std::string>& items) {
  std::string out = "[";
  for (std::size_t i = 0; i < items.size(); ++i) {
    if (i != 0) out += ", ";
    out += Quote(items[i]);
  }
  return out + "]";
}

std::string FromFileArray(const std::vector<OperatorFromFile>& items) {
  std::string out = "[";
  for (std::size_t i = 0; i < items.size(); ++i) {
    const OperatorFromFile& f = items[i];
    if (i != 0) out += ", ";
    out += "{ name = " + Quote(f.name) + ", rank = " + std::to_string(f.rank) +
           ", dtz = " + std::to_string(f.dtz) +
           ", parity = " + std::to_string(f.parity) +
           ", particle_rank = " + std::to_string(f.particle_rank) +
           ", file_2b = " + Quote(f.file_2b);
    if (f.file_3b) out += ", file_3b = " + Quote(*f.file_3b);
    out += ", format = " + Quote(ToString(f.format)) + " }";
  }
  return out + "]";
}

template <class T>
inline constexpr bool kDependentFalse = false;

/// TOML text of one (engaged) field value.
template <class T>
std::string FormatValue(const KeyInfo& key, const T& value) {
  if constexpr (std::is_same_v<T, bool>) {
    return value ? "true" : "false";
  } else if constexpr (std::is_same_v<T, int>) {
    return std::to_string(value);
  } else if constexpr (std::is_same_v<T, double>) {
    return FormatDouble(value);
  } else if constexpr (std::is_same_v<T, std::string>) {
    return Quote(value);
  } else if constexpr (detail::ConfigEnum<T>) {
    return Quote(ToString(value));
  } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
    if (key.string_or_array && value.size() == 1) return Quote(value.front());
    return StringArray(value);
  } else if constexpr (std::is_same_v<T, std::vector<OperatorRequest>>) {
    std::vector<std::string> names;
    for (const OperatorRequest& r : value) names.push_back(r.canonical_name);
    return StringArray(names);
  } else if constexpr (std::is_same_v<T, std::vector<OperatorFromFile>>) {
    return FromFileArray(value);
  } else {
    static_assert(kDependentFalse<T>, "unsupported schema field type");
  }
}

/// `VisitSchema` visitor: collects (table, "key = value") lines in canonical
/// order.
struct LineWriter {
  std::vector<std::pair<std::string, std::string>> lines;
  /// One-stage run: imsrg.generator.valence is inert (resolve resets it to its
  /// default) and an explicit value is V-41 with a valence space, so it is left
  /// out to keep the file loadable.
  bool one_stage = false;

  template <class T>
  void Field(const KeyInfo& key, const T& value) {
    if (one_stage && key.path == "imsrg.generator.valence") return;
    if constexpr (requires { value.has_value(); }) {
      if (value.has_value()) Emit(key, *value);
    } else {
      Emit(key, value);
    }
  }

  template <class Section, class Body>
  void OptionalTable(std::string_view /*table*/,
                     const std::optional<Section>& section, Body&& body) {
    if (section) body(*section);
  }

  /// Appends the line of one key (a helper of `Field`).
  template <class T>
  void Emit(const KeyInfo& key, const T& value) {
    const std::string_view path = key.path;
    const std::size_t dot = path.rfind('.');
    lines.emplace_back(
        std::string(path.substr(0, dot)),
        std::string(path.substr(dot + 1)) + " = " + FormatValue(key, value));
  }
};

}  // namespace

std::string ToToml(const RunConfig& config, bool include_provenance) {
  std::string out =
      "# imsrg resolved configuration (design/16-config-and-cli.md); reproduce "
      "with: imsrg "
      "--config <this file>\n";
  LineWriter writer;
  writer.one_stage = config.imsrg.steps == 1;
  VisitSchema(config, writer);
  std::string current;
  bool first = true;
  for (const auto& [table, line] : writer.lines) {
    if (first || table != current) {
      out += "\n[" + table + "]\n";
      current = table;
      first = false;
    }
    out += line + "\n";
  }
  if (include_provenance) {
    const Provenance& p = config.provenance;
    std::vector<std::string> warnings;
    for (const ConfigIssue& w : p.warnings) warnings.push_back(w.ToString());
    out += "\n[provenance]\n";
    out += "version = " + Quote(p.version) + "\n";
    out += "git_hash = " + Quote(p.git_hash) + "\n";
    out += "date = " + Quote(p.date) + "\n";
    out += "command_line = " + Quote(p.command_line) + "\n";
    out += "legacy_args = " + StringArray(p.legacy_args) + "\n";
    out += "warnings = " + StringArray(warnings) + "\n";
    out += "applied_rules = " + StringArray(p.applied_rules) + "\n";
  }
  return out;
}

void WriteResolvedConfig(const RunConfig& config,
                         const std::filesystem::path& file) {
  const std::string text = ToToml(config);
  if (file.has_parent_path()) {
    std::error_code ec;
    std::filesystem::create_directories(file.parent_path(), ec);
    if (ec) {
      throw InputError("cannot create directory for resolved config '" +
                       file.string() + "': " + ec.message());
    }
  }
  std::ofstream out(file, std::ios::binary | std::ios::trunc);
  out << text;
  out.flush();
  if (!out)
    throw InputError("cannot write resolved config '" + file.string() + "'");
}

}  // namespace imsrg::driver
