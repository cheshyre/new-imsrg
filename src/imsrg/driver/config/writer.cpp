// RunConfig -> TOML (resolved_config.toml, design/16 sections 1.3, 1.7 and 6.1).
#include "imsrg/driver/config/writer.hpp"

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

#include "imsrg/base/error.hpp"
#include "imsrg/driver/config/schema_keys.hpp"

namespace imsrg::driver {
namespace {

/// A TOML basic string: escapes `"`, `\`, the named control characters, and other control
/// characters (< 0x20, 0x7F) as \u00XX; all other bytes are copied verbatim.
std::string quote(std::string_view s) {
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
std::string format_double(double v) {
  if (std::isnan(v)) return "nan";
  if (std::isinf(v)) return v < 0 ? "-inf" : "inf";
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.17g", v);
  std::string text = buf;
  if (text.find_first_of(".eEni") == std::string::npos) text += ".0";
  return text;
}

std::string string_array(const std::vector<std::string>& items) {
  std::string out = "[";
  for (std::size_t i = 0; i < items.size(); ++i) {
    if (i != 0) out += ", ";
    out += quote(items[i]);
  }
  return out + "]";
}

std::string from_file_array(const std::vector<OperatorFromFile>& items) {
  std::string out = "[";
  for (std::size_t i = 0; i < items.size(); ++i) {
    const OperatorFromFile& f = items[i];
    if (i != 0) out += ", ";
    out += "{ name = " + quote(f.name) + ", rank = " + std::to_string(f.rank) +
           ", dtz = " + std::to_string(f.dtz) + ", parity = " + std::to_string(f.parity) +
           ", particle_rank = " + std::to_string(f.particle_rank) +
           ", file_2b = " + quote(f.file_2b);
    if (f.file_3b) out += ", file_3b = " + quote(*f.file_3b);
    out += ", format = " + quote(to_string(f.format)) + " }";
  }
  return out + "]";
}

template <class T>
inline constexpr bool dependent_false = false;

/// TOML text of one (engaged) field value.
template <class T>
std::string format_value(const KeyInfo& key, const T& value) {
  if constexpr (std::is_same_v<T, bool>) {
    return value ? "true" : "false";
  } else if constexpr (std::is_same_v<T, int>) {
    return std::to_string(value);
  } else if constexpr (std::is_same_v<T, double>) {
    return format_double(value);
  } else if constexpr (std::is_same_v<T, std::string>) {
    return quote(value);
  } else if constexpr (ConfigEnum<T>) {
    return quote(to_string(value));
  } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
    if (key.string_or_array && value.size() == 1) return quote(value.front());
    return string_array(value);
  } else if constexpr (std::is_same_v<T, std::vector<OperatorRequest>>) {
    std::vector<std::string> names;
    for (const OperatorRequest& r : value) names.push_back(r.canonical_name);
    return string_array(names);
  } else if constexpr (std::is_same_v<T, std::vector<OperatorFromFile>>) {
    return from_file_array(value);
  } else {
    static_assert(dependent_false<T>, "unsupported schema field type");
  }
}

/// `visit_schema` visitor: collects (table, "key = value") lines in canonical order.
struct LineWriter {
  std::vector<std::pair<std::string, std::string>> lines;
  /// One-stage run: imsrg.generator.valence is inert (resolve resets it to its default) and an
  /// explicit value is V-41 with a valence space, so it is left out to keep the file loadable.
  bool one_stage = false;

  template <class T>
  void field(const KeyInfo& key, const T& value) {
    if (one_stage && key.path == "imsrg.generator.valence") return;
    if constexpr (requires { value.has_value(); }) {
      if (value.has_value()) emit(key, *value);
    } else {
      emit(key, value);
    }
  }

  template <class Section, class Body>
  void optional_table(std::string_view /*table*/, const std::optional<Section>& section,
                      Body&& body) {
    if (section) body(*section);
  }

 private:
  template <class T>
  void emit(const KeyInfo& key, const T& value) {
    const std::string_view path = key.path;
    const std::size_t dot = path.rfind('.');
    lines.emplace_back(std::string(path.substr(0, dot)),
                       std::string(path.substr(dot + 1)) + " = " + format_value(key, value));
  }
};

}  // namespace

std::string to_toml(const RunConfig& config, bool include_provenance) {
  std::string out =
      "# imsrg resolved configuration (design/16-config-and-cli.md); reproduce with: imsrg "
      "--config <this file>\n";
  LineWriter writer;
  writer.one_stage = config.imsrg.steps == 1;
  visit_schema(config, writer);
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
    for (const ConfigIssue& w : p.warnings) warnings.push_back(w.to_string());
    out += "\n[provenance]\n";
    out += "version = " + quote(p.version) + "\n";
    out += "git_hash = " + quote(p.git_hash) + "\n";
    out += "date = " + quote(p.date) + "\n";
    out += "command_line = " + quote(p.command_line) + "\n";
    out += "legacy_args = " + string_array(p.legacy_args) + "\n";
    out += "warnings = " + string_array(warnings) + "\n";
    out += "applied_rules = " + string_array(p.applied_rules) + "\n";
  }
  return out;
}

void write_resolved_config(const RunConfig& config, const std::filesystem::path& file) {
  const std::string text = to_toml(config);
  if (file.has_parent_path()) {
    std::error_code ec;
    std::filesystem::create_directories(file.parent_path(), ec);
    if (ec) {
      throw InputError("cannot create directory for resolved config '" + file.string() +
                       "': " + ec.message());
    }
  }
  std::ofstream out(file, std::ios::binary | std::ios::trunc);
  out << text;
  out.flush();
  if (!out) throw InputError("cannot write resolved config '" + file.string() + "'");
}

}  // namespace imsrg::driver
