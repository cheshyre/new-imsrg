#include "imsrg/driver/config/config_issue.hpp"

#include <algorithm>
#include <utility>

namespace imsrg::driver {

std::string SourceRef::to_string() const {
  switch (kind) {
    case Kind::None:
      return {};
    case Kind::TomlFile:
    case Kind::TomlString: {
      std::string s = name;
      if (line > 0) {
        s += ":" + std::to_string(line);
        if (column > 0) s += ":" + std::to_string(column);
      }
      return s;
    }
    case Kind::LegacyArg:
      return "legacy arg '" + name + "'";
    case Kind::LegacyDefault:
      return "legacy default of '" + name + "'";
    case Kind::SetOption:
      return "--set " + name;
    case Kind::Derived:
      return "derived (" + name + ")";
  }
  return {};
}

std::string ConfigIssue::to_string() const {
  std::string s = severity == Severity::Error ? "error" : "warning";
  if (!code.empty()) s += " " + code;
  if (!key_path.empty()) s += " [" + key_path + "]";
  const std::string origin = where.to_string();
  if (!origin.empty()) s += " (" + origin + ")";
  s += ": " + message;
  if (suggestion) s += " (did you mean '" + *suggestion + "'?)";
  return s;
}

bool has_errors(const ConfigIssues& issues) noexcept {
  return std::any_of(issues.begin(), issues.end(),
                     [](const ConfigIssue& i) { return i.severity == Severity::Error; });
}

ConfigIssue make_error(std::string code, std::string key_path, std::string message, SourceRef where,
                       std::optional<std::string> suggestion) {
  return ConfigIssue{Severity::Error,    std::move(code),       std::move(key_path),
                     std::move(message), std::move(suggestion), std::move(where)};
}

ConfigIssue make_warning(std::string code, std::string key_path, std::string message,
                         SourceRef where, std::optional<std::string> suggestion) {
  return ConfigIssue{Severity::Warning,  std::move(code),       std::move(key_path),
                     std::move(message), std::move(suggestion), std::move(where)};
}

int edit_distance(std::string_view a, std::string_view b) {
  // Optimal string alignment: insertions, deletions, substitutions, adjacent transpositions.
  const std::size_t n = a.size();
  const std::size_t m = b.size();
  std::vector<std::vector<int>> d(n + 1, std::vector<int>(m + 1, 0));
  for (std::size_t i = 0; i <= n; ++i) d[i][0] = static_cast<int>(i);
  for (std::size_t j = 0; j <= m; ++j) d[0][j] = static_cast<int>(j);
  for (std::size_t i = 1; i <= n; ++i) {
    for (std::size_t j = 1; j <= m; ++j) {
      const int cost = a[i - 1] == b[j - 1] ? 0 : 1;
      d[i][j] = std::min({d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost});
      if (i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1]) {
        d[i][j] = std::min(d[i][j], d[i - 2][j - 2] + 1);
      }
    }
  }
  return d[n][m];
}

std::optional<std::string> did_you_mean(std::string_view word,
                                        std::span<const std::string_view> candidates) {
  constexpr int kMaxDistance = 2;
  constexpr std::size_t kMinPrefix = 4;
  std::optional<std::string_view> best;
  int best_distance = kMaxDistance + 1;
  for (std::string_view c : candidates) {
    if (c == word) continue;
    const int dist = edit_distance(word, c);
    if (dist < best_distance) {
      best_distance = dist;
      best = c;
    }
  }
  if (best) return std::string(*best);
  std::size_t best_prefix = 0;
  for (std::string_view c : candidates) {
    if (c == word) continue;
    const auto mismatch = std::mismatch(word.begin(), word.end(), c.begin(), c.end());
    const auto prefix = static_cast<std::size_t>(mismatch.first - word.begin());
    if (prefix > best_prefix) {
      best_prefix = prefix;
      best = c;
    }
  }
  if (best && best_prefix >= kMinPrefix) return std::string(*best);
  return std::nullopt;
}

std::optional<std::string> did_you_mean(std::string_view word,
                                        const std::vector<std::string>& candidates) {
  std::vector<std::string_view> views(candidates.begin(), candidates.end());
  return did_you_mean(word, std::span<const std::string_view>(views));
}

}  // namespace imsrg::driver
