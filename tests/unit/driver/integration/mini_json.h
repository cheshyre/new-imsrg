// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#ifndef TESTS_UNIT_DRIVER_INTEGRATION_MINI_JSON_H_
#define TESTS_UNIT_DRIVER_INTEGRATION_MINI_JSON_H_
/// @file
/// A minimal JSON reader for the integration tests (objects, arrays, strings,
/// numbers, true, false, null). The driver's `results.json` is machine written
/// with `%.17g` numbers, so no escapes beyond the common ones and no unicode
/// handling are needed.

#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace imsrg::it {

/// One JSON value. Numbers are `double`; `null` has kind Null.
struct Json {
  enum class Kind { kNull, kBool, kNumber, kString, kArray, kObject };
  Kind kind = Kind::kNull;
  bool boolean = false;
  double number = 0.0;
  std::string text;
  std::vector<Json> items;  ///< array elements
  std::vector<std::pair<std::string, Json>>
      fields;  ///< object members, in file order

  /// True if this is an object with member `key`.
  bool has(const std::string& key) const {
    for (const auto& f : fields)
      if (f.first == key) return true;
    return false;
  }
  /// Member `key` of an object. @throws std::runtime_error if absent.
  const Json& operator[](const std::string& key) const {
    for (const auto& f : fields)
      if (f.first == key) return f.second;
    throw std::runtime_error("JSON key not found: " + key);
  }
  /// Element `i` of an array. @throws std::out_of_range.
  const Json& at(std::size_t i) const { return items.at(i); }
  /// Number of array elements.
  std::size_t size() const { return items.size(); }
};

namespace detail {

class JsonParser {
 public:
  explicit JsonParser(const std::string& s) : s_(s) {}
  Json Parse() {
    Json v = value();
    Skip();
    if (pos_ != s_.size()) Fail("trailing characters");
    return v;
  }

 private:
  const std::string& s_;
  std::size_t pos_ = 0;

  [[noreturn]] void Fail(const std::string& what) const {
    throw std::runtime_error("JSON parse error at offset " +
                             std::to_string(pos_) + ": " + what);
  }
  void Skip() {
    while (pos_ < s_.size() &&
           std::isspace(static_cast<unsigned char>(s_[pos_])))
      ++pos_;
  }
  char peek() {
    Skip();
    if (pos_ >= s_.size()) Fail("unexpected end");
    return s_[pos_];
  }
  void Expect(char c) {
    if (peek() != c) Fail(std::string("expected '") + c + "'");
    ++pos_;
  }
  bool Literal(const char* word) {
    const std::string w(word);
    if (s_.compare(pos_, w.size(), w) == 0) {
      pos_ += w.size();
      return true;
    }
    return false;
  }
  std::string StringLiteral() {
    Expect('"');
    std::string out;
    while (pos_ < s_.size() && s_[pos_] != '"') {
      char c = s_[pos_++];
      if (c == '\\' && pos_ < s_.size()) {
        const char e = s_[pos_++];
        switch (e) {
          case 'n':
            c = '\n';
            break;
          case 't':
            c = '\t';
            break;
          case 'r':
            c = '\r';
            break;
          default:
            c = e;
            break;
        }
      }
      out.push_back(c);
    }
    Expect('"');
    return out;
  }
  Json value() {
    Json v;
    const char c = peek();
    if (c == '{') {
      ++pos_;
      v.kind = Json::Kind::kObject;
      if (peek() == '}') {
        ++pos_;
        return v;
      }
      while (true) {
        std::string key = StringLiteral();
        Expect(':');
        v.fields.emplace_back(std::move(key), value());
        if (peek() == ',') {
          ++pos_;
          continue;
        }
        Expect('}');
        return v;
      }
    }
    if (c == '[') {
      ++pos_;
      v.kind = Json::Kind::kArray;
      if (peek() == ']') {
        ++pos_;
        return v;
      }
      while (true) {
        v.items.push_back(value());
        if (peek() == ',') {
          ++pos_;
          continue;
        }
        Expect(']');
        return v;
      }
    }
    if (c == '"') {
      v.kind = Json::Kind::kString;
      v.text = StringLiteral();
      return v;
    }
    if (Literal("true")) {
      v.kind = Json::Kind::kBool;
      v.boolean = true;
      return v;
    }
    if (Literal("false")) {
      v.kind = Json::Kind::kBool;
      return v;
    }
    if (Literal("null")) return v;
    const char* begin = s_.c_str() + pos_;
    char* end = nullptr;
    v.number = std::strtod(begin, &end);
    if (end == begin) Fail("bad value");
    v.kind = Json::Kind::kNumber;
    pos_ += static_cast<std::size_t>(end - begin);
    return v;
  }
};

}  // namespace detail

/// Parses `text`. @throws std::runtime_error on malformed input.
inline Json ParseJson(const std::string& text) {
  return detail::JsonParser(text).Parse();
}

}  // namespace imsrg::it

#endif  // TESTS_UNIT_DRIVER_INTEGRATION_MINI_JSON_H_
