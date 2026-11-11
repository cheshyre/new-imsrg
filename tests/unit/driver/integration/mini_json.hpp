#pragma once
/// @file
/// A minimal JSON reader for the integration tests (objects, arrays, strings, numbers, true,
/// false, null). The driver's `results.json` is machine written with `%.17g` numbers, so no
/// escapes beyond the common ones and no unicode handling are needed.

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
  enum class Kind { Null, Bool, Number, String, Array, Object };
  Kind kind = Kind::Null;
  bool boolean = false;
  double number = 0.0;
  std::string text;
  std::vector<Json> items;                           ///< array elements
  std::vector<std::pair<std::string, Json>> fields;  ///< object members, in file order

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
  Json parse() {
    Json v = value();
    skip();
    if (pos_ != s_.size()) fail("trailing characters");
    return v;
  }

 private:
  const std::string& s_;
  std::size_t pos_ = 0;

  [[noreturn]] void fail(const std::string& what) const {
    throw std::runtime_error("JSON parse error at offset " + std::to_string(pos_) + ": " + what);
  }
  void skip() {
    while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_]))) ++pos_;
  }
  char peek() {
    skip();
    if (pos_ >= s_.size()) fail("unexpected end");
    return s_[pos_];
  }
  void expect(char c) {
    if (peek() != c) fail(std::string("expected '") + c + "'");
    ++pos_;
  }
  bool literal(const char* word) {
    const std::string w(word);
    if (s_.compare(pos_, w.size(), w) == 0) {
      pos_ += w.size();
      return true;
    }
    return false;
  }
  std::string string_literal() {
    expect('"');
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
    expect('"');
    return out;
  }
  Json value() {
    Json v;
    const char c = peek();
    if (c == '{') {
      ++pos_;
      v.kind = Json::Kind::Object;
      if (peek() == '}') {
        ++pos_;
        return v;
      }
      while (true) {
        std::string key = string_literal();
        expect(':');
        v.fields.emplace_back(std::move(key), value());
        if (peek() == ',') {
          ++pos_;
          continue;
        }
        expect('}');
        return v;
      }
    }
    if (c == '[') {
      ++pos_;
      v.kind = Json::Kind::Array;
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
        expect(']');
        return v;
      }
    }
    if (c == '"') {
      v.kind = Json::Kind::String;
      v.text = string_literal();
      return v;
    }
    if (literal("true")) {
      v.kind = Json::Kind::Bool;
      v.boolean = true;
      return v;
    }
    if (literal("false")) {
      v.kind = Json::Kind::Bool;
      return v;
    }
    if (literal("null")) return v;
    const char* begin = s_.c_str() + pos_;
    char* end = nullptr;
    v.number = std::strtod(begin, &end);
    if (end == begin) fail("bad value");
    v.kind = Json::Kind::Number;
    pos_ += static_cast<std::size_t>(end - begin);
    return v;
  }
};

}  // namespace detail

/// Parses `text`. @throws std::runtime_error on malformed input.
inline Json parse_json(const std::string& text) { return detail::JsonParser(text).parse(); }

}  // namespace imsrg::it
