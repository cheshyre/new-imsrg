// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "tests/unit/driver/integration/mini_json.h"

#include <cstddef>
#include <cstdlib>
#include <string>
#include <utility>

namespace imsrg::it::detail {

std::string JsonParser::StringLiteral() {
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

Json JsonParser::value() {
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

}  // namespace imsrg::it::detail
