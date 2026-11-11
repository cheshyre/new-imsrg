#include "imsrg/io/input_stream.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <utility>
#include <version>

#include <zlib.h>

#include "imsrg/base/error.hpp"

namespace imsrg::io {

namespace {

constexpr std::size_t kBufferSize = std::size_t{1} << 16;

bool is_space(char c) noexcept {
  return c == ' ' || c == '\n' || c == '\t' || c == '\r' || c == '\f' || c == '\v';
}

/// Parses `s` (the whole string) as a float. Returns false if it is not fully a number.
bool parse_float(const std::string& s, float& out) {
#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
  const char* first = s.data();
  const char* last = s.data() + s.size();
  if (first != last && *first == '+') ++first;  // from_chars rejects a leading '+'
  float v = 0.0F;
  const auto r = std::from_chars(first, last, v);
  if (r.ec != std::errc{} || r.ptr != last) return false;
  out = v;
  return true;
#else
  // s is a std::string, hence NUL-terminated.
  char* end = nullptr;
  const float v = std::strtof(s.c_str(), &end);
  if (end != s.c_str() + s.size() || end == s.c_str()) return false;
  out = v;
  return true;
#endif
}

}  // namespace

struct TextInput::Impl {
  std::filesystem::path path;
  gzFile file = nullptr;
  std::array<char, kBufferSize> buf{};
  std::size_t pos = 0;
  std::size_t len = 0;
  bool eof = false;
  std::string scratch;

  Impl() = default;
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  ~Impl() {
    if (file != nullptr) gzclose(file);
  }

  /// Ensures at least one unread byte is buffered. Returns false at the end of data.
  bool fill() {
    if (pos < len) return true;
    if (eof) return false;
    const int n = gzread(file, buf.data(), static_cast<unsigned>(buf.size()));
    if (n < 0) {
      int errnum = 0;
      const char* msg = gzerror(file, &errnum);
      throw InputError("read error in '" + path.string() + "': " + (msg != nullptr ? msg : "?"));
    }
    pos = 0;
    len = static_cast<std::size_t>(n);
    if (n == 0) eof = true;
    return n > 0;
  }

  bool read_token(std::string& tok) {
    tok.clear();
    for (;;) {  // skip leading whitespace
      if (!fill()) return false;
      if (!is_space(buf[pos])) break;
      ++pos;
    }
    while (fill() && !is_space(buf[pos])) tok.push_back(buf[pos++]);
    return true;
  }
};

TextInput::TextInput(const std::filesystem::path& path) : impl_(std::make_unique<Impl>()) {
  impl_->path = path;
  impl_->file = gzopen(path.c_str(), "rb");
  if (impl_->file == nullptr) throw InputError("cannot open input file '" + path.string() + "'");
  gzbuffer(impl_->file, static_cast<unsigned>(2 * kBufferSize));
}

TextInput::~TextInput() = default;
TextInput::TextInput(TextInput&&) noexcept = default;
TextInput& TextInput::operator=(TextInput&&) noexcept = default;

const std::filesystem::path& TextInput::path() const noexcept { return impl_->path; }

bool TextInput::skip_line() {
  Impl& s = *impl_;
  bool any = false;
  while (s.fill()) {
    any = true;
    const char c = s.buf[s.pos++];
    if (c == '\n') break;
  }
  return any;
}

bool TextInput::next_token(std::string& token) { return impl_->read_token(token); }

bool TextInput::next_float(float& value) {
  Impl& s = *impl_;
  if (!s.read_token(s.scratch)) return false;
  if (!parse_float(s.scratch, value)) {
    throw InputError("'" + s.path.string() + "': malformed numeric token '" + s.scratch + "'");
  }
  return true;
}

struct GzipWriter::Impl {
  std::filesystem::path path;
  gzFile file = nullptr;
  Impl() = default;
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  ~Impl() {
    if (file != nullptr) gzclose(file);
  }
};

GzipWriter::GzipWriter(const std::filesystem::path& path) : impl_(std::make_unique<Impl>()) {
  impl_->path = path;
  impl_->file = gzopen(path.c_str(), "wb");
  if (impl_->file == nullptr) {
    throw InputError("cannot create gzip file '" + path.string() + "'");
  }
}

GzipWriter::~GzipWriter() = default;
GzipWriter::GzipWriter(GzipWriter&&) noexcept = default;
GzipWriter& GzipWriter::operator=(GzipWriter&&) noexcept = default;

void GzipWriter::write(std::string_view text) {
  if (impl_ == nullptr || impl_->file == nullptr) throw InputError("write to a closed GzipWriter");
  std::size_t done = 0;
  while (done < text.size()) {
    const auto chunk = static_cast<unsigned>(std::min<std::size_t>(text.size() - done, 1U << 30));
    const int n = gzwrite(impl_->file, text.data() + done, chunk);
    if (n <= 0) throw InputError("write error in '" + impl_->path.string() + "'");
    done += static_cast<std::size_t>(n);
  }
}

void GzipWriter::close() {
  if (impl_ == nullptr || impl_->file == nullptr) return;
  const int rc = gzclose(impl_->file);
  impl_->file = nullptr;
  if (rc != Z_OK) throw InputError("error closing '" + impl_->path.string() + "'");
}

}  // namespace imsrg::io
