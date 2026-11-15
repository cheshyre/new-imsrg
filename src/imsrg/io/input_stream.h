// Buffered text input over zlib (plain or gzip, detected by content) and a small gzip writer.
#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

namespace imsrg::io {

/// A buffered, whitespace-token reader over a plain or gzip-compressed text file.
///
/// The file is opened with zlib `gzopen`, which reads uncompressed files transparently, so
/// `foo.me2j` and `foo.me2j.gz` are handled identically (detection is by content, not by file
/// name). Reads go through a 64 KiB internal buffer; per-token cost is O(token length).
/// Move-only value type; the destructor closes the file. Not thread-safe.
class TextInput {
 public:
  /// Opens `path` for reading.
  /// @throws InputError if the file does not exist or cannot be opened.
  explicit TextInput(const std::filesystem::path& path);
  ~TextInput();
  TextInput(TextInput&&) noexcept;
  TextInput& operator=(TextInput&&) noexcept;
  TextInput(const TextInput&) = delete;
  TextInput& operator=(const TextInput&) = delete;

  /// The path given at construction (used in error messages).
  const std::filesystem::path& path() const noexcept;

  /// Discards everything up to and including the next '\n' (or to the end of data).
  /// @returns false if no byte was available (end of data before the call), true otherwise.
  /// @throws InputError on a zlib read error (e.g. a corrupt gzip stream).
  bool skip_line();

  /// Reads the next whitespace-delimited token into `token`.
  /// @returns false at the end of data (no further token); `token` is then left empty.
  /// @throws InputError on a zlib read error.
  bool next_token(std::string& token);

  /// Reads the next token as a `float`, correctly rounded directly from the decimal text (never
  /// via double, which could round differently): identical to `std::istream >> float`. Accepts
  /// an optional sign, decimal or scientific notation.
  /// @returns false at the end of data; `value` is then unchanged.
  /// @throws InputError (message names the file and the token) if the token is not a number or
  ///         is not fully consumed by the parse (e.g. "1.5abc").
  bool next_float(float& value);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// A minimal gzip text writer (zlib `gzopen(..., "wb")`), used by tests to produce `.gz` inputs.
/// Move-only; the destructor flushes and closes (errors at close are ignored; call `close()` to
/// have them reported).
class GzipWriter {
 public:
  /// Creates or truncates `path` for gzip-compressed writing.
  /// @throws InputError if the file cannot be created.
  explicit GzipWriter(const std::filesystem::path& path);
  ~GzipWriter();
  GzipWriter(GzipWriter&&) noexcept;
  GzipWriter& operator=(GzipWriter&&) noexcept;
  GzipWriter(const GzipWriter&) = delete;
  GzipWriter& operator=(const GzipWriter&) = delete;

  /// Appends `text` (compressed). @throws InputError on a write error.
  void write(std::string_view text);

  /// Flushes and closes the file; idempotent. @throws InputError if closing fails.
  void close();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace imsrg::io
