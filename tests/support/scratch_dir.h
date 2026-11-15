// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#ifndef TESTS_SUPPORT_SCRATCH_DIR_H_
#define TESTS_SUPPORT_SCRATCH_DIR_H_
/// @file
/// Test helpers for directories: a uniquely named temporary directory (no
/// fixed names in the temp directory: ctest runs test cases in parallel
/// processes) and a scoped change of the working directory.

#include <unistd.h>

#include <atomic>
#include <filesystem>
#include <string>
#include <system_error>

namespace imsrg::test_support {

/// A fresh directory below the canonical temp directory, named by the process
/// id and a counter, removed with its contents on destruction.
class ScratchDir {
 public:
  ScratchDir() {
    static std::atomic<int> counter{0};
    std::error_code ec;
    const std::filesystem::path tmp =
        std::filesystem::canonical(std::filesystem::temp_directory_path(), ec);
    path_ = tmp / ("imsrg_scratch_" + std::to_string(::getpid()) + "_" +
                   std::to_string(counter++));
    std::filesystem::create_directories(path_);
  }
  ~ScratchDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }
  ScratchDir(const ScratchDir&) = delete;
  ScratchDir& operator=(const ScratchDir&) = delete;

  /// The (canonical, absolute) directory.
  const std::filesystem::path& path() const { return path_; }
  /// Creates (and returns the path of) the subdirectory `name`.
  std::filesystem::path Sub(const std::string& name) const {
    std::filesystem::create_directories(path_ / name);
    return path_ / name;
  }

 private:
  std::filesystem::path path_;
};

/// Changes the working directory to `dir` for its lifetime (tests of one
/// executable run sequentially, so this is safe within a test body).
class ScopedCwd {
 public:
  explicit ScopedCwd(const std::filesystem::path& dir)
      : old_(std::filesystem::current_path()) {
    std::filesystem::current_path(dir);
  }
  ~ScopedCwd() {
    std::error_code ec;
    std::filesystem::current_path(old_, ec);
  }
  ScopedCwd(const ScopedCwd&) = delete;
  ScopedCwd& operator=(const ScopedCwd&) = delete;

 private:
  std::filesystem::path old_;
};

}  // namespace imsrg::test_support

#endif  // TESTS_SUPPORT_SCRATCH_DIR_H_
