// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Native float64 operator files (design/19 section 3.2, design/50 C6).
#include "imsrg/io/operator_binary.h"

#include <unistd.h>

#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/operator/random_operator.h"

namespace imsrg::io {
namespace {

namespace fs = std::filesystem;

std::shared_ptr<const basis::ModelSpace> Ms(int emax, const char* ref = "O16",
                                            const char* val = "O16") {
  return basis::ModelSpace::FromNames(basis::BasisTruncation{.emax = emax},
                                      16.0, ref, val);
}

fs::path TmpDir() {
  const fs::path d = fs::temp_directory_path() /
                     ("imsrg_opbin_" + std::to_string(::getpid()));
  fs::create_directories(d);
  return d;
}

void ExpectBitIdentical(const op::Operator& a, const op::Operator& b) {
  EXPECT_EQ(a.spec(), b.spec());
  EXPECT_EQ(a.hermiticity(), b.hermiticity());
  EXPECT_EQ(std::bit_cast<std::uint64_t>(a.zero_body()),
            std::bit_cast<std::uint64_t>(b.zero_body()));
  EXPECT_EQ(a.one_body().buffer().ToVector(), b.one_body().buffer().ToVector());
  EXPECT_EQ(a.two_body().matrix().buffer().ToVector(),
            b.two_body().matrix().buffer().ToVector());
}

TEST(OperatorBinary, RoundTripIsBitExact) {
  const fs::path dir = TmpDir();
  for (int emax : {2, 4}) {
    for (Hermiticity h : {Hermiticity::kHermitian, Hermiticity::kAntiHermitian}) {
      const auto ms = Ms(emax);
      const op::Operator A =
          op::RandomOperator(ms, op::OperatorSpec::Scalar(2, h),
                             {.seed = 17, .zero_body_scale = 3.0});
      const fs::path p = dir / ("op_" + std::to_string(emax) + ".opb");
      WriteOperatorBinary(p, A, "omega 1");
      const op::Operator B = ReadOperatorBinary(p, Ms(emax));
      ExpectBitIdentical(A, B);
      const OperatorFileHeader hd = ReadOperatorBinaryHeader(p);
      EXPECT_EQ(hd.label, "omega 1");
      EXPECT_EQ(hd.emax, emax);
      EXPECT_EQ(hd.e2max, 2 * emax);
      EXPECT_EQ(hd.hw, 16.0);
      EXPECT_EQ(hd.spec.hermiticity, h);
      EXPECT_EQ(hd.layout_hash, ModelSpaceLayoutHash(*ms));
    }
  }
  // No temporary file is left behind; overwriting replaces the file.
  for (const auto& e : fs::directory_iterator(dir))
    EXPECT_EQ(e.path().string().find(".tmp."), std::string::npos);
  const auto ms = Ms(2);
  const op::Operator C =
      op::RandomOperator(ms, op::OperatorSpec::Scalar(2), {.seed = 99});
  WriteOperatorBinary(dir / "op_2.opb", C);
  ExpectBitIdentical(C, ReadOperatorBinary(dir / "op_2.opb", ms));
  fs::remove_all(dir);
}

TEST(OperatorBinary, OneBodyOperator) {
  const fs::path dir = TmpDir();
  const auto ms = Ms(3);
  const op::Operator A =
      op::RandomOperator(ms, op::OperatorSpec::Scalar(1), {.seed = 4});
  WriteOperatorBinary(dir / "a.opb", A);
  const op::Operator B = ReadOperatorBinary(dir / "a.opb", ms);
  EXPECT_EQ(B.spec().particle_rank, 1);
  EXPECT_EQ(A.one_body().buffer().ToVector(), B.one_body().buffer().ToVector());
  fs::remove_all(dir);
}

TEST(OperatorBinary, LayoutAndReferenceMismatch) {
  const fs::path dir = TmpDir();
  const op::Operator A =
      op::RandomOperator(Ms(2), op::OperatorSpec::Scalar(2), {.seed = 5});
  WriteOperatorBinary(dir / "a.opb", A);
  EXPECT_THROW(ReadOperatorBinary(dir / "a.opb", Ms(3)), InputError);
  // Same layout, other occupations (He4 reference on the O16 basis).
  const auto he4 = Ms(2, "He4", "O16");
  ASSERT_EQ(ModelSpaceLayoutHash(*he4), ModelSpaceLayoutHash(*Ms(2)));
  EXPECT_THROW(ReadOperatorBinary(dir / "a.opb", he4), InputError);
  const op::Operator B = ReadOperatorBinary(dir / "a.opb", he4,
                                            {.check_reference = false});
  EXPECT_EQ(A.two_body().matrix().buffer().ToVector(),
            B.two_body().matrix().buffer().ToVector());
  fs::remove_all(dir);
}

TEST(OperatorBinary, CorruptAndTruncatedFiles) {
  const fs::path dir = TmpDir();
  const auto ms = Ms(2);
  const op::Operator A =
      op::RandomOperator(ms, op::OperatorSpec::Scalar(2), {.seed = 6});
  const fs::path p = dir / "a.opb";
  WriteOperatorBinary(p, A);
  std::vector<char> bytes(fs::file_size(p));
  {
    std::ifstream in(p, std::ios::binary);
    in.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  auto write = [&](const fs::path& q, std::size_t n) {
    std::ofstream out(q, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(n));
  };
  bytes[bytes.size() / 2] ^= 0x10;  // one payload bit
  write(dir / "corrupt.opb", bytes.size());
  EXPECT_THROW(ReadOperatorBinary(dir / "corrupt.opb", ms), InputError);
  bytes[bytes.size() / 2] ^= 0x10;
  write(dir / "short.opb", bytes.size() - 100);
  EXPECT_THROW(ReadOperatorBinary(dir / "short.opb", ms), InputError);
  write(dir / "tiny.opb", 5);
  EXPECT_THROW(ReadOperatorBinaryHeader(dir / "tiny.opb"), InputError);
  EXPECT_THROW(ReadOperatorBinary(dir / "missing.opb", ms), InputError);
  fs::remove_all(dir);
}

}  // namespace
}  // namespace imsrg::io
