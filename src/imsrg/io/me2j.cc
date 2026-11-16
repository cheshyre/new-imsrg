// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/io/me2j.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "imsrg/base/error.h"
#include "imsrg/base/types.h"
#include "imsrg/io/input_stream.h"

namespace imsrg::io {

namespace {

constexpr double kSqrt2 = 1.4142135623730950488016887242097;

/// A single-particle orbit label of the file (the isospin label is implicit).
struct FileOrbit {
  std::int32_t n;
  std::int32_t l;
  std::int32_t j2;
  std::int32_t e() const noexcept { return 2 * n + l; }
};

/// The truncation with the imsrg++ defaults resolved (a non-positive value
/// selects the default).
struct Resolved {
  std::int32_t e1max;
  std::int32_t e2max;
  std::int32_t lmax;
};

Resolved Resolve(const Me2jTruncation& t) {
  if (t.e1max < 0) throw InputError("me2j: file2e1max must be >= 0");
  return {t.e1max, t.e2max > 0 ? t.e2max : 2 * t.e1max,
          t.lmax > 0 ? t.lmax : t.e1max};
}

/// The file's orbit list for shells e <= emax: e ascending, then l ascending,
/// then 2j ascending.
std::vector<FileOrbit> FileOrbits(std::int32_t emax, std::int32_t lmax) {
  std::vector<FileOrbit> out;
  for (std::int32_t e = 0; e <= emax; ++e) {
    for (std::int32_t l = e % 2; l <= std::min(e, lmax); l += 2) {
      const std::int32_t n = (e - l) / 2;
      for (std::int32_t j2 = std::abs(2 * l - 1); j2 <= 2 * l + 1; j2 += 2) {
        out.push_back({n, l, j2});
      }
    }
  }
  return out;
}

/// Calls `f(nlj1, nlj2, nlj3, nlj4, J)` for every (abcd, J) record of the file
/// prefix with shells up to `emax_eff`, in file order (the loop structure of
/// imsrg++ ReadBareTBME_Darmstadt).
template <class F>
void ForEachRecord(const std::vector<FileOrbit>& orbs, std::int32_t e2max,
                   F&& f) {
  const std::size_t count = orbs.size();
  for (std::size_t i1 = 0; i1 < count; ++i1) {
    const FileOrbit& o1 = orbs[i1];
    for (std::size_t i2 = 0; i2 <= i1; ++i2) {
      const FileOrbit& o2 = orbs[i2];
      if (o1.e() + o2.e() > e2max) break;
      for (std::size_t i3 = 0; i3 <= i1; ++i3) {
        const FileOrbit& o3 = orbs[i3];
        const std::size_t i4max = (i3 == i1) ? i2 : i3;
        for (std::size_t i4 = 0; i4 <= i4max; ++i4) {
          const FileOrbit& o4 = orbs[i4];
          if (o3.e() + o4.e() > e2max) break;
          if ((o1.l + o2.l + o3.l + o4.l) % 2 != 0) continue;
          const std::int32_t jmin =
              std::max(std::abs(o1.j2 - o2.j2), std::abs(o3.j2 - o4.j2)) / 2;
          const std::int32_t jmax = std::min(o1.j2 + o2.j2, o3.j2 + o4.j2) / 2;
          for (std::int32_t J = jmin; J <= jmax; ++J) f(i1, i2, i3, i4, J);
        }
      }
    }
  }
}

std::int32_t EffectiveEmax(const Resolved& r, std::int32_t emax_read) {
  return (emax_read < 0) ? r.e1max : std::min(emax_read, r.e1max);
}

std::int64_t CountValues(const Resolved& r, std::int32_t emax_read) {
  const auto orbs = FileOrbits(EffectiveEmax(r, emax_read), r.lmax);
  std::int64_t records = 0;
  ForEachRecord(orbs, r.e2max,
                [&](std::size_t, std::size_t, std::size_t, std::size_t,
                    std::int32_t) { ++records; });
  return 4 * records;
}

/// Proton and neutron model-space orbits of a file orbit (both absent if not in
/// the space).
struct PnOrbit {
  OrbitIndex p;
  OrbitIndex n;
  bool present = false;
};

PnOrbit Locate(const basis::SingleParticleBasis& b, const FileOrbit& o) {
  const basis::Orbit proton{o.n, o.l, TwoJ{o.j2}, TwoTz{-1}};
  const basis::Orbit neutron{o.n, o.l, TwoJ{o.j2}, TwoTz{1}};
  const auto p = b.find(proton);
  const auto n = b.find(neutron);
  if (!p || !n) return {};
  return {*p, *n, true};
}

/// The four values of one file record, in file order, widened from float.
struct RecordValues {
  double v00;  // T = 0 pn
  double vnn;
  double v10;  // T = 1 pn
  double vpp;
};

/// Stores one (abcd, J) record in `tb` (the file is unnormalized: identical
/// pairs are normalized in the imsrg++ order of operations); `i1..i4` are the
/// file-orbit indices of `a..d`. Nothing is stored unless all four orbits are
/// present in the model space.
void StoreRecord(op::TwoBody& tb, const PnOrbit& a, const PnOrbit& b,
                 const PnOrbit& c, const PnOrbit& d, std::size_t i1,
                 std::size_t i2, std::size_t i3, std::size_t i4, std::int32_t J,
                 const RecordValues& rv) {
  if (!(a.present && b.present && c.present && d.present)) return;

  double norm = 1.0;
  const bool ab = (i1 == i2);
  const bool cd = (i3 == i4);
  if (ab) norm /= kSqrt2;
  if (cd) norm /= kSqrt2;
  const TwoJ tj{2 * J};

  if (norm > 0.9 || J % 2 == 0) {
    tb.set_normalized(a.p, b.p, c.p, d.p, tj, rv.vpp * norm);
    tb.set_normalized(a.n, b.n, c.n, d.n, tj, rv.vnn * norm);
  }
  // Set_pn_TBME_from_iso for T = 1 (even J for identical pairs) and T = 0.
  const auto add_iso = [&](double value, int phase) {
    double t = value * norm;
    t *= 0.5;
    if (ab) t *= kSqrt2;
    if (cd) t *= kSqrt2;
    tb.AddNormalized(a.p, b.n, c.p, d.n, tj, t);
    if (i3 != i4) tb.AddNormalized(a.p, b.n, c.n, d.p, tj, t * phase);
    if (i1 != i2 && i3 != i4) tb.AddNormalized(a.n, b.p, c.n, d.p, tj, t);
    if (i1 != i2 && (i1 != i3 || i2 != i4)) {
      tb.AddNormalized(a.n, b.p, c.p, d.n, tj, t * phase);
    }
  };
  if (norm > 0.9 || J % 2 == 0) add_iso(rv.v10, +1);
  if (norm > 0.9 || J % 2 != 0) add_iso(rv.v00, -1);
}

/// Checks that nothing beyond the `expected` values is left in `in` (`read`
/// were consumed): otherwise the truncation does not describe the file.
void CheckNoExtraValues(TextInput& in, const std::filesystem::path& path,
                        std::int64_t read, std::int64_t expected) {
  float extra = 0.0F;
  std::int64_t found = read;
  while (in.NextFloat(extra)) ++found;
  if (found != expected) {
    throw InputError(
        "'" + path.string() + "': found " + std::to_string(found) +
        " values but expected " + std::to_string(expected) +
        " for the given file truncation; check file2e1max, file2e2max and "
        "file2lmax");
  }
}

/// Size in bytes of the uninterpreted header of a `.bin` file (D57; imsrg++
/// `BINARYHEADERSIZE`, `ReadWrite.cc:30`).
constexpr std::int64_t kBinaryHeaderBytes = 255;
/// Floats per read of a `.bin` file (4 MiB chunks; the file is never held
/// whole).
constexpr std::size_t kBinaryChunkFloats = std::size_t{1} << 20;

static_assert(sizeof(float) == 4, "me2j .bin files hold 4-byte floats");
static_assert(std::endian::native == std::endian::little,
              "me2j .bin files are little-endian; byte swapping is not "
              "implemented");

/// True if `path` names a binary me2j file (suffix `.bin`).
bool IsBinaryPath(const std::filesystem::path& path) {
  return path.extension() == ".bin";
}

/// A chunked reader of the binary32 values of a `.bin` file, after its
/// 255-byte header. Every read is checked; failures throw `InputError` naming
/// the file and the byte offset.
class BinaryInput {
 public:
  /// Opens `path` and checks that its size is exactly 255 + 4 *
  /// `expected_values` (64-bit arithmetic), then skips the header.
  /// @throws InputError if the file cannot be opened or has another size.
  BinaryInput(const std::filesystem::path& path, std::int64_t expected_values)
      : path_(path), buffer_(kBinaryChunkFloats) {
    std::error_code ec;
    const std::uintmax_t size = std::filesystem::file_size(path, ec);
    if (ec) {
      throw InputError(
          "'" + path.string() +
          "': cannot determine the size of the binary file: " + ec.message());
    }
    const std::int64_t expected_size = kBinaryHeaderBytes + 4 * expected_values;
    if (size != static_cast<std::uintmax_t>(expected_size)) {
      throw InputError(
          "'" + path.string() + "': binary me2j file has " +
          std::to_string(size) + " bytes but " + std::to_string(expected_size) +
          " (= 255 header bytes + 4 * " + std::to_string(expected_values) +
          " values) are expected for the given file truncation; check "
          "file2e1max, file2e2max and file2lmax");
    }
    in_.open(path, std::ios::binary);
    if (!in_) throw InputError("'" + path.string() + "': cannot open file");
    in_.seekg(kBinaryHeaderBytes);
    if (!in_) {
      throw InputError("'" + path.string() +
                       "': cannot skip the 255-byte header");
    }
    offset_ = kBinaryHeaderBytes;
    total_floats_ = expected_values;
  }

  /// The next value widened to double. @throws InputError on a short read.
  double Next() {
    if (pos_ == have_) Refill();
    return static_cast<double>(buffer_[pos_++]);
  }

 private:
  void Refill() {
    const std::int64_t remaining =
        total_floats_ - (offset_ - kBinaryHeaderBytes) / 4;
    const std::size_t want = static_cast<std::size_t>(
        std::min<std::int64_t>(remaining, std::int64_t(kBinaryChunkFloats)));
    if (want == 0) {
      throw InputError("'" + path_.string() +
                       "': read past the end at offset " +
                       std::to_string(offset_));
    }
    in_.read(reinterpret_cast<char*>(buffer_.data()),
             static_cast<std::streamsize>(4 * want));
    if (!in_ || static_cast<std::size_t>(in_.gcount()) != 4 * want) {
      throw InputError(
          "'" + path_.string() + "': short or failed read at offset " +
          std::to_string(offset_) + " (wanted " + std::to_string(4 * want) +
          " bytes, got " + std::to_string(in_.gcount()) + ")");
    }
    offset_ += static_cast<std::int64_t>(4 * want);
    have_ = want;
    pos_ = 0;
  }

  std::filesystem::path path_;
  std::ifstream in_;
  std::vector<float> buffer_;
  std::size_t pos_ = 0;
  std::size_t have_ = 0;
  std::int64_t offset_ = 0;
  std::int64_t total_floats_ = 0;
};

/// Reads every record of the file prefix into `tb` and returns the number of
/// values consumed; `next_value()` returns the next value (widened) and throws
/// at the end of the data.
template <class NextValue>
std::int64_t ReadRecordsFrom(NextValue&& next_value, const Resolved& r,
                             const std::vector<FileOrbit>& orbs,
                             const std::vector<PnOrbit>& pn, op::TwoBody& tb) {
  std::int64_t read = 0;
  const auto next = [&]() {
    const double v = next_value();
    ++read;
    return v;
  };
  ForEachRecord(orbs, r.e2max,
                [&](std::size_t i1, std::size_t i2, std::size_t i3,
                    std::size_t i4, std::int32_t J) {
                  // File order: T=0 pn, nn, T=1 pn, pp (widened from float).
                  RecordValues rv{};
                  rv.v00 = next();
                  rv.vnn = next();
                  rv.v10 = next();
                  rv.vpp = next();
                  StoreRecord(tb, pn[i1], pn[i2], pn[i3], pn[i4], i1, i2, i3,
                              i4, J, rv);
                });
  return read;
}

/// Reads the records of a text/gzip file; `expected` is only used for the
/// truncation message.
std::int64_t ReadRecords(TextInput& in, const std::filesystem::path& path,
                         const Resolved& r, const std::vector<FileOrbit>& orbs,
                         const std::vector<PnOrbit>& pn, op::TwoBody& tb,
                         std::int64_t expected) {
  std::int64_t consumed = 0;
  const auto next_value = [&]() {
    float v = 0.0F;
    if (!in.NextFloat(v)) {
      throw InputError("'" + path.string() + "': data ended after " +
                       std::to_string(consumed) + " values; " +
                       std::to_string(expected) +
                       " are needed for the given file truncation (check "
                       "file2e1max, file2e2max and file2lmax)");
    }
    ++consumed;
    return static_cast<double>(v);
  };
  return ReadRecordsFrom(next_value, r, orbs, pn, tb);
}

}  // namespace

std::int64_t Me2jValueCount(const Me2jTruncation& file,
                            std::int32_t emax_read) {
  return CountValues(Resolve(file), emax_read);
}

op::Operator ReadMe2j(const std::filesystem::path& path,
                      const Me2jReadOptions& options,
                      std::shared_ptr<const basis::ModelSpace> ms,
                      Me2jReadReport* report) {
  const auto t0 = std::chrono::steady_clock::now();
  if (!ms) throw InputError("read_me2j: null model space");
  const Resolved r = Resolve(options.file);
  const basis::SingleParticleBasis& sp = ms->basis();

  // Prefix of the file the model space needs (the whole file when verifying
  // it).
  const std::int32_t emax_read = options.read_whole_file ? r.e1max : sp.emax();
  const auto orbs = FileOrbits(EffectiveEmax(r, emax_read), r.lmax);
  std::vector<PnOrbit> pn;
  pn.reserve(orbs.size());
  for (const FileOrbit& o : orbs) pn.push_back(Locate(sp, o));

  op::Operator H(ms, op::OperatorSpec::Scalar(2, Hermiticity::kHermitian));
  op::TwoBody& tb = H.two_body();

  const std::int64_t expected = CountValues(r, emax_read);
  std::int64_t read = 0;
  if (IsBinaryPath(path)) {
    // The size must match the whole file's truncation, even when only a prefix
    // is read.
    BinaryInput bin(path, CountValues(r, r.e1max));
    read = ReadRecordsFrom([&]() { return bin.Next(); }, r, orbs, pn, tb);
  } else {
    TextInput in(path);
    in.SkipLine();  // header line
    read = ReadRecords(in, path, r, orbs, pn, tb, expected);
    if (options.read_whole_file) {
      CheckNoExtraValues(in, path, read, expected);
    }
  }

  if (report != nullptr) {
    report->values_read = read;
    report->seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
  }
  H.SyncToDevice();  // the elements were set through the host mirror (no-op
                     // on host backends)
  return H;
}

}  // namespace imsrg::io
