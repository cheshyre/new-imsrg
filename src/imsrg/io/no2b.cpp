#include "imsrg/io/no2b.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <fstream>
#include <string>
#include <system_error>
#include <type_traits>
#include <vector>

#include "imsrg/base/error.hpp"
#include "imsrg/base/half.hpp"
#include "imsrg/base/types.hpp"
#include "imsrg/io/input_stream.hpp"

namespace imsrg::io {

namespace {

static_assert(std::endian::native == std::endian::little,
              "binary NO2B files are little-endian; a byte swap is not implemented");

constexpr double kStoreCut = 1e-8;
constexpr std::size_t kChunkBytes = std::size_t{8} << 20;

struct FileOrbit {
  std::int32_t n, l, j, e;
};

No2bFileTruncation resolved(const No2bFileTruncation& f) {
  if (f.e1max < 0 || f.e2max < 0 || f.e3max < 0) {
    throw InputError("NO2B file truncations must be non-negative (file3e1max=" +
                     std::to_string(f.e1max) + ", file3e2max=" + std::to_string(f.e2max) +
                     ", file3e3max=" + std::to_string(f.e3max) + ")");
  }
  No2bFileTruncation r = f;
  if (r.lmax < 0) r.lmax = r.e1max;
  return r;
}

std::vector<FileOrbit> file_orbits(const No2bFileTruncation& f) {
  std::vector<FileOrbit> orbs;
  for (std::int32_t e = 0; e <= f.e1max; ++e) {
    for (std::int32_t l = e % 2; l <= std::min(e, f.lmax); l += 2) {
      for (std::int32_t j = std::abs(2 * l - 1); j <= 2 * l + 1; j += 2) {
        orbs.push_back({(e - l) / 2, l, j, e});
      }
    }
  }
  return orbs;
}

/// Calls `block(i1..i6, jmin, jmax)` for every orbit sextuple of the file with a non-empty J range,
/// in file order. `emax_first` cuts the first orbit (a break, as the orbits are ordered by e).
template <class Block>
void walk_blocks(const std::vector<FileOrbit>& orbs, const No2bFileTruncation& f,
                 std::int32_t emax_first, Block&& block) {
  const std::size_t norb = orbs.size();
  for (std::size_t i1 = 0; i1 < norb; ++i1) {
    const FileOrbit& o1 = orbs[i1];
    if (o1.e > emax_first) break;
    for (std::size_t i2 = 0; i2 <= i1; ++i2) {
      const FileOrbit& o2 = orbs[i2];
      if (o1.e + o2.e > f.e2max) continue;
      for (std::size_t i3 = 0; i3 < norb; ++i3) {
        const FileOrbit& o3 = orbs[i3];
        if (o2.e + o3.e > f.e2max || o1.e + o3.e > f.e2max) continue;
        if (o1.e + o2.e + o3.e > f.e3max) continue;
        for (std::size_t i4 = 0; i4 <= i1; ++i4) {
          const FileOrbit& o4 = orbs[i4];
          for (std::size_t i5 = 0; i5 <= i4; ++i5) {
            const FileOrbit& o5 = orbs[i5];
            if (o4.e + o5.e > f.e2max) continue;
            for (std::size_t i6 = 0; i6 < norb; ++i6) {
              const FileOrbit& o6 = orbs[i6];
              if (o6.j != o3.j || o6.l != o3.l) continue;
              if (o4.e + o6.e > f.e2max || o5.e + o6.e > f.e2max) continue;
              if (o4.e + o5.e + o6.e > f.e3max) continue;
              if ((o1.l + o2.l + o3.l) % 2 != (o4.l + o5.l + o6.l) % 2) continue;
              const std::int32_t jmin = std::max(std::abs(o1.j - o2.j), std::abs(o4.j - o5.j)) / 2;
              const std::int32_t jmax = std::min(o1.j + o2.j, o4.j + o5.j) / 2;
              if (jmin > jmax) continue;
              block(i1, i2, i3, i4, i5, i6, jmin, jmax);
            }
          }
        }
      }
    }
  }
}

/// Value source over a binary file read in large chunks.
template <class Value>
class BinarySource {
 public:
  BinarySource(const std::filesystem::path& path, std::int64_t total)
      : in_(path, std::ios::binary), path_(path), remaining_(total) {
    if (!in_) throw InputError("cannot open NO2B file " + path.string());
    buf_.resize(kChunkBytes / sizeof(Value));
  }
  float next() {
    if (pos_ == len_) refill();
    return convert(buf_[pos_++]);
  }
  void skip(std::int64_t n) {
    while (n > 0) {
      if (pos_ == len_) refill();
      const std::int64_t k = std::min<std::int64_t>(n, static_cast<std::int64_t>(len_ - pos_));
      pos_ += static_cast<std::size_t>(k);
      n -= k;
    }
  }

 private:
  static float convert(Value v) {
    if constexpr (std::is_same_v<Value, float>) {
      return v;
    } else {
      return half_to_float(v);
    }
  }
  void refill() {
    const std::size_t want = static_cast<std::size_t>(
        std::min<std::int64_t>(remaining_, static_cast<std::int64_t>(buf_.size())));
    if (want == 0) throw InputError("NO2B file " + path_.string() + ": data ended early");
    in_.read(reinterpret_cast<char*>(buf_.data()),
             static_cast<std::streamsize>(want * sizeof(Value)));
    if (static_cast<std::size_t>(in_.gcount()) != want * sizeof(Value)) {
      throw InputError("NO2B file " + path_.string() + ": read error or file shrank");
    }
    pos_ = 0;
    len_ = want;
    remaining_ -= static_cast<std::int64_t>(want);
  }
  std::ifstream in_;
  std::filesystem::path path_;
  std::int64_t remaining_;
  std::vector<Value> buf_;
  std::size_t pos_ = 0, len_ = 0;
};

/// Value source over a text file (header line skipped).
class TextSource {
 public:
  explicit TextSource(const std::filesystem::path& path) : in_(path) { in_.skip_line(); }
  float next() {
    float v = 0;
    if (!in_.next_float(v)) {
      throw InputError("NO2B file " + in_.path().string() +
                       ": data ended before all values were read (check file3e1max, "
                       "file3e2max, file3e3max)");
    }
    return v;
  }
  void skip(std::int64_t n) {
    for (; n > 0; --n) next();
  }

 private:
  TextInput in_;
};

template <class Source>
void fill(Source& src, const std::vector<FileOrbit>& orbs, const No2bFileTruncation& f,
          op::No2bThreeBody& v3, No2bReadReport& rep) {
  const op::No2bTruncation tr = v3.truncation();
  // Container index of each file orbit (-1: not stored).
  std::vector<std::int32_t> cidx(orbs.size(), -1);
  for (std::size_t i = 0; i < orbs.size(); ++i) {
    if (orbs[i].e > tr.emax) continue;
    if (auto k = v3.isospin_orbit_index(orbs[i].n, orbs[i].l, TwoJ{orbs[i].j})) cidx[i] = *k;
  }
  walk_blocks(orbs, f, tr.emax,
              [&](std::size_t i1, std::size_t i2, std::size_t i3, std::size_t i4, std::size_t i5,
                  std::size_t i6, std::int32_t jmin, std::int32_t jmax) {
                const std::int64_t n = static_cast<std::int64_t>(jmax + 1 - jmin) * 5;
                rep.values_read += n;
                const std::int32_t e1 = orbs[i1].e, e2 = orbs[i2].e, e3 = orbs[i3].e,
                                   e4 = orbs[i4].e, e5 = orbs[i5].e, e6 = orbs[i6].e;
                const bool wanted =
                    cidx[i1] >= 0 && cidx[i2] >= 0 && cidx[i3] >= 0 && cidx[i4] >= 0 &&
                    cidx[i5] >= 0 && cidx[i6] >= 0 && e1 + e2 <= tr.e2max && e1 + e3 <= tr.e2max &&
                    e2 + e3 <= tr.e2max && e4 + e5 <= tr.e2max && e4 + e6 <= tr.e2max &&
                    e5 + e6 <= tr.e2max && e1 + e2 + e3 <= tr.e3max && e4 + e5 + e6 <= tr.e3max;
                if (!wanted) {
                  src.skip(n);
                  return;
                }
                for (std::int32_t J = jmin; J <= jmax; ++J) {
                  for (std::int32_t T12 : {0, 1}) {
                    for (std::int32_t T45 : {0, 1}) {
                      for (std::int32_t T3 = 1; T3 <= std::min(2 * T12 + 1, 2 * T45 + 1); T3 += 2) {
                        const float v = src.next();
                        if (i1 == i2 && (J + T12) % 2 == 0) continue;  // Pauli
                        if (i4 == i5 && (J + T45) % 2 == 0) continue;
                        if (std::abs(static_cast<double>(v)) > kStoreCut) {
                          v3.set_isospin(cidx[i1], cidx[i2], cidx[i3], T12, cidx[i4], cidx[i5],
                                         cidx[i6], T45, J, T3, static_cast<double>(v));
                          ++rep.values_stored;
                        }
                      }
                    }
                  }
                }
              });
}

}  // namespace

std::int64_t no2b_value_count(const No2bFileTruncation& file) {
  const No2bFileTruncation f = resolved(file);
  const auto orbs = file_orbits(f);
  std::int64_t count = 0;
  walk_blocks(orbs, f, f.e1max,
              [&](std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t,
                  std::int32_t jmin,
                  std::int32_t jmax) { count += static_cast<std::int64_t>(jmax + 1 - jmin) * 5; });
  return count;
}

void read_no2b(const std::filesystem::path& path, const No2bReadOptions& options,
               op::No2bThreeBody& v3, No2bReadReport* report) {
  const auto t0 = std::chrono::steady_clock::now();
  const No2bFileTruncation f = resolved(options.file);
  const auto orbs = file_orbits(f);
  No2bReadReport rep;
  const bool binary = path.filename().string().find("stream.bin") != std::string::npos;
  if (binary) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec) throw InputError("cannot open NO2B file " + path.string() + ": " + ec.message());
    const std::int64_t count = no2b_value_count(f);
    const bool half = options.precision == No2bPrecision::Half;
    const std::uint64_t expected =
        static_cast<std::uint64_t>(count) * (half ? sizeof(std::uint16_t) : sizeof(float));
    if (size != expected) {
      throw InputError(
          "NO2B file " + path.string() + " has " + std::to_string(size) + " bytes but " +
          std::to_string(expected) + " are expected (" + std::to_string(count) + " " +
          (half ? "half" : "single") +
          "-precision values) for file3e1max=" + std::to_string(f.e1max) +
          ", file3e2max=" + std::to_string(f.e2max) + ", file3e3max=" + std::to_string(f.e3max) +
          ", file3lmax=" + std::to_string(f.lmax) + "; check these truncations and the precision");
    }
    if (half) {
      BinarySource<std::uint16_t> src(path, count);
      fill(src, orbs, f, v3, rep);
    } else {
      BinarySource<float> src(path, count);
      fill(src, orbs, f, v3, rep);
    }
  } else {
    TextSource src(path);
    fill(src, orbs, f, v3, rep);
  }
  rep.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (report != nullptr) *report = rep;
}

}  // namespace imsrg::io
