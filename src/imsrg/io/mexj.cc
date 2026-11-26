// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/io/mexj.h"

#include <unistd.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/base/error.h"
#include "imsrg/io/input_stream.h"

namespace imsrg::io {

namespace {

constexpr double kSqrt2 = 1.4142135623730950488016887242097;
constexpr std::int64_t kBinaryHeaderBytes = 255;
constexpr std::size_t kChunkFloats = std::size_t{1} << 16;

static_assert(sizeof(float) == 4);
static_assert(std::endian::native == std::endian::little,
              "binary ME2JP files are little-endian");

/// Unnormalization factor of a record: sqrt(2) per identical-orbit pair (2.0
/// exactly when both pairs are identical, so that it is its own exact inverse).
double PairFactor(bool pq_identical, bool rs_identical) noexcept {
  if (pq_identical && rs_identical) return 2.0;
  if (pq_identical || rs_identical) return kSqrt2;
  return 1.0;
}

std::int64_t Key(std::int64_t n, std::int32_t p, std::int32_t q, std::int32_t r,
                 std::int32_t s) {
  return ((static_cast<std::int64_t>(p) * n + q) * n + r) * n + s;
}

std::string HermString(Hermiticity h) {
  return h == Hermiticity::kHermitian ? "+1" : "-1";
}

std::string HeaderLine(std::string_view tag, const op::Operator& op,
                       std::string_view name) {
  return std::string(tag) +
         " emax=" + std::to_string(op.model_space().basis().emax()) +
         " herm=" + HermString(op.hermiticity()) + " op=" + std::string(name) +
         " code=imsrg";
}

void CheckWritable(const op::Operator& op) {
  if (op.empty()) throw InputError("mexj writer: empty operator");
  const op::OperatorSpec& s = op.spec();
  if (!s.is_scalar() || !s.is_number_conserving()) {
    throw NotImplementedError(
        "ME1J/ME2JP files hold scalar, number-conserving operators only");
  }
  if (s.particle_rank > 2) {
    throw NotImplementedError("ME1J/ME2JP writer: particle rank 3 (ME3JP)");
  }
  if (s.hermiticity == Hermiticity::kNonHermitian) {
    throw NotImplementedError(
        "ME1J/ME2JP writer: non-Hermitian operators are not representable");
  }
}

/// Map from NLJJT index to model-space orbit (nullopt if absent).
std::vector<std::optional<OrbitIndex>> MapOrbits(
    const std::vector<NljjtOrbit>& orbs, const basis::SingleParticleBasis& b) {
  std::vector<std::optional<OrbitIndex>> out;
  out.reserve(orbs.size());
  for (const NljjtOrbit& o : orbs) {
    out.push_back(b.find(basis::Orbit{o.n, o.l, TwoJ{o.j2}, TwoTz{o.tz2}}));
  }
  return out;
}

/// Writes `path` atomically: `body(ostream)` writes into a temporary that is
/// renamed on success and removed on failure.
template <class Body>
void AtomicWrite(const std::filesystem::path& path, Body&& body) {
  std::filesystem::path tmp = path;
  tmp += ".tmp" + std::to_string(::getpid());
  try {
    {
      std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
      if (!out) {
        throw InputError("'" + path.string() + "': cannot create file");
      }
      body(out);
      out.flush();
      if (!out) {
        throw InputError("'" + path.string() + "': write failed");
      }
    }
    std::filesystem::rename(tmp, path);
  } catch (...) {
    std::error_code ec;
    std::filesystem::remove(tmp, ec);
    throw;
  }
}

/// `%.17g` text sink, five values per line.
class TextSink {
 public:
  explicit TextSink(std::ostream& out) : out_(out) {}
  ~TextSink() = default;
  TextSink(const TextSink&) = delete;
  TextSink& operator=(const TextSink&) = delete;
  void Put(double v) {
    char buf[40];
    const int n = std::snprintf(buf, sizeof buf, "%.17g", v);
    text_.append(buf, static_cast<std::size_t>(n));
    text_.push_back(++count_ % 5 == 0 ? '\n' : ' ');
    if (text_.size() > (1U << 20)) Flush();
  }
  void Finish() {
    if (!text_.empty() && count_ % 5 != 0) text_.back() = '\n';
    Flush();
  }

 private:
  void Flush() {
    out_.write(text_.data(), static_cast<std::streamsize>(text_.size()));
    text_.clear();
  }
  std::ostream& out_;
  std::string text_;
  std::int64_t count_ = 0;
};

bool IsBinary(const std::filesystem::path& p) {
  return p.extension() == ".bin";
}

/// Value of the `key=value` token of a header line, if present.
std::optional<std::string> HeaderToken(const std::string& line,
                                       std::string_view key) {
  const std::string k = std::string(key) + "=";
  std::size_t pos = 0;
  while (pos < line.size()) {
    while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t')) ++pos;
    std::size_t end = pos;
    while (end < line.size() && line[end] != ' ' && line[end] != '\t') ++end;
    if (end - pos >= k.size() && line.compare(pos, k.size(), k) == 0) {
      return line.substr(pos + k.size(), end - pos - k.size());
    }
    pos = end;
  }
  return std::nullopt;
}

/// Cross-checks the informational tokens of a header line.
void CheckHeader(const std::string& line, const std::filesystem::path& path,
                 std::int32_t emax, Hermiticity h) {
  if (const auto e = HeaderToken(line, "emax")) {
    char* end = nullptr;
    const std::int64_t fe = std::strtoll(e->c_str(), &end, 10);
    if (end == e->c_str() || *end != '\0') {
      throw InputError("'" + path.string() + "': bad emax token in header");
    }
    if (fe < emax) {
      throw InputError("'" + path.string() + "': file emax " +
                       std::to_string(fe) + " is below the model space emax " +
                       std::to_string(emax));
    }
  }
  if (const auto hv = HeaderToken(line, "herm")) {
    if (*hv != HermString(h)) {
      throw InputError("'" + path.string() + "': header herm=" + *hv +
                       " does not match the requested hermiticity");
    }
  }
}

/// Reads the first line of a text file.
std::string FirstLine(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw InputError("'" + path.string() + "': cannot open file");
  std::string line;
  std::getline(in, line);
  return line;
}

/// Source of values: text tokens or binary32 floats, one `Next()` per value.
class ValueSource {
 public:
  /// Text file: the header line must contain `tag`.
  static ValueSource Text(const std::filesystem::path& path,
                          std::string_view tag, std::int32_t emax,
                          Hermiticity h) {
    const std::string line = FirstLine(path);
    if (line.find(tag) == std::string::npos) {
      throw InputError("'" + path.string() + "': header line lacks the tag '" +
                       std::string(tag) + "'");
    }
    CheckHeader(line, path, emax, h);
    ValueSource s(path);
    s.text_ = std::make_unique<TextInput>(path);
    s.text_->SkipLine();
    return s;
  }

  /// Binary file with at least `expected` values.
  static ValueSource Binary(const std::filesystem::path& path,
                            std::string_view tag, std::int32_t emax,
                            Hermiticity h, std::int64_t expected) {
    std::error_code ec;
    const std::uintmax_t size = std::filesystem::file_size(path, ec);
    if (ec) {
      throw InputError("'" + path.string() +
                       "': cannot determine the file size: " + ec.message());
    }
    if (size < static_cast<std::uintmax_t>(kBinaryHeaderBytes) ||
        (size - kBinaryHeaderBytes) % 4 != 0 ||
        static_cast<std::int64_t>((size - kBinaryHeaderBytes) / 4) < expected) {
      throw InputError(
          "'" + path.string() + "': binary file has " + std::to_string(size) +
          " bytes, expected 255 + 4n with n >= " + std::to_string(expected));
    }
    ValueSource s(path);
    s.bin_.open(path, std::ios::binary);
    if (!s.bin_) throw InputError("'" + path.string() + "': cannot open file");
    std::string header(kBinaryHeaderBytes, ' ');
    s.bin_.read(header.data(), kBinaryHeaderBytes);
    if (!s.bin_) {
      throw InputError("'" + path.string() + "': cannot read the header");
    }
    for (const char* other : {"me1j", "me3jp"}) {
      if (std::string_view(other) != tag &&
          header.find(other) != std::string::npos) {
        throw InputError("'" + path.string() + "': header has the tag '" +
                         other + "', expected '" + std::string(tag) + "'");
      }
    }
    if (header.find(tag) != std::string::npos) {
      CheckHeader(header, path, emax, h);
    }
    s.offset_ = kBinaryHeaderBytes;
    s.remaining_ = (static_cast<std::int64_t>(size) - kBinaryHeaderBytes) / 4;
    s.buffer_.resize(kChunkFloats);
    return s;
  }

  ValueSource(ValueSource&&) = default;
  ValueSource& operator=(ValueSource&&) = default;
  ValueSource(const ValueSource&) = delete;
  ValueSource& operator=(const ValueSource&) = delete;
  ~ValueSource() = default;

  /// The next value. @throws InputError at the end of data or on a bad token.
  double Next() {
    if (text_) {
      if (!text_->NextToken(token_)) {
        throw InputError("'" + path_.string() + "': data ended after " +
                         std::to_string(count_) + " values");
      }
      char* end = nullptr;
      const double v = std::strtod(token_.c_str(), &end);
      if (end == token_.c_str() || *end != '\0') {
        throw InputError("'" + path_.string() + "': bad number '" + token_ +
                         "' after " + std::to_string(count_) + " values");
      }
      ++count_;
      return v;
    }
    if (pos_ == have_) {
      const auto want = static_cast<std::size_t>(std::min<std::int64_t>(
          remaining_, static_cast<std::int64_t>(kChunkFloats)));
      if (want == 0) {
        throw InputError("'" + path_.string() +
                         "': read past the end at offset " +
                         std::to_string(offset_));
      }
      bin_.read(reinterpret_cast<char*>(buffer_.data()),
                static_cast<std::streamsize>(4 * want));
      if (!bin_ || static_cast<std::size_t>(bin_.gcount()) != 4 * want) {
        throw InputError("'" + path_.string() + "': short read at offset " +
                         std::to_string(offset_));
      }
      offset_ += static_cast<std::int64_t>(4 * want);
      remaining_ -= static_cast<std::int64_t>(want);
      have_ = want;
      pos_ = 0;
    }
    ++count_;
    return static_cast<double>(buffer_[pos_++]);
  }

 private:
  explicit ValueSource(std::filesystem::path p) : path_(std::move(p)) {}
  std::filesystem::path path_;
  std::unique_ptr<TextInput> text_;
  std::string token_;
  std::ifstream bin_;
  std::vector<float> buffer_;
  std::size_t pos_ = 0;
  std::size_t have_ = 0;
  std::int64_t offset_ = 0;
  std::int64_t remaining_ = 0;
  std::int64_t count_ = 0;
};

}  // namespace

std::vector<NljjtOrbit> NljjtOrbits(std::int32_t emax) {
  IMSRG_CHECK(emax >= 0, "NljjtOrbits: emax must be >= 0");
  std::vector<NljjtOrbit> out;
  for (std::int32_t e = 0; e <= emax; ++e) {
    for (std::int32_t l = e % 2; l <= e; l += 2) {
      const std::int32_t n = (e - l) / 2;
      for (std::int32_t j2 = std::max(2 * l - 1, 1); j2 <= 2 * l + 1; j2 += 2) {
        out.push_back({n, l, j2, +1});
        out.push_back({n, l, j2, -1});
      }
    }
  }
  return out;
}

std::int32_t NljjtIndex(std::int32_t n, std::int32_t l, std::int32_t j2,
                        std::int32_t tz2) {
  IMSRG_CHECK(n >= 0 && l >= 0 &&
                  (j2 == 2 * l + 1 || (l > 0 && j2 == 2 * l - 1)) &&
                  (tz2 == 1 || tz2 == -1),
              "NljjtIndex: invalid orbit");
  const std::int32_t e = 2 * n + l;
  std::int32_t nlj = e * (e + 1) / 2;  // nlj orbits of the lower shells
  for (std::int32_t lp = e % 2; lp < l; lp += 2) nlj += (lp > 0 ? 2 : 1);
  if (j2 == 2 * l + 1 && l > 0) nlj += 1;
  return 2 * nlj + (tz2 < 0 ? 1 : 0);
}

JRange Me2jpJRange(const NljjtOrbit& p, const NljjtOrbit& q,
                   const NljjtOrbit& r, const NljjtOrbit& s) noexcept {
  if ((p.l + q.l + r.l + s.l) % 2 != 0) return {};
  if (p.tz2 + q.tz2 != r.tz2 + s.tz2) return {};
  const std::int32_t lo =
      std::max(std::abs(p.j2 - q.j2), std::abs(r.j2 - s.j2)) / 2;
  const std::int32_t hi = std::min(p.j2 + q.j2, r.j2 + s.j2) / 2;
  return {lo, hi};
}

std::int64_t Me2jpValueCount(std::int32_t emax) {
  std::int64_t n = 0;
  ForEachMe2jpRecord(emax,
                     [&](std::int32_t, std::int32_t, std::int32_t, std::int32_t,
                         const JRange& r) { n += r.size(); });
  return n;
}

std::int64_t Me1jValueCount(std::int32_t emax) {
  const std::int64_t n = static_cast<std::int64_t>(emax + 1) * (emax + 2);
  return 1 + n * (n + 1) / 2;
}

Me2jpTable::Me2jpTable(std::int32_t emax, Hermiticity h,
                       std::vector<double> values)
    : emax_(emax),
      hermiticity_(h),
      num_orbits_((emax + 1) * (emax + 2)),
      orbits_(NljjtOrbits(emax)),
      values_(std::move(values)) {
  if (h == Hermiticity::kNonHermitian) {
    throw InputError("Me2jpTable: hermiticity must be +1 or -1");
  }
  std::int64_t offset = 0;
  ForEachMe2jpRecord(emax, [&](std::int32_t p, std::int32_t q, std::int32_t r,
                               std::int32_t s, const JRange& range) {
    blocks_.emplace(Key(num_orbits_, p, q, r, s), Block{offset, range});
    offset += range.size();
  });
  if (offset != static_cast<std::int64_t>(values_.size())) {
    throw InputError("Me2jpTable: " + std::to_string(values_.size()) +
                     " values given, " + std::to_string(offset) +
                     " expected for emax " + std::to_string(emax));
  }
}

std::optional<Me2jpTable::Block> Me2jpTable::FindBlock(std::int32_t p,
                                                       std::int32_t q,
                                                       std::int32_t r,
                                                       std::int32_t s) const {
  const auto it = blocks_.find(Key(num_orbits_, p, q, r, s));
  if (it == blocks_.end()) return std::nullopt;
  return it->second;
}

double Me2jpTable::Get(std::int32_t p, std::int32_t q, std::int32_t r,
                       std::int32_t s, std::int32_t J) const {
  for (const std::int32_t i : {p, q, r, s}) {
    IMSRG_CHECK(i >= 0 && i < num_orbits_, "Me2jpTable::Get: orbit index");
  }
  if (J < 0) return 0.0;
  const auto swap_phase = [&](std::int32_t a, std::int32_t b) {
    const std::int32_t k = (orbits_[static_cast<std::size_t>(a)].j2 +
                            orbits_[static_cast<std::size_t>(b)].j2) /
                               2 -
                           J;
    return (k & 1) != 0 ? 1.0 : -1.0;  // -(-1)^k
  };
  double phase = 1.0;
  if (p < q) {
    phase *= swap_phase(p, q);
    std::swap(p, q);
  }
  if (r < s) {
    phase *= swap_phase(r, s);
    std::swap(r, s);
  }
  if (r > p || (r == p && s > q)) {
    std::swap(p, r);
    std::swap(q, s);
    phase *= static_cast<double>(op::HermiticitySign(hermiticity_));
  }
  const auto block = FindBlock(p, q, r, s);
  if (!block || !block->range.contains(J)) return 0.0;
  const std::int64_t index = block->offset + (J - block->range.jmin);
  IMSRG_CHECK(J - block->range.jmin < block->range.size() && index >= 0 &&
                  index < static_cast<std::int64_t>(values_.size()),
              "Me2jpTable::Get: offset outside the record");
  return phase * values_[static_cast<std::size_t>(index)];
}

void WriteMe1j(const std::filesystem::path& path, const op::Operator& op,
               std::string_view op_name) {
  CheckWritable(op);
  op.SyncToHost();
  const std::int32_t emax = op.model_space().basis().emax();
  const auto orbs = NljjtOrbits(emax);
  const auto map = MapOrbits(orbs, op.model_space().basis());
  AtomicWrite(path, [&](std::ostream& out) {
    out << HeaderLine("me1j", op, op_name) << '\n';
    TextSink sink(out);
    sink.Put(op.zero_body());
    for (std::size_t p = 0; p < orbs.size(); ++p) {
      for (std::size_t q = 0; q <= p; ++q) {
        sink.Put(map[p] && map[q] ? op.one_body().get(*map[p], *map[q]) : 0.0);
      }
    }
    sink.Finish();
  });
}

void WriteMe2jp(const std::filesystem::path& path, const op::Operator& op,
                std::string_view op_name) {
  CheckWritable(op);
  op.SyncToHost();
  const std::int32_t emax = op.model_space().basis().emax();
  const auto orbs = NljjtOrbits(emax);
  const auto map = MapOrbits(orbs, op.model_space().basis());
  const bool binary = IsBinary(path);
  const std::string header = HeaderLine("me2jp", op, op_name);
  if (binary && static_cast<std::int64_t>(header.size()) > kBinaryHeaderBytes) {
    throw InputError("WriteMe2jp: operator name too long for the header");
  }
  const op::TwoBody& tb = op.two_body();
  const auto value = [&](std::int32_t p, std::int32_t q, std::int32_t r,
                         std::int32_t s, std::int32_t J) {
    const auto& a = map[static_cast<std::size_t>(p)];
    const auto& b = map[static_cast<std::size_t>(q)];
    const auto& c = map[static_cast<std::size_t>(r)];
    const auto& d = map[static_cast<std::size_t>(s)];
    if (!(a && b && c && d)) return 0.0;
    if (((p == q) || (r == s)) && J % 2 != 0) return 0.0;  // Pauli
    return PairFactor(p == q, r == s) *
           tb.get_normalized(*a, *b, *c, *d, TwoJ{2 * J});
  };
  AtomicWrite(path, [&](std::ostream& out) {
    if (binary) {
      std::string h = header;
      h.resize(kBinaryHeaderBytes, ' ');
      out.write(h.data(), kBinaryHeaderBytes);
      std::vector<float> buf;
      buf.reserve(kChunkFloats);
      const auto flush = [&]() {
        out.write(reinterpret_cast<const char*>(buf.data()),
                  static_cast<std::streamsize>(4 * buf.size()));
        buf.clear();
      };
      ForEachMe2jpRecord(
          emax, [&](std::int32_t p, std::int32_t q, std::int32_t r,
                    std::int32_t s, const JRange& range) {
            for (std::int32_t J = range.jmin; J <= range.jmax; ++J) {
              buf.push_back(static_cast<float>(value(p, q, r, s, J)));
              if (buf.size() == kChunkFloats) flush();
            }
          });
      flush();
    } else {
      out << header << '\n';
      TextSink sink(out);
      ForEachMe2jpRecord(
          emax, [&](std::int32_t p, std::int32_t q, std::int32_t r,
                    std::int32_t s, const JRange& range) {
            for (std::int32_t J = range.jmin; J <= range.jmax; ++J) {
              sink.Put(value(p, q, r, s, J));
            }
          });
      sink.Finish();
    }
  });
}

op::Operator ReadMe1jMe2jp(const std::filesystem::path& me1j_path,
                           const std::filesystem::path& me2jp_path,
                           std::shared_ptr<const basis::ModelSpace> ms,
                           Hermiticity h) {
  if (!ms) throw InputError("ReadMe1jMe2jp: null model space");
  if (h == Hermiticity::kNonHermitian) {
    throw InputError("ReadMe1jMe2jp: hermiticity must be +1 or -1");
  }
  const std::int32_t emax = ms->basis().emax();
  const auto orbs = NljjtOrbits(emax);
  const auto map = MapOrbits(orbs, ms->basis());
  op::Operator O(ms, op::OperatorSpec::Scalar(2, h));

  {
    ValueSource src = ValueSource::Text(me1j_path, "me1j", emax, h);
    O.zero_body() = src.Next();
    for (std::size_t p = 0; p < orbs.size(); ++p) {
      for (std::size_t q = 0; q <= p; ++q) {
        const double v = src.Next();
        if (!(map[p] && map[q])) continue;
        const NljjtOrbit& a = orbs[p];
        const NljjtOrbit& b = orbs[q];
        if (a.l != b.l || a.j2 != b.j2 || a.tz2 != b.tz2) {
          if (v != 0.0) {
            throw InputError("'" + me1j_path.string() +
                             "': nonzero element between orbits that a scalar "
                             "operator does not connect (orbits " +
                             std::to_string(p) + ", " + std::to_string(q) +
                             ")");
          }
          continue;
        }
        O.one_body().set(*map[p], *map[q], v);
      }
    }
  }

  ValueSource src = IsBinary(me2jp_path)
                        ? ValueSource::Binary(me2jp_path, "me2jp", emax, h,
                                              Me2jpValueCount(emax))
                        : ValueSource::Text(me2jp_path, "me2jp", emax, h);
  op::TwoBody& tb = O.two_body();
  ForEachMe2jpRecord(emax, [&](std::int32_t p, std::int32_t q, std::int32_t r,
                               std::int32_t s, const JRange& range) {
    const auto& a = map[static_cast<std::size_t>(p)];
    const auto& b = map[static_cast<std::size_t>(q)];
    const auto& c = map[static_cast<std::size_t>(r)];
    const auto& d = map[static_cast<std::size_t>(s)];
    const double factor = PairFactor(p == q, r == s);
    for (std::int32_t J = range.jmin; J <= range.jmax; ++J) {
      const double v = src.Next();
      if (!(a && b && c && d)) continue;
      if (((p == q) || (r == s)) && J % 2 != 0) {
        if (v != 0.0) {
          throw InputError("'" + me2jp_path.string() +
                           "': nonzero Pauli-forbidden element");
        }
        continue;
      }
      tb.set_normalized(*a, *b, *c, *d, TwoJ{2 * J}, v / factor);
    }
  });
  O.SyncToDevice();
  return O;
}

}  // namespace imsrg::io
