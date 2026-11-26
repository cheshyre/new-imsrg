// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/io/operator_binary.h"

#include <unistd.h>

#include <bit>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "imsrg/base/error.h"

namespace imsrg::io {
namespace {

static_assert(std::endian::native == std::endian::little,
              "the native operator format is little-endian");

constexpr char kMagic[8] = {'I', 'M', 'S', 'R', 'G', 'O', 'P', 'B'};
constexpr std::uint32_t kVersion = 1;

/// FNV-1a 64 accumulator.
class Fnv {
 public:
  void Add(const void* data, std::size_t n) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < n; ++i) {
      h_ ^= p[i];
      h_ *= 0x100000001b3ULL;
    }
  }
  template <class T>
  void AddValue(const T& v) {
    Add(&v, sizeof(T));
  }
  std::uint64_t value() const noexcept { return h_; }

 private:
  std::uint64_t h_ = 0xcbf29ce484222325ULL;
};

/// Byte sink that also feeds the checksum.
class Writer {
 public:
  explicit Writer(std::ostream& out) : out_(out) {}
  void Bytes(const void* p, std::size_t n) {
    out_.write(static_cast<const char*>(p), static_cast<std::streamsize>(n));
    sum_.Add(p, n);
  }
  template <class T>
  void Value(const T& v) {
    Bytes(&v, sizeof(T));
  }
  std::uint64_t checksum() const noexcept { return sum_.value(); }

 private:
  std::ostream& out_;
  Fnv sum_;
};

/// Checked byte source that also feeds the checksum.
class Reader {
 public:
  Reader(std::istream& in, std::string name)
      : in_(in), name_(std::move(name)) {}
  void Bytes(void* p, std::size_t n) {
    in_.read(static_cast<char*>(p), static_cast<std::streamsize>(n));
    if (static_cast<std::size_t>(in_.gcount()) != n) {
      throw InputError(name_ + ": truncated native operator file at byte " +
                       std::to_string(offset_));
    }
    sum_.Add(p, n);
    offset_ += n;
  }
  template <class T>
  T Value() {
    T v{};
    Bytes(&v, sizeof(T));
    return v;
  }
  std::uint64_t checksum() const noexcept { return sum_.value(); }
  std::uint64_t offset() const noexcept { return offset_; }

 private:
  std::istream& in_;
  std::string name_;
  Fnv sum_;
  std::uint64_t offset_ = 0;
};

void WriteHeader(Writer& w, const OperatorFileHeader& h) {
  w.Value(h.layout_hash);
  w.Value(h.reference_hash);
  w.Value(static_cast<std::int32_t>(h.spec.particle_rank));
  w.Value(static_cast<std::int32_t>(h.spec.rank_J));
  w.Value(static_cast<std::int32_t>(h.spec.parity));
  w.Value(static_cast<std::int32_t>(h.spec.delta_Tz));
  w.Value(static_cast<std::int32_t>(h.spec.hermiticity));
  w.Value(static_cast<std::int32_t>(h.spec.delta_particle_number));
  w.Value(h.emax);
  w.Value(h.e2max);
  w.Value(h.hw);
  w.Value(static_cast<std::uint32_t>(h.label.size()));
  w.Bytes(h.label.data(), h.label.size());
  w.Value(h.one_body_count);
  w.Value(h.two_body_count);
}

/// Reads magic, version, header length and the header.
OperatorFileHeader ReadHeader(std::istream& in, Reader& r,
                              const std::string& name) {
  char magic[8];
  in.read(magic, 8);
  if (in.gcount() != 8 || std::memcmp(magic, kMagic, 8) != 0) {
    throw InputError(name + ": not a native imsrg operator file");
  }
  OperatorFileHeader h;
  h.version = r.Value<std::uint32_t>();
  if (h.version != kVersion) {
    throw InputError(name + ": unsupported native operator format version " +
                     std::to_string(h.version));
  }
  r.Value<std::uint32_t>();  // header length (informational)
  h.layout_hash = r.Value<std::uint64_t>();
  h.reference_hash = r.Value<std::uint64_t>();
  h.spec.particle_rank = r.Value<std::int32_t>();
  h.spec.rank_J = r.Value<std::int32_t>();
  const auto parity = r.Value<std::int32_t>();
  h.spec.delta_Tz = r.Value<std::int32_t>();
  const auto herm = r.Value<std::int32_t>();
  h.spec.delta_particle_number = r.Value<std::int32_t>();
  if (parity < 0 || parity > 1 || herm < 0 || herm > 2) {
    throw InputError(name + ": corrupt operator spec in the header");
  }
  h.spec.parity = static_cast<Parity>(parity);
  h.spec.hermiticity = static_cast<Hermiticity>(herm);
  h.emax = r.Value<std::int32_t>();
  h.e2max = r.Value<std::int32_t>();
  h.hw = r.Value<double>();
  const auto len = r.Value<std::uint32_t>();
  if (len > (1U << 20)) throw InputError(name + ": corrupt label length");
  h.label.resize(len);
  r.Bytes(h.label.data(), len);
  h.one_body_count = r.Value<std::uint64_t>();
  h.two_body_count = r.Value<std::uint64_t>();
  return h;
}

std::ifstream OpenIn(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw InputError("cannot open '" + path.string() + "'");
  return in;
}

}  // namespace

std::uint64_t ModelSpaceLayoutHash(const basis::ModelSpace& ms) {
  Fnv f;
  const auto& basis = ms.basis();
  f.AddValue(basis.num_orbits());
  for (const basis::Orbit& o : basis.orbits()) {
    f.AddValue(o.n);
    f.AddValue(o.l);
    f.AddValue(o.j2.twice);
    f.AddValue(o.tz2.twice);
  }
  f.AddValue(std::bit_cast<std::uint64_t>(basis.hbar_omega()));
  const basis::TwoBodyChannels& tb = ms.two_body();
  f.AddValue(tb.num_channels());
  for (std::int32_t c = 0; c < tb.num_channels(); ++c) {
    const basis::TwoBodyChannel& ch = tb.channel(ChannelIndex(c));
    f.AddValue(ch.two_j.twice);
    f.AddValue(static_cast<std::int32_t>(ch.parity));
    f.AddValue(ch.two_tz.twice);
    f.AddValue(ch.num_kets);
    for (std::int32_t i = 0; i < ch.num_kets; ++i) {
      const basis::Ket& k =
          tb.ket(tb.global_ket(ChannelIndex(c), LocalKetIndex(i)));
      f.AddValue(k.p.value());
      f.AddValue(k.q.value());
    }
  }
  return f.value();
}

std::uint64_t ReferenceHash(const basis::ModelSpace& ms) {
  Fnv f;
  for (const double n : ms.reference().occupations()) {
    f.AddValue(std::bit_cast<std::uint64_t>(n));
  }
  return f.value();
}

void WriteFileAtomically(const std::filesystem::path& path,
                         const std::function<void(std::ostream&)>& write) {
  std::filesystem::path tmp = path;
  tmp += ".tmp." + std::to_string(::getpid());
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) throw InputError("cannot write '" + tmp.string() + "'");
    write(out);
    out.flush();
    if (!out) throw InputError("error while writing '" + tmp.string() + "'");
  }
  std::error_code ec;
  std::filesystem::rename(tmp, path, ec);
  if (ec) {
    std::filesystem::remove(tmp, ec);
    throw InputError("cannot rename '" + tmp.string() + "' to '" +
                     path.string() + "'");
  }
}

void WriteOperatorBinary(const std::filesystem::path& path,
                         const op::Operator& op, std::string_view label) {
  if (op.spec().particle_rank > 2) {
    throw NotImplementedError(
        "WriteOperatorBinary: three-body parts are not supported yet");
  }
  op.SyncToHost();
  const auto& ms = op.model_space();
  OperatorFileHeader h;
  h.version = kVersion;
  h.layout_hash = ModelSpaceLayoutHash(ms);
  h.reference_hash = ReferenceHash(ms);
  h.spec = op.spec();
  h.emax = ms.basis().emax();
  h.e2max = 2 * h.emax;
  h.hw = ms.basis().hbar_omega();
  h.label = std::string(label);
  const std::span<const double> b1 = op.one_body().buffer().host_span();
  std::span<const double> b2;
  if (op.two_body().allocated()) b2 = op.two_body().matrix().buffer().host_span();
  h.one_body_count = b1.size();
  h.two_body_count = b2.size();
  // Header length: everything WriteHeader emits.
  const auto header_len = static_cast<std::uint32_t>(
      2 * 8 + 8 * 4 + 8 + 4 + h.label.size() + 2 * 8);
  WriteFileAtomically(path, [&](std::ostream& out) {
    out.write(kMagic, 8);
    Writer w(out);
    w.Value(kVersion);
    w.Value(header_len);
    WriteHeader(w, h);
    w.Value(op.zero_body());
    w.Bytes(b1.data(), b1.size_bytes());
    w.Bytes(b2.data(), b2.size_bytes());
    const std::uint64_t sum = w.checksum();
    out.write(reinterpret_cast<const char*>(&sum), sizeof(sum));
  });
}

OperatorFileHeader ReadOperatorBinaryHeader(const std::filesystem::path& path) {
  std::ifstream in = OpenIn(path);
  Reader r(in, path.string());
  return ReadHeader(in, r, path.string());
}

op::Operator ReadOperatorBinary(const std::filesystem::path& path,
                                std::shared_ptr<const basis::ModelSpace> ms,
                                const OperatorBinaryReadOptions& options) {
  const std::string name = path.string();
  std::ifstream in = OpenIn(path);
  Reader r(in, name);
  const OperatorFileHeader h = ReadHeader(in, r, name);
  if (h.layout_hash != ModelSpaceLayoutHash(*ms)) {
    std::ostringstream m;
    m << name << ": model-space layout hash " << std::hex << h.layout_hash
      << " differs from the target's " << ModelSpaceLayoutHash(*ms)
      << " (emax " << std::dec << h.emax << " in the file, "
      << ms->basis().emax() << " here)";
    throw InputError(m.str());
  }
  if (options.check_reference && h.reference_hash != ReferenceHash(*ms)) {
    throw InputError(name +
                     ": the reference occupations differ from the target "
                     "model space's");
  }
  if (h.spec.particle_rank > 2) {
    throw InputError(name + ": particle rank 3 is not supported");
  }
  op::Operator out(std::move(ms), h.spec);
  std::span<double> b1 = out.one_body().buffer().host_span();
  std::span<double> b2;
  if (out.two_body().allocated()) {
    b2 = out.two_body().matrix().buffer().host_span();
  }
  if (h.one_body_count != b1.size() || h.two_body_count != b2.size()) {
    throw InputError(name + ": buffer sizes differ from the operator's (" +
                     std::to_string(h.one_body_count) + ", " +
                     std::to_string(h.two_body_count) + " vs " +
                     std::to_string(b1.size()) + ", " +
                     std::to_string(b2.size()) + ")");
  }
  out.zero_body() = r.Value<double>();
  r.Bytes(b1.data(), b1.size_bytes());
  r.Bytes(b2.data(), b2.size_bytes());
  const std::uint64_t expect = r.checksum();
  std::uint64_t sum = 0;
  in.read(reinterpret_cast<char*>(&sum), sizeof(sum));
  if (in.gcount() != sizeof(sum)) {
    throw InputError(name + ": truncated native operator file (checksum)");
  }
  if (sum != expect) throw InputError(name + ": checksum mismatch");
  if (in.peek() != std::char_traits<char>::eof()) {
    throw InputError(name + ": trailing bytes after the checksum");
  }
  out.SyncToDevice();
  return out;
}

}  // namespace imsrg::io
