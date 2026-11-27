// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Port of imsrg++ ReadWrite::WriteTensorTokyo (devel ReadWrite.cc:6087 and the
// 79e9330 fork ReadWrite.cc:5733), including the sticky stream-state quirks.
#include "imsrg/io/tensor_writers.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <string>
#include <vector>

#include "imsrg/base/error.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/operator/norms.h"

namespace imsrg::io {

namespace {

using basis::ModelSpace;

constexpr int kWint = 4;

std::string BasenameOf(const std::string& path) {
  const std::size_t pos = path.find_last_of("/\\");
  return pos == std::string::npos ? path : path.substr(pos + 1);
}

std::int32_t RoundToInt(double x) {
  return static_cast<std::int32_t>(std::lround(x));
}

struct Row1 {
  std::int32_t a, b;
  double v;
};

struct Row2 {
  std::int32_t a, b, c, d, jb, jk;
  double v;
};

/// Resets the formatting state of a stream to the defaults and restores the
/// previous state on scope exit.
class FormatGuard {
 public:
  explicit FormatGuard(std::ostream& s)
      : s_(s), flags_(s.flags()), precision_(s.precision()), width_(s.width()) {
    s_.flags(std::ios_base::skipws | std::ios_base::dec);
    s_.precision(6);
    s_.width(0);
  }
  FormatGuard(const FormatGuard&) = delete;
  FormatGuard& operator=(const FormatGuard&) = delete;
  ~FormatGuard() {
    s_.flags(flags_);
    s_.precision(precision_);
    s_.width(width_);
  }

 private:
  std::ostream& s_;
  std::ios_base::fmtflags flags_;
  std::streamsize precision_;
  std::streamsize width_;
};

}  // namespace

void WriteTensorTokyo(std::ostream& out, const op::Operator& op,
                      const TensorTokyoOptions& options) {
  if (op.spec().is_scalar())
    throw InputError("WriteTensorTokyo: the operator is a scalar");
  const ModelSpace& ms = op.model_space();
  const auto& basis = ms.basis();
  const auto& tb = ms.two_body();
  const bool fork = options.format == TokyoFormat::kFork79e9330;
  const int wdouble = fork ? 12 : 16;
  const int pdouble = fork ? 6 : 16;

  // kshell numbering: valence protons first, then neutrons, each ascending.
  const auto valence = ms.partition().valence();
  std::vector<OrbitIndex> orbits;
  for (const OrbitIndex a : valence)
    if (basis.orbit(a).is_proton()) orbits.push_back(a);
  const std::size_t num_protons = orbits.size();
  for (const OrbitIndex a : valence)
    if (basis.orbit(a).is_neutron()) orbits.push_back(a);
  const std::size_t num_neutrons = orbits.size() - num_protons;
  std::vector<std::int32_t> kshell(static_cast<std::size_t>(ms.num_orbits()),
                                   0);
  std::vector<bool> is_valence(static_cast<std::size_t>(ms.num_orbits()),
                               false);
  for (std::size_t k = 0; k < orbits.size(); ++k) {
    kshell[static_cast<std::size_t>(orbits[k].value())] =
        static_cast<std::int32_t>(k + 1);
    is_valence[static_cast<std::size_t>(orbits[k].value())] = true;
  }
  auto idx = [&](OrbitIndex o) {
    return kshell[static_cast<std::size_t>(o.value())];
  };

  std::int32_t a_head, z_head;
  if (fork) {
    a_head = options.a_ref;
    z_head = options.z_ref;
  } else {
    a_head = RoundToInt(ms.partition().A_core());
    z_head = RoundToInt(ms.partition().Z_core());
  }
  const std::int32_t n_head = a_head - z_head;

  // The devel format thresholds against imsrg-devel's integer tensor norm
  // (AL-A-Q6); if that is NaN (negative degeneracy sum), the weighted one.
  double norm1b = 0.0;
  if (!fork) {
    norm1b = op::OneBodyNorm(op, op::TensorOneBodyNorm::kImsrgppInteger);
    if (!std::isfinite(norm1b)) norm1b = op::OneBodyNorm(op);
  }
  const double norm2b = fork ? 0.0 : op::TwoBodyNorm(op);
  auto drop1 = [&](double v) {
    return fork ? std::abs(v) < 1.0e-7
                : (std::abs(v) < norm1b * 1.0e-7 || norm1b == 0.0);
  };
  auto drop2 = [&](double v) {
    return fork ? std::abs(v) < 1.0e-7 : std::abs(v) < 1.0e-9;
  };

  // One-body rows: all ordered valence pairs.
  std::vector<Row1> one;
  for (const OrbitIndex a : valence)
    for (const OrbitIndex b : valence) {
      const double v = op.one_body().get(a, b);
      if (drop1(v)) continue;
      one.push_back({idx(a), idx(b), v});
    }

  // Two-body rows: stored blocks in (ch_bra, ch_ket) order.
  std::vector<Row2> two;
  if (fork || norm2b > 1.0e-8) {
    std::vector<std::vector<std::int32_t>> vv(
        static_cast<std::size_t>(tb.num_channels()));
    for (std::int32_t c = 0; c < tb.num_channels(); ++c) {
      const auto kets = tb.channel_kets(ChannelIndex{c});
      for (std::size_t i = 0; i < kets.size(); ++i) {
        const auto& k = tb.ket(kets[i]);
        if (is_valence[static_cast<std::size_t>(k.p.value())] &&
            is_valence[static_cast<std::size_t>(k.q.value())])
          vv[static_cast<std::size_t>(c)].push_back(
              static_cast<std::int32_t>(i));
      }
    }
    for (std::int32_t cb = 0; cb < tb.num_channels(); ++cb) {
      if (vv[static_cast<std::size_t>(cb)].empty()) continue;
      for (std::int32_t ck = cb; ck < tb.num_channels(); ++ck) {
        if (!op.two_body().block_index(ChannelIndex{cb}, ChannelIndex{ck}))
          continue;
        const auto kb = tb.channel_kets(ChannelIndex{cb});
        const auto kk = tb.channel_kets(ChannelIndex{ck});
        for (const std::int32_t ib : vv[static_cast<std::size_t>(cb)]) {
          const auto& bra = tb.ket(kb[static_cast<std::size_t>(ib)]);
          for (const std::int32_t ik : vv[static_cast<std::size_t>(ck)]) {
            const auto& ket = tb.ket(kk[static_cast<std::size_t>(ik)]);
            const double v =
                op.two_body().get(ChannelIndex{cb}, ChannelIndex{ck},
                                  LocalKetIndex{ib}, LocalKetIndex{ik});
            if (drop2(v)) continue;
            two.push_back({idx(bra.p), idx(bra.q), idx(ket.p), idx(ket.q),
                           tb.channel(ChannelIndex{cb}).J(),
                           tb.channel(ChannelIndex{ck}).J(), v});
          }
        }
      }
    }
  }

  FormatGuard guard(out);
  out << "! shell model effective operator generated by IMSRG version "
      << options.version << "\n";
  out << "! input 2N: " << BasenameOf(options.file_2n) << "\n";
  out << "! input 3N: " << BasenameOf(options.file_3n) << "\n";
  out << "! e1max: " << options.e1max << "  e2max: " << options.e2max
      << "   e3max: " << options.e3max << "   hw: " << options.hbar_omega;
  out << "   Aref: " << a_head << "  Zref: " << z_head
      << "  A_for_kinetic_energy: " << options.a_target << "\n";
  out << "! Zero body term: " << op.zero_body() << "\n";
  out << "! \n";
  out << "! model space\n";
  out << std::setw(kWint) << num_protons << std::setw(kWint) << num_neutrons
      << std::setw(kWint) << z_head << std::setw(kWint) << n_head << "\n";
  for (std::size_t k = 0; k < orbits.size(); ++k) {
    const auto& oi = basis.orbit(orbits[k]);
    out << std::setw(kWint) << k + 1 << std::setw(kWint) << oi.n
        << std::setw(kWint) << oi.l << std::setw(kWint) << oi.j2.twice
        << std::setw(kWint) << oi.tz2.twice << "\n";
  }

  out << "! reduced matrix elements\n";
  out << one.size() << " " << 0 << " " << options.hbar_omega << "\n";
  for (const Row1& r : one) {
    if (fork) {
      out << std::setw(kWint) << r.a << std::setw(kWint) << r.b
          << std::setw(wdouble) << " " << std::setiosflags(std::ios::scientific)
          << std::setprecision(pdouble) << r.v << "\n";
    } else {
      out << std::setw(kWint) << r.a << " " << std::setw(kWint) << r.b << "   "
          << std::setw(wdouble) << std::setiosflags(std::ios::fixed)
          << std::setprecision(pdouble) << r.v << "\n";
    }
  }

  out << two.size() << " " << 0 << " " << options.hbar_omega << "\n";
  for (const Row2& r : two) {
    out << std::setw(kWint) << r.a << " " << std::setw(kWint) << r.b << " "
        << std::setw(kWint) << r.c << " " << std::setw(kWint) << r.d << "   "
        << std::setw(kWint) << r.jb << " " << std::setw(kWint) << r.jk << "   "
        << std::setw(wdouble) << std::setprecision(pdouble) << r.v << "\n";
  }
}

void WriteTensorTokyo(const std::filesystem::path& path, const op::Operator& op,
                      const TensorTokyoOptions& options) {
  std::ofstream out(path, std::ofstream::out);
  if (!out)
    throw RuntimeError("cannot open '" + path.string() + "' for writing");
  WriteTensorTokyo(out, op, options);
  out.flush();
  if (!out) throw RuntimeError("error while writing '" + path.string() + "'");
}

}  // namespace imsrg::io
