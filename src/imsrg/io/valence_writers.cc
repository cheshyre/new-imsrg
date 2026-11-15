// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Valence-space interaction writers (tokyo .snt, NuShellX .int and .sp); a port
// of imsrg++ ReadWrite::WriteTokyo, WriteNuShellX_intfile and WriteNuShellX_sps
// with the same thresholds, ordering and number formats, including the
// stream-state quirks of the original.
#include "imsrg/io/valence_writers.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "imsrg/base/constants.h"
#include "imsrg/base/error.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/operator/norms.h"

namespace imsrg::io {

namespace {

using basis::Ket;
using basis::ModelSpace;
using basis::TwoBodyChannel;

constexpr int kWint = 4;  // width for printing integers

/// The valence orbits in file order and the inverse numbering.
struct ValenceNumbering {
  std::vector<OrbitIndex>
      orbits;  ///< 1-based file index k is orbits[k-1] (protons first).
  std::vector<std::int32_t>
      file_index;  ///< Orbit index -> file index (0 if not valence).
  std::size_t num_protons = 0;
  std::size_t num_neutrons = 0;
};

/// Protons then neutrons, each ascending (imsrg++ orb2kshell / orb2nushell).
ValenceNumbering NumberValenceOrbits(const ModelSpace& ms) {
  ValenceNumbering v;
  v.file_index.assign(static_cast<std::size_t>(ms.num_orbits()), 0);
  const auto valence = ms.partition().valence();
  for (const OrbitIndex a : valence)
    if (ms.basis().orbit(a).is_proton()) v.orbits.push_back(a);
  v.num_protons = v.orbits.size();
  for (const OrbitIndex a : valence)
    if (ms.basis().orbit(a).is_neutron()) v.orbits.push_back(a);
  v.num_neutrons = v.orbits.size() - v.num_protons;
  for (std::size_t k = 0; k < v.orbits.size(); ++k)
    v.file_index[static_cast<std::size_t>(v.orbits[k].value())] =
        static_cast<std::int32_t>(k + 1);
  return v;
}

/// One two-body matrix element of the valence space, in imsrg++ iteration
/// order.
struct TbmeRecord {
  OrbitIndex a, b, c, d;  ///< The orbits of the bra (a <= b) and ket (c <= d).
  const Ket* bra = nullptr;  ///< Bra ket (for the exchange phase).
  const Ket* ket = nullptr;  ///< Ket ket.
  const TwoBodyChannel* ch = nullptr;
  double value =
      0.0;  ///< Normalized matrix element (isospin averaged in "op" mode).
};

/// Visits all valence-valence (bra, ket) pairs with ket >= bra (channel-local
/// order) of every channel in channel order and calls `f(record)` with the
/// matrix element filled in.
template <class F>
void ForEachValenceTbme(const op::Operator& op, ValenceWriteMode mode,
                        const std::vector<bool>& is_valence, F&& f) {
  const ModelSpace& ms = op.model_space();
  const auto& tb = ms.two_body();
  const auto& basis = ms.basis();
  for (std::int32_t chi = 0; chi < tb.num_channels(); ++chi) {
    const ChannelIndex ch{chi};
    const TwoBodyChannel& tbc = tb.channel(ch);
    // Valence-valence kets in channel-local order.
    std::vector<std::int32_t> vv;
    const auto kets = tb.channel_kets(ch);
    for (std::size_t i = 0; i < kets.size(); ++i) {
      const Ket& k = tb.ket(kets[i]);
      if (is_valence[static_cast<std::size_t>(k.p.value())] &&
          is_valence[static_cast<std::size_t>(k.q.value())])
        vv.push_back(static_cast<std::int32_t>(i));
    }
    for (std::size_t ib = 0; ib < vv.size(); ++ib) {
      const Ket& bra = tb.ket(kets[static_cast<std::size_t>(vv[ib])]);
      for (std::size_t ik = ib; ik < vv.size(); ++ik) {
        const Ket& ket = tb.ket(kets[static_cast<std::size_t>(vv[ik])]);
        TbmeRecord r{bra.p, bra.q, ket.p, ket.q, &bra, &ket, &tbc, 0.0};
        r.value = op.two_body().get(ch, ch, LocalKetIndex{vv[ib]},
                                    LocalKetIndex{vv[ik]});
        if (mode == ValenceWriteMode::kIsospinAveragedOperator &&
            basis.orbit(bra.p).tz2 != basis.orbit(bra.q).tz2) {
          auto partner = [&](OrbitIndex o) {
            return OrbitIndex{o.value() - basis.orbit(o).tz2.twice};
          };
          r.value += op.two_body().get_normalized(
              partner(bra.p), partner(bra.q), partner(ket.p), partner(ket.q),
              tbc.two_j);
          r.value /= 2;
        }
        f(r);
      }
    }
  }
}

std::string BasenameOf(const std::string& path) {
  const std::size_t pos = path.find_last_of("/\\");
  return pos == std::string::npos ? path : path.substr(pos + 1);
}

std::int32_t RoundToInt(double x) {
  return static_cast<std::int32_t>(std::lround(x));
}

std::ofstream OpenForWriting(const std::filesystem::path& path) {
  std::ofstream out(path, std::ofstream::out);
  if (!out)
    throw RuntimeError("cannot open '" + path.string() + "' for writing");
  return out;
}

void CheckWritten(std::ofstream& out, const std::filesystem::path& path) {
  out.flush();
  if (!out) throw RuntimeError("error while writing '" + path.string() + "'");
}

std::vector<bool> ValenceMask(const ModelSpace& ms) {
  std::vector<bool> m(static_cast<std::size_t>(ms.num_orbits()), false);
  for (const OrbitIndex a : ms.partition().valence())
    m[static_cast<std::size_t>(a.value())] = true;
  return m;
}

/// The comment header shared by the .snt and .int files (up to and including
/// the A/Z line).
void WriteProvenance(std::ostream& out, const ValenceWriteOptions& o) {
  out << "! shell model effective interaction generated by IMSRG version "
      << o.version << "\n";
  out << "! input 2N: " << BasenameOf(o.file_2n) << "\n";
  out << "! input 3N: " << BasenameOf(o.file_3n) << "\n";
  out << "! e1max: " << o.e1max << "  e2max: " << o.e2max
      << "   e3max: " << o.e3max << "   hw: " << o.hbar_omega;
  out << "   Aref: " << o.a_ref << "  Zref: " << o.z_ref
      << "  A_for_kinetic_energy: " << o.a_target << "\n";
}

}  // namespace

void WriteTokyo(const std::filesystem::path& path, const op::Operator& op,
                const TokyoWriteOptions& options) {
  const ModelSpace& ms = op.model_space();
  const auto& basis = ms.basis();
  const ValenceNumbering num = NumberValenceOrbits(ms);
  const std::vector<bool> is_valence = ValenceMask(ms);
  const std::int32_t Zcore = RoundToInt(ms.partition().Z_core());
  const std::int32_t Ncore = RoundToInt(ms.partition().A_core()) - Zcore;
  const double norm1b = op::OneBodyNorm(op);
  const double norm2b = op::TwoBodyNorm(op);

  // One-body elements (a >= b over the ascending valence orbits) that survive
  // the threshold.
  struct Row1 {
    std::int32_t a, b;
    double v;
  };
  std::vector<Row1> one;
  for (const OrbitIndex a : ms.partition().valence())
    for (const OrbitIndex b : ms.partition().valence()) {
      if (a < b) continue;
      const double obme = op.one_body().get(a, b);
      if (std::abs(obme) < 1e-7 * norm1b || norm1b < 1e-8) continue;
      one.push_back({num.file_index[static_cast<std::size_t>(a.value())],
                     num.file_index[static_cast<std::size_t>(b.value())],
                     obme});
    }

  struct Row2 {
    std::int32_t a, b, c, d, J;
    double v;
  };
  std::vector<Row2> two;
  if (norm2b >= 1e-7) {
    ForEachValenceTbme(op, options.mode, is_valence, [&](const TbmeRecord& r) {
      if (std::abs(r.value) < norm2b * 1e-7) return;
      auto idx = [&](OrbitIndex o) {
        return num.file_index[static_cast<std::size_t>(o.value())];
      };
      two.push_back(
          {idx(r.a), idx(r.b), idx(r.c), idx(r.d), r.ch->J(), r.value});
    });
  }

  std::ofstream out = OpenForWriting(path);
  WriteProvenance(out, options);
  out << "! Zero body term: " << std::setprecision(9) << op.zero_body() << "\n";
  out << "! \n";
  out << "! model space\n";
  out << std::setw(kWint) << num.num_protons << std::setw(kWint)
      << num.num_neutrons << std::setw(kWint) << Zcore << std::setw(kWint)
      << Ncore << "\n";
  for (std::size_t k = 0; k < num.orbits.size(); ++k) {
    const auto& oi = basis.orbit(num.orbits[k]);
    out << std::setw(kWint) << k + 1 << std::setw(kWint) << oi.n
        << std::setw(kWint) << oi.l << std::setw(kWint) << oi.j2.twice
        << std::setw(kWint) << oi.tz2.twice << "\n";
  }

  out << "! interaction\n";
  out << one.size() << " " << 0 << " " << options.hbar_omega << "\n";
  for (const Row1& r : one)
    out << std::setw(kWint) << r.a << std::setw(kWint) << r.b << "   "
        << std::setw(16) << std::setiosflags(std::ios::fixed)
        << std::setprecision(16) << r.v << "\n";

  out << two.size() << " " << 0 << " " << options.hbar_omega << "\n";
  for (const Row2& r : two)
    out << std::setw(kWint) << r.a << std::setw(kWint) << r.b
        << std::setw(kWint) << r.c << std::setw(kWint) << r.d
        << std::setw(kWint) << r.J << "  " << std::setw(16)
        << std::setiosflags(std::ios::fixed) << std::setprecision(16) << r.v
        << "\n";
  CheckWritten(out, path);
}

void WriteNushellxInt(const std::filesystem::path& path, const op::Operator& op,
                      const NushellxWriteOptions& options) {
  const ModelSpace& ms = op.model_space();
  const auto& basis = ms.basis();
  const ValenceNumbering num = NumberValenceOrbits(ms);
  const std::vector<bool> is_valence = ValenceMask(ms);
  const std::int32_t Acore = RoundToInt(ms.partition().A_core());

  std::ofstream out = OpenForWriting(path);
  WriteProvenance(out, options);
  out << "! Zero body term: " << op.zero_body() << "\n";
  out << "! Index   n l j tz\n";
  for (std::size_t k = 0; k < num.orbits.size(); ++k) {
    const auto& oi = basis.orbit(num.orbits[k]);
    out << "!  " << k + 1 << "   " << oi.n << " " << oi.l << " " << oi.j2.twice
        << "/2" << " " << oi.tz2.twice << "/2\n";
  }
  out << "!\n";
  out << "-999  ";
  for (const OrbitIndex a : num.orbits) out << op.one_body().get(a, a) << "  ";
  out << "  " << Acore << "  " << Acore + 2 << "  0.00000 \n";

  ForEachValenceTbme(op, options.mode, is_valence, [&](const TbmeRecord& r) {
    auto idx = [&](OrbitIndex o) {
      return num.file_index[static_cast<std::size_t>(o.value())];
    };
    const auto& oa = basis.orbit(r.a);
    const auto& ob = basis.orbit(r.b);
    const auto& oc = basis.orbit(r.c);
    const auto& od = basis.orbit(r.d);
    std::int32_t a_ind = idx(r.a), b_ind = idx(r.b), c_ind = idx(r.c),
                 d_ind = idx(r.d);
    std::int32_t T = std::abs(r.ch->Tz());
    double tbme = r.value;
    if (std::abs(tbme) < 1e-6) return;
    if (T == 0) {
      if (oa.j2 != ob.j2 || oa.l != ob.l || oa.n != ob.n)
        tbme *= constants::kSqrt2;
      if (oc.j2 != od.j2 || oc.l != od.l || oc.n != od.n)
        tbme *= constants::kSqrt2;
      T = (r.ch->J() + 1) % 2;
    }
    // NuShellX wants the proton orbit first: ascending bra and ket indices.
    if (a_ind > b_ind) {
      std::swap(a_ind, b_ind);
      tbme *= basis::ExchangePhase(oa.j2, ob.j2, r.ch->two_j);
    }
    if (c_ind > d_ind) {
      std::swap(c_ind, d_ind);
      tbme *= basis::ExchangePhase(oc.j2, od.j2, r.ch->two_j);
    }
    if (a_ind > c_ind || (c_ind == a_ind && b_ind > d_ind)) {
      std::swap(a_ind, c_ind);
      std::swap(b_ind, d_ind);
    }
    out << std::setw(kWint) << a_ind << std::setw(kWint) << b_ind
        << std::setw(kWint) << c_ind << std::setw(kWint) << d_ind << "    "
        << std::setw(kWint) << r.ch->J() << " " << std::setw(kWint) << T
        << "       " << std::setw(12) << std::setiosflags(std::ios::fixed)
        << std::setprecision(6) << tbme << "\n";
  });
  CheckWritten(out, path);
}

void WriteNushellxSps(const std::filesystem::path& path, const op::Operator& op,
                      const NushellxWriteOptions& /*options*/) {
  const ModelSpace& ms = op.model_space();
  const ValenceNumbering num = NumberValenceOrbits(ms);
  std::ofstream out = OpenForWriting(path);
  out << "! modelspace for IMSRG interaction\n";
  out << "pn\n";
  out << RoundToInt(ms.partition().A_core()) << " "
      << RoundToInt(ms.partition().Z_core()) << "\n";
  out << ms.partition().valence().size() << "\n";
  out << "2 " << num.num_protons << " " << num.num_neutrons << "\n";
  for (std::size_t k = 0; k < num.orbits.size(); ++k) {
    const auto& oi = ms.basis().orbit(num.orbits[k]);
    out << k + 1 << " " << oi.n + 1 << " " << oi.l << " " << oi.j2.twice
        << "\n";
  }
  out << "\n";
  CheckWritten(out, path);
}

}  // namespace imsrg::io
