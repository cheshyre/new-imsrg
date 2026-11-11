#include "imsrg/opslib/detail/pair_engine.hpp"

#include <algorithm>
#include <cstdlib>

#include "imsrg/base/constants.hpp"

namespace imsrg::opslib::detail {

PairEngine::PairEngine(int emax, int lmax, double hbar_omega)
    : hbar_omega_(hbar_omega), mosh_(2 * emax), nj_(lmax) {}

double PairEngine::element(PairKind kind, const basis::Orbit& oa, const basis::Orbit& ob,
                           const basis::Orbit& oc, const basis::Orbit& od, TwoJ two_j) const {
  const int fab = oa.e() + ob.e();
  const int fcd = oc.e() + od.e();
  // p1.p2 and r1.r2 only connect pair energies that differ by 0 or 2 (parity conserved).
  if (std::abs(fab - fcd) % 2 > 0 || std::abs(fab - fcd) > 2) {
    return 0.0;
  }
  const int J = two_j.twice / 2;
  const double off = (kind == PairKind::Momentum) ? 1.0 : -1.0;  // sign of the sqrt terms
  const auto mosh = mosh_.host_view();
  const auto nj = nj_.host_view();

  double sum = 0.0;
  for (int Lab = std::abs(oa.l - ob.l); Lab <= oa.l + ob.l; ++Lab) {
    for (int Sab = 0; Sab <= 1; ++Sab) {
      if (std::abs(Lab - Sab) > J || Lab + Sab < J) {
        continue;
      }
      const double njab = nj(oa.l, oa.j2, ob.l, ob.j2, Lab, Sab, two_j);
      if (njab == 0.0) {
        continue;
      }
      const double njcd = nj(oc.l, oc.j2, od.l, od.j2, Lab, Sab, two_j);
      if (njcd == 0.0) {
        continue;
      }
      for (int Nab = 0; Nab <= fab / 2; ++Nab) {
        for (int Lam = 0; Lam <= fab - 2 * Nab; ++Lam) {
          for (int lam = (fab - 2 * Nab - Lam) % 2; lam <= fab - 2 * Nab - Lam; lam += 2) {
            if (Lab < std::abs(Lam - lam) || Lab > Lam + lam) {
              continue;
            }
            // Antisymmetrization factor of imsrg++ (integer arithmetic, phase (-1)^(lam+S)).
            const int phase = ((lam + Sab) % 2 == 0) ? 1 : -1;
            const int asymm = (std::abs(oa.tz2.twice + oc.tz2.twice) +
                               std::abs(oa.tz2.twice + od.tz2.twice) * phase) /
                              2;
            if (asymm == 0) {
              continue;
            }
            const int nab = (fab - 2 * Nab - Lam - lam) / 2;
            const double mab = mosh({Nab, Lam}, {nab, lam}, {oa.n, oa.l}, {ob.n, ob.l}, Lab);
            if (std::abs(mab) < 1e-8) {
              continue;
            }
            for (int Ncd = std::max(0, Nab - 1); Ncd <= Nab + 1; ++Ncd) {
              const int ncd = (fcd - 2 * Ncd - Lam - lam) / 2;
              if (ncd < 0) {
                continue;
              }
              if (nab != ncd && Nab != Ncd) {
                continue;
              }
              const double mcd = mosh({Ncd, Lam}, {ncd, lam}, {oc.n, oc.l}, {od.n, od.l}, Lab);
              if (std::abs(mcd) < 1e-8) {
                continue;
              }
              double cm = 0.0;
              double rel = 0.0;
              if (nab == ncd) {
                if (Nab == Ncd) {
                  cm = 2 * Nab + Lam + 1.5;
                } else if (Nab == Ncd + 1) {
                  cm = off * std::sqrt(Nab * (Nab + Lam + 0.5));
                } else if (Nab == Ncd - 1) {
                  cm = off * std::sqrt(Ncd * (Ncd + Lam + 0.5));
                }
              }
              if (Nab == Ncd) {
                if (nab == ncd) {
                  rel = 2 * nab + lam + 1.5;
                } else if (nab == ncd + 1) {
                  rel = off * std::sqrt(nab * (nab + lam + 0.5));
                } else if (nab == ncd - 1) {
                  rel = off * std::sqrt(ncd * (ncd + lam + 0.5));
                }
              }
              sum += (cm - rel) * njab * njcd * mab * mcd * asymm;
            }
          }
        }
      }
    }
  }

  const double dab = (oa == ob) ? 2.0 : 1.0;
  const double dcd = (oc == od) ? 2.0 : 1.0;
  const double unit = (kind == PairKind::Momentum) ? hbar_omega_
                                                   : constants::hbarc * constants::hbarc /
                                                         constants::m_nucleon / hbar_omega_;
  return sum * 0.5 * unit / std::sqrt(dab * dcd);
}

int max_orbit_shell(const basis::ModelSpace& ms) {
  int e = 0;
  for (const basis::Orbit& o : ms.basis().orbits()) {
    e = std::max(e, static_cast<int>(o.e()));
  }
  return e;
}

}  // namespace imsrg::opslib::detail
