#pragma once
/// @file
/// Second- and third-order many-body perturbation energies of a normal-ordered Hamiltonian
/// expressed in the (HF) basis where the one-body part is diagonal (imsrg++
/// `Operator::GetMP2_Energy` / `GetMP3_Energy`, Operator.cc:1185-1418). Energies in MeV; the
/// unperturbed energies are the diagonal of the one-body part; open-shell occupations enter as
/// weights exactly as in imsrg++. Host only.

#include "imsrg/operator/operator.hpp"

namespace imsrg::mbpt {

/// Third-order energy split into the imsrg++ terms (Epp, Ehh, Eph) and their sum.
struct Mp3Energy {
  double pp = 0.0;  ///< Particle-particle ladder.
  double hh = 0.0;  ///< Hole-hole ladder.
  double ph = 0.0;  ///< Particle-hole ring term (imsrg++ comments it as unreliable; matched as is).
  double total = 0.0;  ///< pp + hh + ph.
};

/// MBPT(2) correlation energy E2 = sum (2j_a+1) n_a f_ia^2/(f_aa-f_ii) + sum_{i<=j particles,
/// a<=b holes} (2J+1) n_a n_b <ab|V|ij>_J^2 / (e_a+e_b-e_i-e_j), with normalized matrix elements,
/// even J only when a==b or i==j. H must be in the basis where f is diagonal. Syncs H to host.
/// Complexity: O(N_p^2 N_h^2 J_max).
double mp2_energy(op::Operator& H);

/// MBPT(3) energy terms (pp, hh, ph) as in imsrg++ `GetMP3_Energy`. Same preconditions as
/// `mp2_energy`. Complexity: O(sum_ch n_pp^2 n_hh + n_hh^2 n_pp) plus the ph term
/// O(sum_Jph n_ph^2 J_max).
Mp3Energy mp3_energy(op::Operator& H);

}  // namespace imsrg::mbpt
