// rh_basis.cpp -- `refharness dump-basis`: the imsrg++ basis bookkeeping as plain text.
//
// Format "imsrg-basis-dump v1" (documented in tests/golden/README.md): one record per line,
// fields separated by single spaces, `#` comments, floats with %.17g. Sections in order:
//   header, orbits (with occupation and cvq), orbit sets, kets (p <= q, global index order),
//   2B channels (J, parity, Tz, kets in local order, ket subsets as local indices),
//   ph / cross-coupled channels (TwoBodyChannel_CC: J, parity, |Tz|, kets p <= q),
//   3B channels (2J, parity, 2Tz, number of kets) and the number of 3B kets.
//
// Usage: refharness dump-basis emax=<e> reference=<nuc|default> valence_space=<name> \
//          [e3max=12] [lmax=99999] [hw=16] [out=<file>|-]

#include <fstream>
#include <iostream>

#include "IMSRG.hh"
#include "rh_common.hh"

namespace rh {

namespace {

template <class Container>
void write_set(std::ostream& o, const char* name, const Container& s) {
  o << name << " " << s.size();
  for (auto i : s) o << " " << i;
  o << "\n";
}

void write_subset(std::ostream& o, int ch, const char* name, const arma::uvec& v) {
  o << "channel_subset " << ch << " " << name << " " << v.n_elem;
  for (arma::uword i = 0; i < v.n_elem; ++i) o << " " << v(i);
  o << "\n";
}

}  // namespace

int cmd_dump_basis(const Args& a) {
  SpaceParams p = space_params(a);
  auto msp = make_modelspace(p);
  ModelSpace& ms = *msp;
  std::string out = a.str("out", "-");
  std::ofstream fout;
  if (out != "-") {
    fout.open(out);
    if (!fout) throw std::runtime_error("cannot write " + out);
  }
  std::ostream& o = (out == "-") ? std::cout : fout;

  const int lmax_eff = std::min(ms.GetLmax(), ms.GetEmax());
  o << "# imsrg-basis-dump v1\n";
  o << "source " << reference_version() << " refharness dump-basis\n";
  o << "basis emax " << ms.GetEmax() << " e2max " << ms.GetE2max() << " e3max " << ms.GetE3max()
    << " lmax " << lmax_eff << " hw " << fmt17(ms.GetHbarOmega()) << "\n";
  o << "reference " << p.reference << " valence_space "
    << (p.valence_space.empty() ? "none" : p.valence_space) << "\n";
  o << "counts Aref " << fmt17(ms.GetAref()) << " Zref " << fmt17(ms.GetZref()) << " Acore "
    << fmt17(ms.GetAcore()) << " Zcore " << fmt17(ms.GetZcore()) << " target_mass "
    << ms.GetTargetMass() << " target_Z " << ms.GetTargetZ() << "\n";

  // Orbits: index n l 2j 2tz occupation cvq (cvq: 0 core, 1 valence, 2 q-space)
  o << "norbits " << ms.GetNumberOrbits() << "\n";
  for (size_t i = 0; i < ms.GetNumberOrbits(); ++i) {
    const Orbit& oi = ms.GetOrbit(i);
    o << "orbit " << oi.index << " " << oi.n << " " << oi.l << " " << oi.j2 << " " << oi.tz2 << " "
      << fmt17(oi.occ) << " " << oi.cvq << "\n";
  }
  write_set(o, "holes", ms.holes);
  write_set(o, "particles", ms.particles);
  write_set(o, "core", ms.core);
  write_set(o, "valence", ms.valence);
  write_set(o, "qspace", ms.qspace);

  // Kets |pq>, p <= q, in global index order (Index2)
  o << "nkets " << ms.GetNumberKets() << "\n";
  for (size_t k = 0; k < ms.GetNumberKets(); ++k) {
    Ket& ket = ms.GetKet(k);
    o << "ket " << k << " " << ket.p << " " << ket.q << "\n";
  }

  // 2B channels in imsrg++ channel-index order: ch = 6J + 2(Tz+1) + parity
  o << "ntwobody_channels " << ms.GetNumberTwoBodyChannels() << "\n";
  for (size_t ch = 0; ch < ms.GetNumberTwoBodyChannels(); ++ch) {
    TwoBodyChannel& tbc = ms.GetTwoBodyChannel(ch);
    o << "channel " << ch << " " << tbc.J << " " << tbc.parity << " " << tbc.Tz << " "
      << tbc.GetNumberKets() << "\n";
    o << "channel_kets " << ch;
    for (size_t i = 0; i < tbc.GetNumberKets(); ++i) o << " " << tbc.GetKetIndex(i);
    o << "\n";
    write_subset(o, ch, "pp", tbc.GetKetIndex_pp());
    write_subset(o, ch, "hh", tbc.GetKetIndex_hh());
    write_subset(o, ch, "ph", tbc.GetKetIndex_ph());
    write_subset(o, ch, "cc", tbc.GetKetIndex_cc());
    write_subset(o, ch, "vc", tbc.GetKetIndex_vc());
    write_subset(o, ch, "qc", tbc.GetKetIndex_qc());
    write_subset(o, ch, "vv", tbc.GetKetIndex_vv());
    write_subset(o, ch, "qv", tbc.GetKetIndex_qv());
    write_subset(o, ch, "qq", tbc.GetKetIndex_qq());
  }

  // Cross-coupled (Pandya) channels: J, parity, |tz_p - tz_q|/2; kets p <= q (a CC matrix has
  // dimension 2*nkets: local index + nkets stands for the swapped ket |qp>)
  o << "nph_channels " << ms.GetNumberTwoBodyChannels_CC() << "\n";
  for (size_t ch = 0; ch < ms.GetNumberTwoBodyChannels_CC(); ++ch) {
    TwoBodyChannel_CC& tbc = ms.GetTwoBodyChannel_CC(ch);
    o << "ph_channel " << ch << " " << tbc.J << " " << tbc.parity << " " << tbc.Tz << " "
      << tbc.GetNumberKets() << "\n";
    o << "ph_channel_kets " << ch;
    for (size_t i = 0; i < tbc.GetNumberKets(); ++i) o << " " << tbc.GetKetIndex(i);
    o << "\n";
  }

  // 3B channels (counts only): 2J parity 2Tz nkets; kets are |pqr; Jpq> with p<=q<=r and
  // 2(n_p+n_q+n_r)+l_p+l_q+l_r <= e3max
  o << "nthreebody_channels " << ms.GetNumberThreeBodyChannels() << "\n";
  size_t total3 = 0;
  for (size_t ch = 0; ch < ms.GetNumberThreeBodyChannels(); ++ch) {
    ThreeBodyChannel& c3 = ms.GetThreeBodyChannel(ch);
    o << "threebody_channel " << ch << " " << c3.twoJ << " " << c3.parity << " " << c3.twoTz << " "
      << c3.GetNumberKets() << "\n";
    total3 += c3.GetNumberKets();
  }
  o << "nkets3 " << ms.Kets3.size() << " sum_channel_kets3 " << total3 << "\n";
  o << "end\n";
  return 0;
}

}  // namespace rh
