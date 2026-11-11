// rh_hamiltonian.cpp -- `refharness dump-hamiltonian` and `refharness hf`.
//
//   refharness dump-hamiltonian emax=2 reference=O16 [3bme=FILE] [ops=Rp2,random_H] out=DIR
//     Writes DIR/{T,V2,H}.dump (HO basis), DIR/V3_NO2B.dump (0B/1B/2B of the 3N normal ordered
//     w.r.t. the reference, HO basis; only with 3bme), DIR/<op>.dump for each extra operator,
//     and DIR/norms.json (imsrg++ norms of every dumped operator).
//
//   refharness hf emax=4 reference=O16 [3bme=FILE] out=DIR [dump_hno=true]
//     Runs HartreeFock as imsrg++ does (basis=HF, occupations unfrozen) and writes
//     DIR/hf.json (E_HF and its parts, iterations, SPEs, occupations, C, MP2, MP3) and, with
//     dump_hno, DIR/HNO.dump: the HF-basis normal-ordered H (NO2B), plus its norms.

#include <fstream>
#include <iostream>
#include <sstream>

#include "IMSRG.hh"
#include "rh_common.hh"

namespace rh {

namespace {

std::string matrix_json(const arma::mat& M) {
  std::string s = "[";
  for (arma::uword i = 0; i < M.n_rows; ++i) {
    s += (i ? ",\n  [" : "\n  [");
    for (arma::uword j = 0; j < M.n_cols; ++j) s += (j ? ", " : "") + fmt17(M(i, j));
    s += "]";
  }
  return s + "\n]";
}

std::string source_text(const std::string& what) {
  return reference_version() + " refharness " + what;
}

// Extra operators by name: random_H / random_A (UnitTest::RandomOp, rank-2 scalar,
// seed via key random_seed), anything else via imsrg_util::OperatorFromString (e.g. Rp2).
Operator make_named_operator(ModelSpace& ms, const std::string& name, const Args& a) {
  if (name == "random_H" || name == "random_A") {
    UnitTest ut(ms);
    ut.SetRandomSeed(static_cast<uint64_t>(a.integer("random_seed", 1)));
    return ut.RandomOp(ms, 0, 0, 0, 2, name == "random_H" ? +1 : -1);
  }
  return imsrg_util::OperatorFromString(ms, name);
}

std::vector<std::string> split_commas(const std::string& s) {
  std::vector<std::string> v;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ','))
    if (!t.empty()) v.push_back(t);
  return v;
}

}  // namespace

int cmd_dump_hamiltonian(const Args& a) {
  SpaceParams sp = space_params(a);
  InteractionParams ip = interaction_params(a);
  auto msp = make_modelspace(sp);
  ModelSpace& ms = *msp;
  const std::string out = a.str("out", ".");
  const std::string ref = reference_name(sp);
  double t0 = omp_get_wtime();

  // Pieces without 3N (the 3N is handled separately below)
  InteractionParams ip2 = ip;
  ip2.file3 = "none";
  BareH b = make_bare_hamiltonian(ms, ip2);

  Json norms;
  write_opdump(out + "/T.dump", b.T, {source_text("dump-hamiltonian T = imsrg_util::Trel_Op"), ref});
  norms.raw("T", norms_json(b.T));
  write_opdump(out + "/V2.dump", b.V2, {source_text("dump-hamiltonian V2 = me2j " + ip.file2), ref});
  norms.raw("V2", norms_json(b.V2));
  write_opdump(out + "/H.dump", b.H, {source_text("dump-hamiltonian H = T + V2 (bare, HO basis)"), ref});
  norms.raw("H", norms_json(b.H));

  if (ip.file3 != "none") {
    // 3N alone in a rank-3 operator (no2b storage), normal ordered w.r.t. the reference
    Operator V3(ms, 0, 0, 0, 3);
    V3.SetHermitian();
    V3.ThreeBody.SetMode(ip.no2b_precision == "half" ? "no2bhalf" : "no2b");
    V3.ThreeBody.ReadFile({ip.file3}, {ip.file3e1max, ip.file3e2max, ip.file3e3max, ip.file3e1max});
    Operator V3no = V3.DoNormalOrdering();
    write_opdump(out + "/V3_NO2B.dump", V3no,
                 {source_text("dump-hamiltonian V3_NO2B = (3N only).DoNormalOrdering(), HO basis, " +
                              ip.file3),
                  ref});
    norms.raw("V3_NO2B", norms_json(V3no));
  }

  for (const auto& name : split_commas(a.str("ops", ""))) {
    Operator op = make_named_operator(ms, name, a);
    write_opdump(out + "/" + name + ".dump", op,
                 {source_text("dump-hamiltonian operator " + name + " (HO basis)"), ref});
    norms.raw(name, norms_json(op));
  }
  norms.num("wall_time_s", omp_get_wtime() - t0);
  norms.write(out + "/norms.json");
  return 0;
}

int cmd_hf(const Args& a) {
  SpaceParams sp = space_params(a);
  InteractionParams ip = interaction_params(a);
  auto msp = make_modelspace(sp);
  ModelSpace& ms = *msp;
  const std::string out = a.str("out", ".");
  const std::string ref = reference_name(sp);
  double t0 = omp_get_wtime();

  BareH b = make_bare_hamiltonian(ms, ip);
  // imsrg++.cc:560-567, 658-668 (freeze_occupations=false, basis=HF, no2b -> rank 2)
  HFMBPT hf(b.H);
  hf.UnFreezeOccupations();
  hf.Solve();
  Operator HNO = hf.GetNormalOrderedH(2);

  double emp2 = HNO.GetMP2_Energy();
  std::array<double, 3> emp3 = HNO.GetMP3_Energy();

  Json j;
  j.str("source", source_text("hf"));
  j.integer("emax", ms.GetEmax());
  j.str("reference", ref);
  j.str("2bme", ip.file2);
  j.str("3bme", ip.file3);
  j.num("hw", ms.GetHbarOmega());
  j.num("EHF", hf.EHF);
  j.num("e1hf", hf.e1hf);
  j.num("e2hf", hf.e2hf);
  j.num("e3hf", hf.e3hf);
  j.integer("iterations", hf.iterations);
  j.num("HNO_zero_body", HNO.ZeroBody);
  j.num("EMP2", emp2);
  j.num("EMP3_pp", emp3[0]);
  j.num("EMP3_hh", emp3[1]);
  j.num("EMP3_ph", emp3[2]);
  j.num("EMP3", emp3[0] + emp3[1] + emp3[2]);
  std::vector<double> spe(hf.energies.begin(), hf.energies.end());
  std::vector<double> occ;
  std::vector<long long> n, l, j2, tz2;
  for (size_t i = 0; i < ms.GetNumberOrbits(); ++i) {
    const Orbit& oi = ms.GetOrbit(i);
    occ.push_back(oi.occ);
    n.push_back(oi.n); l.push_back(oi.l); j2.push_back(oi.j2); tz2.push_back(oi.tz2);
  }
  j.ints("orbit_n", n).ints("orbit_l", l).ints("orbit_2j", j2).ints("orbit_2tz", tz2);
  j.nums("spe", spe);
  j.nums("occupation", occ);
  j.raw("C", matrix_json(hf.C));  // C(ho, hf): column k = HF orbit k in the HO basis
  j.raw("HNO_norms", norms_json(HNO));
  j.num("wall_time_s", omp_get_wtime() - t0);
  j.write(out + "/hf.json");

  if (a.flag("dump_hno"))
    write_opdump(out + "/HNO.dump", HNO,
                 {source_text("hf HNO = HartreeFock::GetNormalOrderedH(2), HF basis" +
                              std::string(ip.file3 != "none" ? ", 2N+NO2B 3N" : ", 2N only")),
                  ref});
  return 0;
}

}  // namespace rh
