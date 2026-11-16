// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The distributed commutator on real MPI ranks (M12b phase 1, D14; run with 2
// and 4 ranks): `Commutator` on a workspace over the test world against the
// same call on a serial workspace, for every IMSRG(2) term alone and fused, the
// N7 terms with the three-body gate open and closed, workspace reuse
// interleaved with the replicated 3f2 functions, and one BCH transform. Every
// case checks max|dZ| <= 1e-12 max(1, max|Z_serial|) per operator part and
// that all ranks hold identical bits (a 64-bit FNV-1a hash is gathered).

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/comm/communicator.h"
#include "imsrg/commutator/commutator.h"
#include "imsrg/commutator/factorized.h"
#include "imsrg/commutator/options.h"
#include "imsrg/commutator/workspace.h"
#include "imsrg/flow/bch.h"
#include "imsrg/operator/arithmetic.h"
#include "imsrg/operator/compare.h"
#include "imsrg/operator/norms.h"
#include "imsrg/operator/operator.h"
#include "tests/support/commutator_test_support.h"
#include "tests/support/mpi_test_main.h"
#include "tests/unit/commutator/n7_test_support.h"

namespace imsrg::commutator {
namespace {

constexpr double kTol = 1.0e-12;  // per part, relative to max(1, max|Z_ser|)

using ModelSpacePtr = std::shared_ptr<const basis::ModelSpace>;

const comm::Communicator& World() { return test_support::TestWorld(); }

/// Overwrites the data of `o` on every rank with rank 0's, so all ranks start
/// from identical bits whatever the operand generator does.
void BroadcastOperator(op::Operator& o) {
  const comm::Communicator& w = World();
  o.SyncToDevice();
  double z0 = o.zero_body();
  w.Broadcast(std::span<double>(&z0, 1), 0);
  o.zero_body() = z0;
  w.Broadcast(o.one_body().buffer().host_span(), 0);
  w.Broadcast(o.two_body().matrix().buffer().host_span(), 0);
  if (o.three_body().has_value()) {
    w.Broadcast(o.three_body()->host_span(), 0);
  }
  o.SyncToDevice();
}

void HashDoubles(std::span<const double> v, std::uint64_t& h) {
  for (const double x : v) {
    h ^= std::bit_cast<std::uint64_t>(x);
    h *= 1099511628211ULL;
  }
}

/// 64-bit FNV-1a (over whole words) of the zero-body value and all buffers.
std::int64_t Fingerprint(const op::Operator& z) {
  z.SyncToHost();
  std::uint64_t h = 14695981039346656037ULL;
  const double z0 = z.zero_body();
  HashDoubles(std::span<const double>(&z0, 1), h);
  HashDoubles(z.one_body().buffer().host_span(), h);
  HashDoubles(z.two_body().matrix().buffer().host_span(), h);
  if (z.three_body().has_value()) HashDoubles(z.three_body()->host_span(), h);
  return std::bit_cast<std::int64_t>(h);
}

void ExpectIdenticalOnAllRanks(const op::Operator& z,
                               const std::string& label) {
  const std::vector<std::int64_t> all = World().Allgather(Fingerprint(z));
  for (const std::int64_t f : all) {
    EXPECT_EQ(f, all.front()) << label << ": ranks hold different bits";
  }
}

/// Checks z against z_ser per part; returns the largest max|dZ| over parts.
double ExpectNear(const op::Operator& z, const op::Operator& z_ser,
                  const std::string& label) {
  const op::PartMaxima d = op::MaxAbsDifference(z, z_ser);
  const op::PartMaxima m = op::MaxAbs(z_ser);
  EXPECT_LE(d.zero_body, kTol * std::max(1.0, m.zero_body)) << label << " 0B";
  EXPECT_LE(d.one_body, kTol * std::max(1.0, m.one_body)) << label << " 1B";
  EXPECT_LE(d.two_body, kTol * std::max(1.0, m.two_body)) << label << " 2B";
  EXPECT_LE(d.three_body, kTol * std::max(1.0, m.three_body)) << label << " 3B";
  return d.max();
}

/// Compares, reports on rank 0 and checks the bits across ranks.
void CheckAgainstSerial(const op::Operator& z, const op::Operator& z_ser,
                        const std::string& label, const char* group) {
  const double dz = ExpectNear(z, z_ser, label);
  ExpectIdenticalOnAllRanks(z, label);
  if (World().is_root()) {
    std::printf("[%s] %-44s max|dZ| = %.3e\n", group, label.c_str(), dz);
  }
}

/// One commutator on the world and on the serial workspace, compared.
void CheckCommutator(const op::Operator& X, const op::Operator& Y,
                     const CommutatorOptions& opts, Workspace& ws_world,
                     Workspace& ws_serial, const std::string& label,
                     const char* group) {
  const op::Operator z = Commutator(X, Y, opts, ws_world);
  const op::Operator z_ser = Commutator(X, Y, opts, ws_serial);
  CheckAgainstSerial(z, z_ser, label, group);
}

struct Mask {
  const char* name;
  TermMask mask;
};

std::vector<Mask> Imsrg2Masks() {
  return {
      {"imsrg2", TermMask::Imsrg2()},
      {"110", TermMask::Only(Term::kComm110ss)},
      {"220", TermMask::Only(Term::kComm220ss)},
      {"111", TermMask::Only(Term::kComm111ss)},
      {"121", TermMask::Only(Term::kComm121ss)},
      {"122", TermMask::Only(Term::kComm122ss)},
      {"221", TermMask::Only(Term::kComm221ss)},
      {"222pphh", TermMask::Only(Term::kComm222PpHhss)},
      {"222pphh+221",
       TermMask::Only(Term::kComm222PpHhss) | TermMask::Only(Term::kComm221ss)},
      {"222ph", TermMask::Only(Term::kComm222Phss)},
  };
}

struct OperandPair {
  const char* name;
  Hermiticity hx;
  Hermiticity hy;
};

const OperandPair kPairs[] = {
    {"H_A", Hermiticity::kHermitian, Hermiticity::kAntiHermitian},
    {"H_H", Hermiticity::kHermitian, Hermiticity::kHermitian},
    {"A_A", Hermiticity::kAntiHermitian, Hermiticity::kAntiHermitian},
    {"N_N", Hermiticity::kNonHermitian, Hermiticity::kNonHermitian},
};

op::Operator MakeBroadcast(const ModelSpacePtr& ms, Hermiticity h,
                           std::uint64_t seed) {
  op::Operator o = testing::RandomOperator(ms, h, seed);
  BroadcastOperator(o);
  return o;
}

/// All operand pairs with all masks (or only the full IMSRG(2) mask).
void RunMasks(const ModelSpacePtr& ms, const std::string& space,
              bool all_masks) {
  Workspace ws_world(ms, WorkspaceOptions{World()});
  Workspace ws_serial(ms);
  std::uint64_t seed = 100;
  for (const OperandPair& p : kPairs) {
    const op::Operator X = MakeBroadcast(ms, p.hx, seed++);
    const op::Operator Y = MakeBroadcast(ms, p.hy, seed++);
    for (const Mask& m : Imsrg2Masks()) {
      if (!all_masks && m.mask != TermMask::Imsrg2()) continue;
      CommutatorOptions opts;
      opts.terms = m.mask;
      CheckCommutator(X, Y, opts, ws_world, ws_serial,
                      space + " " + p.name + " " + m.name, "masks");
    }
  }
}

TEST(CommutatorMpi, MasksHe4Emax1) {
  RunMasks(testing::MakeModelSpace(1, "He4"), "He4e1", true);
}

TEST(CommutatorMpi, MasksHe4Emax2) {
  RunMasks(testing::MakeModelSpace(2, "He4"), "He4e2", true);
}

TEST(CommutatorMpi, MasksHe6Emax2Fractional) {
  RunMasks(testing::FractionalModelSpace(2, "He6"), "He6e2frac", true);
}

TEST(CommutatorMpi, MasksO16Emax2) {
  RunMasks(testing::MakeModelSpace(2, "O16"), "O16e2", true);
}

TEST(CommutatorMpi, MasksO16Emax3) {
  RunMasks(testing::MakeModelSpace(3, "O16"), "O16e3", true);
}

/// The N7 commutator with `threshold` on both workspaces, compared.
void CheckN7(const testing::N7Inputs& in, double threshold, Workspace& ws_world,
             Workspace& ws_serial, const std::string& label) {
  CommutatorOptions opts;
  opts.terms = TermMask::Imsrg3N7();
  opts.threebody_threshold = threshold;
  CheckCommutator(in.x, in.y, opts, ws_world, ws_serial, label, "n7");
}

/// Z3 of the N7 commutator with `threshold` on the world workspace.
double ThreeBodyNormOf(const testing::N7Inputs& in, double threshold,
                       Workspace& ws) {
  CommutatorOptions opts;
  opts.terms = TermMask::Imsrg3N7();
  opts.threebody_threshold = threshold;
  return op::ThreeBodyNorm(Commutator(in.x, in.y, opts, ws));
}

void RunN7(const char* reference) {
  testing::N7Inputs in = testing::GoldenN7Inputs(2, reference, 3);
  BroadcastOperator(in.x);
  BroadcastOperator(in.y);
  Workspace ws_world(in.ms3, WorkspaceOptions{World()});
  Workspace ws_serial(in.ms3);
  const std::string name = std::string(reference) + "e2 ";
  const double lo = std::min(op::Norm(in.x), op::Norm(in.y));
  const double hi = std::max(op::Norm(in.x), op::Norm(in.y));
  CheckN7(in, 0.0, ws_world, ws_serial, name + "N7 threshold 0");
  CheckN7(in, 0.5 * lo, ws_world, ws_serial, name + "N7 gate open");
  CheckN7(in, 2.0 * hi, ws_world, ws_serial, name + "N7 gate closed");
  // The closed gate leaves Z3 zero, the open ones do not.
  EXPECT_GT(ThreeBodyNormOf(in, 0.5 * lo, ws_world), 0.0);
  EXPECT_EQ(ThreeBodyNormOf(in, 2.0 * hi, ws_world), 0.0);
}

TEST(CommutatorMpi, N7He4Emax2) { RunN7("He4"); }

TEST(CommutatorMpi, N7O16Emax2) { RunN7("O16"); }

/// F(eta, gamma) (the replicated 3f2 double commutator) on both workspaces.
void CheckFactorized(const op::Operator& eta, const op::Operator& gamma,
                     Workspace& ws_world, Workspace& ws_serial,
                     const std::string& label) {
  op::Operator z;
  op::Operator z_ser;
  FactorizedDoubleCommutator(eta, gamma, z, FactorizedOptions{}, ws_world);
  FactorizedDoubleCommutator(eta, gamma, z_ser, FactorizedOptions{}, ws_serial);
  CheckAgainstSerial(z, z_ser, label, "reuse");
}

/// F(prepared eta, gamma) on both workspaces.
void CheckPrepared(const FactorizedEta& pre, const FactorizedEta& pre_ser,
                   const op::Operator& gamma, Workspace& ws_world,
                   Workspace& ws_serial, const std::string& label) {
  op::Operator z;
  op::Operator z_ser;
  FactorizedDoubleCommutator(pre, gamma, z, FactorizedOptions{}, ws_world);
  FactorizedDoubleCommutator(pre_ser, gamma, z_ser, FactorizedOptions{},
                             ws_serial);
  CheckAgainstSerial(z, z_ser, label, "reuse");
}

/// Commutator into a reused Z on both workspaces, compared.
void CheckReusedZ(const op::Operator& X, const op::Operator& Y, op::Operator& z,
                  op::Operator& z_ser, Workspace& ws_world,
                  Workspace& ws_serial, const std::string& label) {
  const CommutatorOptions opts;
  Commutator(X, Y, z, opts, ws_world);
  Commutator(X, Y, z_ser, opts, ws_serial);
  CheckAgainstSerial(z, z_ser, label, "reuse");
}

void RunReuse(const ModelSpacePtr& ms, const std::string& space) {
  Workspace ws_world(ms, WorkspaceOptions{World()});
  Workspace ws_serial(ms);
  const op::Operator eta = MakeBroadcast(ms, Hermiticity::kAntiHermitian, 7);
  const op::Operator eta2 = MakeBroadcast(ms, Hermiticity::kAntiHermitian, 8);
  const op::Operator h = MakeBroadcast(ms, Hermiticity::kHermitian, 9);
  op::Operator z;
  op::Operator z_ser;
  const std::string s = space + " ";
  CheckReusedZ(eta, h, z, z_ser, ws_world, ws_serial, s + "[eta,H] 1");
  CheckFactorized(eta, h, ws_world, ws_serial, s + "F(eta,H) 1");
  CheckReusedZ(h, h, z, z_ser, ws_world, ws_serial, s + "[H,H] 2");
  CheckFactorized(eta2, h, ws_world, ws_serial, s + "F(eta2,H) 2");
  CheckReusedZ(eta, eta2, z, z_ser, ws_world, ws_serial, s + "[eta,eta2] 3");
  // A prepared eta reused for several gammas, between commutators.
  const FactorizedEta pre(eta, ws_world);
  const FactorizedEta pre_ser(eta, ws_serial);
  CheckPrepared(pre, pre_ser, h, ws_world, ws_serial, s + "F(pre eta,H)");
  CheckReusedZ(eta, h, z, z_ser, ws_world, ws_serial, s + "[eta,H] 4");
  CheckPrepared(pre, pre_ser, eta2, ws_world, ws_serial, s + "F(pre eta,eta2)");
  CheckReusedZ(h, eta, z, z_ser, ws_world, ws_serial, s + "[H,eta] 5");
  CheckFactorized(eta, eta2, ws_world, ws_serial, s + "F(eta,eta2) 3");
}

TEST(CommutatorMpi, ReuseHe4Emax2) {
  RunReuse(testing::MakeModelSpace(2, "He4"), "He4e2");
}

TEST(CommutatorMpi, ReuseO16Emax2) {
  RunReuse(testing::MakeModelSpace(2, "O16"), "O16e2");
}

TEST(CommutatorMpi, BchTransformHe4Emax2) {
  const ModelSpacePtr ms = testing::MakeModelSpace(2, "He4");
  Workspace ws_world(ms, WorkspaceOptions{World()});
  Workspace ws_serial(ms);
  const op::Operator omega = MakeBroadcast(ms, Hermiticity::kAntiHermitian, 31);
  const op::Operator h = MakeBroadcast(ms, Hermiticity::kHermitian, 32);
  // A small Omega keeps the series short and convergent.
  const op::Operator small = (0.05 / std::max(1.0, op::Norm(omega))) * omega;
  const flow::BchOptions bopts;
  const CommutatorOptions copts;
  const op::Operator z = flow::BchTransform(h, small, bopts, copts, ws_world);
  const op::Operator z_ser =
      flow::BchTransform(h, small, bopts, copts, ws_serial);
  CheckAgainstSerial(z, z_ser, "He4e2 BchTransform", "bch");
}

}  // namespace
}  // namespace imsrg::commutator
