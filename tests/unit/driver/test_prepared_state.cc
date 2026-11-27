// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// PreparationHash (which options a stored prepared Hamiltonian may be reused
// for, design/00 D75) and the termination flag of the SIGTERM handler.
#include <csignal>
#include <cstdint>

#include <gtest/gtest.h>

#include "imsrg/driver/config/schema.h"
#include "imsrg/driver/pipeline/checkpoint.h"
#include "imsrg/driver/pipeline/prepared_state.h"
#include "imsrg/driver/termination.h"

namespace imsrg::driver {
namespace {

/// A small resolved-looking configuration (the hash only serializes it).
RunConfig BaseConfig() {
  RunConfig c;
  c.basis.emax = 4;
  c.basis.hw = 16.0;
  c.basis.mass_number = 27;
  c.reference.nucleus = "Al27";
  c.valence.space = "sd-shell";
  c.interaction.two_body.file = {"/inputs/vnn_a.me2j.gz"};
  c.interaction.two_body.e1max = 14;
  c.interaction.two_body.e2max = 28;
  c.basis.kind = BasisKind::kNat;
  c.imsrg.space = ImsrgSpaceConfig{};
  c.imsrg.space->emax = 2;
  c.imsrg.space->e2max = 4;
  return c;
}

TEST(PreparationHash, IgnoresFlowAndOutputOptions) {
  const RunConfig base = BaseConfig();
  const std::uint64_t h = pipeline::PreparationHash(base);
  RunConfig c = base;
  c.imsrg.step.omega_norm_max = 0.1;
  c.imsrg.step.domega = 0.5;
  c.imsrg.step.eta_criterion = 1.0e-3;
  c.imsrg.step.smax = 50.0;
  c.imsrg.generator.core = Generator::kImaginaryTime;
  c.imsrg.method = Method::kFlowRk4;
  c.imsrg.steps = 1;
  c.imsrg.perturbative_triples = true;
  c.output.directory = "/elsewhere";
  c.output.write_omega = true;
  c.checkpoint.directory = "/ckpt";
  c.checkpoint.restart = true;
  c.checkpoint.prepared_from = "/other";
  c.operators.list.push_back(OperatorRequest{});
  c.run.log_level = LogLevel::kDebug;
  c.run.task = Task::kHf;
  EXPECT_EQ(pipeline::PreparationHash(c), h);
  // The flow hash does see the step options.
  EXPECT_NE(pipeline::FlowConfigHash(c), pipeline::FlowConfigHash(base));
}

TEST(PreparationHash, SeesTheHamiltonian) {
  const RunConfig base = BaseConfig();
  const std::uint64_t h = pipeline::PreparationHash(base);
  RunConfig c = base;
  c.basis.emax = 6;
  EXPECT_NE(pipeline::PreparationHash(c), h);
  c = base;
  c.interaction.two_body.file = {"/inputs/vnn_b.me2j.gz"};
  EXPECT_NE(pipeline::PreparationHash(c), h);
  c = base;
  c.imsrg.space->emax = 3;
  c.imsrg.space->e2max = 6;
  EXPECT_NE(pipeline::PreparationHash(c), h);
  c = base;
  c.basis.kind = BasisKind::kHf;
  EXPECT_NE(pipeline::PreparationHash(c), h);
  c = base;
  c.hartree_fock.mixing = HartreeFockMixing::kLinear == base.hartree_fock.mixing
                              ? HartreeFockMixing::kDiis
                              : HartreeFockMixing::kLinear;
  EXPECT_NE(pipeline::PreparationHash(c), h);
  c = base;
  c.natural_orbitals.use_hf_reference = !base.natural_orbitals.use_hf_reference;
  EXPECT_NE(pipeline::PreparationHash(c), h);
  c = base;
  c.run.compat = Compat::kImsrgFork79e9330;
  EXPECT_NE(pipeline::PreparationHash(c), h);
}

TEST(Termination, FlagIsSetByTheHandlerAndCleared) {
  ClearTerminationRequest();
  EXPECT_FALSE(TerminationRequested());
  RequestTermination();
  EXPECT_TRUE(TerminationRequested());
  ClearTerminationRequest();
  EXPECT_FALSE(TerminationRequested());
  InstallTerminationHandler();
  std::raise(SIGTERM);  // handled: the process continues
  EXPECT_TRUE(TerminationRequested());
  RestoreDefaultTerminationHandler();
  ClearTerminationRequest();
  EXPECT_FALSE(TerminationRequested());
}

}  // namespace
}  // namespace imsrg::driver
