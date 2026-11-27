// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/driver/pipeline/operator_frame.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/base/error.h"
#include "imsrg/driver/config/compat.h"
#include "imsrg/io/tensor_writers.h"
#include "imsrg/operator/basis_transform.h"
#include "imsrg/operator/truncate.h"
#include "imsrg/opslib/center_of_mass.h"
#include "imsrg/opslib/kinetic.h"
#include "imsrg/opslib/nreft.h"
#include "imsrg/opslib/radius.h"

namespace imsrg::driver::pipeline {
namespace {

[[noreturn]] void Planned(const std::string& what) {
  throw ConfigError("E-PLANNED: " + what + " is not implemented yet");
}

}  // namespace

void ApplyCompatCommutatorOptions(const RunConfig& config,
                                  commutator::CommutatorOptions& copts) {
  const CompatOptions c = CompatOptionsFor(config.run.compat);
  if (c.tensor_ninej_threshold_1e8) copts.tensor_ninej_threshold = 1.0e-8;
  copts.tensor_fork_ph_missing_rank_phase = c.tensor_fork_ph_missing_rank_phase;
}

std::vector<double> CoreOccupations(const basis::ModelSpace& ms) {
  std::vector<double> occ(static_cast<std::size_t>(ms.num_orbits()), 0.0);
  for (const OrbitIndex a : ms.partition().core()) occ[a.ToSize()] = 1.0;
  return occ;
}

bool NeedsRenormalOrdering(const basis::ModelSpace& ms) {
  if (ms.partition().is_single_reference()) return false;
  const std::vector<double> core = CoreOccupations(ms);
  const auto occ = ms.reference().occupations();
  for (std::size_t a = 0; a < core.size(); ++a) {
    if (std::abs(occ[a] - core[a]) > 1.0e-12) return true;
  }
  return false;
}

OperatorFrame MakeOperatorFrame(const PreparedHamiltonian& ph,
                                const RunConfig& config) {
  OperatorFrame f;
  f.ms_bare = ph.ms_bare;
  if (ph.nat) {
    f.c = ph.nat->C;
  } else if (ph.hf) {
    f.c = ph.hf->C;
  }
  f.ms_imsrg = ph.hno.model_space_ptr();
  f.ms_normal_ordering =
      ph.ms_normal_ordering ? ph.ms_normal_ordering : f.ms_imsrg;
  if (NeedsRenormalOrdering(*f.ms_imsrg)) {
    f.ms_core = f.ms_imsrg->WithReference(basis::Reference::FromOccupations(
        f.ms_imsrg->basis(), CoreOccupations(*f.ms_imsrg), "core"));
  }
  f.mass_number = ph.mass_number;
  f.proton_number = ph.proton_number;
  f.tensor_no.fork79e9330_diagonal_bug =
      CompatOptionsFor(config.run.compat).tensor_no_fork_diagonal_bug;
  return f;
}

op::Operator BuildBareOperator(const OperatorRequest& request,
                               const OperatorFrame& frame) {
  const auto& ms = frame.ms_bare;
  const int A = frame.mass_number;
  const int Z = frame.proton_number;
  switch (request.kind) {
    case OperatorKind::kNreft: {
      const auto s = opslib::nreft::ParseShorthand(request.canonical_name);
      IMSRG_CHECK(s.has_value());
      // One-body only (particle rank 1): the operator has no two-body part
      // before the BCH (which promotes it to rank 2 at emax_imsrg), and a
      // zero two-body part at basis.emax 14 would cost up to 20 GB for a
      // rank-5 tensor (design/50 section 6). The results are the same bits.
      return opslib::nreft::BuildShorthandOperator(ms, *s, 1);
    }
    case OperatorKind::kRp2:
      return opslib::Rp2(ms, A, Z);
    case OperatorKind::kRso:
      return opslib::RpSpinOrbitCorrection(ms);
    case OperatorKind::kRn2:
      return opslib::Rn2(ms, A, Z);
    case OperatorKind::kRm2:
      return opslib::Rm2(ms, A);
    case OperatorKind::kTcm:
      return opslib::TCm(ms, A);
    case OperatorKind::kTrel:
      return opslib::TRel(ms, A);
    case OperatorKind::kTlab:
      return opslib::KineticEnergy(ms);
    case OperatorKind::kHcm: {
      std::optional<double> hw;
      if (!request.params.empty()) {
        if (const auto* d = std::get_if<double>(&request.params.front())) {
          hw = *d;
        } else if (const auto* i =
                       std::get_if<std::int64_t>(&request.params.front())) {
          hw = static_cast<double>(*i);
        }
      }
      return opslib::HCm(ms, A, hw);
    }
    default:
      Planned("operator '" + request.canonical_name + "'");
  }
}

op::Operator PrepareBareOperator(const op::Operator& bare,
                                 const OperatorFrame& frame) {
  op::Operator o = bare;
  if (frame.c) o = op::TransformBasis(o, frame.c->const_view());
  if (frame.ms_normal_ordering.get() != o.model_space_ptr().get()) {
    o = op::WithModelSpace(o, frame.ms_normal_ordering);
  }
  o = o.spec().is_scalar() ? op::NormalOrder(o)
                           : op::NormalOrderTensor(o, frame.tensor_no);
  // [imsrg.space]: normal ordered in the full space, then cut (imsrg++.cc:
  // 1427-1475).
  if (frame.ms_imsrg.get() != frame.ms_normal_ordering.get()) {
    o = op::Truncate(o, frame.ms_imsrg);
  }
  return o;
}

op::Operator PrepareOperator(const OperatorRequest& request,
                             const OperatorFrame& frame) {
  return PrepareBareOperator(BuildBareOperator(request, frame), frame);
}

op::Operator FinishOperator(const op::Operator& transformed,
                            const OperatorFrame& frame) {
  op::Operator o = transformed;
  if (o.spec().particle_rank >= 3) {
    // imsrg++.cc:1177-1180, 1256-1260 throw the residual 3-body part away.
    o.SyncToHost();
    op::OperatorSpec spec = o.spec();
    spec.particle_rank = 2;
    op::Operator two(o.model_space_ptr(), spec);
    two.set_hermiticity(o.hermiticity());
    two.zero_body() = o.zero_body();
    two.one_body() = o.one_body();
    two.two_body() = o.two_body();
    two.SyncToDevice();
    o = std::move(two);
  }
  if (frame.ms_core) o = op::ReNormalOrder(o, frame.ms_core, frame.tensor_no);
  return o;
}

io::ValenceWriteOptions MakeValenceWriteOptions(const RunConfig& c,
                                                const OperatorFrame& frame,
                                                io::ValenceWriteMode mode) {
  io::ValenceWriteOptions o;
  o.mode = mode;
  o.file_2n = c.interaction.two_body.file.empty()
                  ? ""
                  : c.interaction.two_body.file.front();
  o.file_3n =
      c.interaction.three_body ? c.interaction.three_body->file : "none";
  o.e1max = c.basis.emax;
  o.e2max = 2 * c.basis.emax;
  o.e3max = c.basis.e3max;
  o.hbar_omega = c.basis.hw;
  o.a_ref = static_cast<std::int32_t>(frame.ms_bare->reference().target_mass());
  o.z_ref = static_cast<std::int32_t>(frame.ms_bare->reference().target_Z());
  o.a_target = static_cast<double>(frame.mass_number);
  if (CompatOptionsFor(c.run.compat).tokyo_writer_fork_79e9330) {
    // The fork writes the core's A and Z and, as a label artifact of
    // SetReference(core), the core mass as "A_for_kinetic_energy".
    const basis::Partition& p = frame.ms_imsrg->partition();
    o.a_ref = static_cast<std::int32_t>(std::lround(p.A_core()));
    o.z_ref = static_cast<std::int32_t>(std::lround(p.Z_core()));
    o.a_target = static_cast<double>(o.a_ref);
    o.tokyo_format = io::TokyoNumberFormat::kFork79e9330;
  }
  o.version =
      c.provenance.version.empty() ? "imsrg" : "imsrg " + c.provenance.version;
  return o;
}

std::string OperatorFileName(const std::string& stem, const std::string& name,
                             bool tensor, bool fork_names) {
  return stem + (fork_names ? "" : "_") + name + (tensor ? "_2b" : "") +
         ".snt";
}

std::string OperatorFileName(const std::string& stem, const std::string& name,
                             const op::OperatorSpec& spec,
                             const RunConfig& config) {
  return OperatorFileName(
      stem, name, !spec.is_scalar(),
      CompatOptionsFor(config.run.compat).fork_operator_file_names);
}

void WriteTensorSnt(const std::filesystem::path& path, op::Operator O,
                    const RunConfig& config, const OperatorFrame& frame) {
  O.SyncToHost();
  const io::ValenceWriteOptions v = MakeValenceWriteOptions(
      config, frame, io::ValenceWriteMode::kHamiltonian);
  io::TensorTokyoOptions t;
  t.format = CompatOptionsFor(config.run.compat).tokyo_writer_fork_79e9330
                 ? io::TokyoFormat::kFork79e9330
                 : io::TokyoFormat::kDevel;
  t.file_2n = v.file_2n;
  t.file_3n = v.file_3n;
  t.e1max = v.e1max;
  t.e2max = v.e2max;
  t.e3max = v.e3max;
  t.hbar_omega = v.hbar_omega;
  t.a_ref = v.a_ref;
  t.z_ref = v.z_ref;
  t.a_target = v.a_target;
  t.version = v.version;
  io::WriteTensorTokyo(path, O, t);
}

}  // namespace imsrg::driver::pipeline
