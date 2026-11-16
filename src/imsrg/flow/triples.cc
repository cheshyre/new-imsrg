// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/flow/triples.h"

#include <cstddef>
#include <memory>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/basis/three_body_channels.h"
#include "imsrg/basis/three_body_space.h"
#include "imsrg/commutator/three_body.h"

namespace imsrg::flow {

op::Operator CombineOmegaStack(const std::vector<op::Operator>& omegas,
                               const BchOptions& options,
                               commutator::Workspace& workspace) {
  IMSRG_CHECK(!omegas.empty(), "CombineOmegaStack: empty Omega stack");
  op::Operator omega = omegas.front();
  for (std::size_t n = 1; n < omegas.size(); ++n) {
    omega = BchProduct(omegas[n], omega, options,
                       commutator::CommutatorOptions{}, workspace);
  }
  return omega;
}

TriplesResult PerturbativeTriples(MagnusSolver& solver,
                                  const TriplesOptions& options) {
  const std::vector<op::Operator>& omegas = solver.omegas();
  // Transformation order (D49, M10-Q40): not imsrg++'s BchProduct(omega, n).
  const op::Operator omega = CombineOmegaStack(
      omegas, solver.bch_options(), solver.workspace());

  BchOptions skip = solver.bch_options();
  skip.skip_first_commutator = true;
  const op::Operator htilde =
      solver.Transform(solver.initial_hamiltonian(), skip);

  op::Operator h = solver.hamiltonian();
  h.SyncToHost();
  const std::shared_ptr<const basis::ModelSpace> ms = h.model_space_ptr();
  const std::int32_t num_orbits = ms->num_orbits();
  std::vector<double> epsilon(static_cast<std::size_t>(num_orbits));
  for (std::int32_t a = 0; a < num_orbits; ++a) {
    epsilon[static_cast<std::size_t>(a)] =
        h.one_body().get(OrbitIndex{a}, OrbitIndex{a});
  }

  basis::ThreeBodyCuts cuts;
  cuts.e3max = 3 * ms->basis().emax();
  cuts.emax_3body = -1;
  cuts.de3max = options.de3max;
  cuts.occnat3_cut = options.occnat3_cut;
  const std::shared_ptr<const basis::ModelSpace> ms3 = ms->WithThreeBody(cuts);
  const commutator::ThreeBodyContext ctx(ms3);

  TriplesResult result;
  result.de =
      commutator::PerturbativeTriplesEnergy(omega, htilde, epsilon, ctx);
  result.kept_rows = ms3->three_body_space()->NumRows();
  result.total_kets = ms3->three_body()->SumChannelKets();
  return result;
}

}  // namespace imsrg::flow
