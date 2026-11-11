// Particle-hole (Pandya) term: stub until task T4.

#include "imsrg/base/error.hpp"
#include "imsrg/commutator/commutator.hpp"
#include "imsrg/commutator/detail/workspace_impl.hpp"

namespace imsrg::commutator::terms {

void comm222_phss(const op::Operator& X, const op::Operator& Y, op::Operator& Z, Workspace& ws) {
  detail::check_term_inputs(X, Y, Z, ws);
  throw NotImplementedError("comm222_phss: lands in task T3b/T4");
}

}  // namespace imsrg::commutator::terms
