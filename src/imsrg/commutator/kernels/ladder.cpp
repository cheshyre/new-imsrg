// Ladder (pp/hh) terms: stubs until task T3b.

#include "imsrg/base/error.hpp"
#include "imsrg/commutator/commutator.hpp"
#include "imsrg/commutator/detail/workspace_impl.hpp"

namespace imsrg::commutator::terms {

void comm222_pp_hhss(const op::Operator& X, const op::Operator& Y, op::Operator& Z, Workspace& ws) {
  detail::check_term_inputs(X, Y, Z, ws);
  throw NotImplementedError("comm222_pp_hhss: lands in task T3b/T4");
}

void comm221ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z, Workspace& ws) {
  detail::check_term_inputs(X, Y, Z, ws);
  throw NotImplementedError("comm221ss: lands in task T3b/T4");
}

void comm222_pp_hh_221ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
                         Workspace& ws) {
  detail::check_term_inputs(X, Y, Z, ws);
  throw NotImplementedError("comm222_pp_hh_221ss: lands in task T3b/T4");
}

}  // namespace imsrg::commutator::terms
