#include "imsrg/commutator/commutator.hpp"

#include <algorithm>
#include <string>

#include "imsrg/base/assert.hpp"
#include "imsrg/base/error.hpp"
#include "imsrg/commutator/detail/workspace_impl.hpp"

namespace imsrg::commutator {

namespace detail {

void check_term_inputs(const op::Operator& X, const op::Operator& Y, const op::Operator& Z,
                       const Workspace& ws) {
  IMSRG_CHECK(ws.compatible(X), "commutator: X is not on the workspace's model space");
  IMSRG_CHECK(ws.compatible(Y), "commutator: Y is not on the workspace's model space");
  IMSRG_CHECK(ws.compatible(Z), "commutator: Z is not on the workspace's model space");
  IMSRG_CHECK(&Z != &X && &Z != &Y, "commutator: Z must not alias X or Y");
  if (X.hermiticity() == Hermiticity::NonHermitian || Y.hermiticity() == Hermiticity::NonHermitian)
    throw NotImplementedError("commutator: NonHermitian X or Y is not supported");
  IMSRG_CHECK(Z.hermiticity() != Hermiticity::NonHermitian,
              "commutator: Z must be Hermitian or anti-Hermitian");
}

}  // namespace detail

op::OperatorSpec commutator_spec(const op::OperatorSpec& x, const op::OperatorSpec& y) {
  if (!x.is_scalar() || !y.is_scalar())
    throw NotImplementedError("commutator: only scalar operators are supported");
  if (!x.is_number_conserving() || !y.is_number_conserving())
    throw NotImplementedError("commutator: only number-conserving operators are supported");
  if (x.particle_rank > 2 || y.particle_rank > 2)
    throw NotImplementedError("commutator: particle rank 3 is not supported (IMSRG(3), M10)");
  op::OperatorSpec z = op::OperatorSpec::scalar(std::max(x.particle_rank, y.particle_rank));
  if (x.hermiticity == Hermiticity::NonHermitian || y.hermiticity == Hermiticity::NonHermitian) {
    z.hermiticity = Hermiticity::NonHermitian;
  } else {
    z.hermiticity =
        (x.hermiticity == y.hermiticity) ? Hermiticity::AntiHermitian : Hermiticity::Hermitian;
  }
  return z;
}

void commutator(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
                const CommutatorOptions& opts, Workspace& ws) {
  IMSRG_CHECK(ws.compatible(X), "commutator: X is not on the workspace's model space");
  IMSRG_CHECK(ws.compatible(Y), "commutator: Y is not on the workspace's model space");
  IMSRG_CHECK(&Z != &X && &Z != &Y, "commutator: Z must not alias X or Y");
  if (X.hermiticity() == Hermiticity::NonHermitian || Y.hermiticity() == Hermiticity::NonHermitian)
    throw NotImplementedError("commutator: NonHermitian X or Y is not supported");

  const op::OperatorSpec spec = commutator_spec(X.spec(), Y.spec());
  if (!Z.empty() && Z.spec() == spec && ws.compatible(Z)) {
    Z.set_zero();
  } else {
    Z = op::Operator(ws.model_space_ptr(), spec);
  }

  const TermMask m = opts.terms;
  if (m.test(Term::comm110ss)) terms::comm110ss(X, Y, Z, ws);
  if (m.test(Term::comm220ss)) terms::comm220ss(X, Y, Z, ws);
  if (m.test(Term::comm111ss)) terms::comm111ss(X, Y, Z, ws);
  if (m.test(Term::comm121ss)) terms::comm121ss(X, Y, Z, ws);
  if (m.test(Term::comm122ss)) terms::comm122ss(X, Y, Z, ws);
  if (m.test(Term::comm222_pp_hhss) && m.test(Term::comm221ss)) {
    terms::comm222_pp_hh_221ss(X, Y, Z, ws);
  } else {
    if (m.test(Term::comm221ss)) terms::comm221ss(X, Y, Z, ws);
    if (m.test(Term::comm222_pp_hhss)) terms::comm222_pp_hhss(X, Y, Z, ws);
  }
  if (m.test(Term::comm222_phss)) terms::comm222_phss(X, Y, Z, ws);
}

op::Operator commutator(const op::Operator& X, const op::Operator& Y, const CommutatorOptions& opts,
                        Workspace& ws) {
  op::Operator Z;
  commutator(X, Y, Z, opts, ws);
  return Z;
}

}  // namespace imsrg::commutator
