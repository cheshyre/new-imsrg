// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

/// @file
/// Bindings of the commutator module: Term, TermMask, CommutatorOptions,
/// the workspace, the commutator and the factorized double commutator
/// (design/18 section 2.6).

#include "imsrg/commutator/commutator.h"

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "imsrg/commutator/factorized.h"
#include "imsrg/commutator/options.h"
#include "python/src/bindings.h"

namespace imsrg::py {

namespace nb = nanobind;
namespace {

struct Cache {
  std::mutex mutex;
  std::vector<std::shared_ptr<PyWorkspace>> entries;
};

Cache& GetCache() {
  static Cache* c = new Cache;
  return *c;
}

constexpr std::size_t kCacheSize = 4;

}  // namespace

std::shared_ptr<PyWorkspace> WorkspaceFor(
    const std::shared_ptr<const basis::ModelSpace>& ms) {
  EnsureRuntime();
  auto& c = GetCache();
  std::lock_guard lock(c.mutex);
  for (auto it = c.entries.begin(); it != c.entries.end(); ++it) {
    if (!(*it)->released && (*it)->model_space() == ms) {
      auto hit = *it;  // most recently used goes to the back
      c.entries.erase(it);
      c.entries.push_back(hit);
      return hit;
    }
  }
  auto w = std::make_shared<PyWorkspace>(ms);
  c.entries.push_back(w);
  if (c.entries.size() > kCacheSize) c.entries.erase(c.entries.begin());
  return w;
}

void PurgeWorkspaceCache() {
  auto& c = GetCache();
  std::vector<std::shared_ptr<PyWorkspace>> dead;
  {
    std::unique_lock lock(c.mutex, std::try_to_lock);
    if (!lock.owns_lock()) return;  // a lookup is running; purge next time
    auto it = c.entries.begin();
    while (it != c.entries.end()) {
      // Held by the cache alone, and the model space only by the workspace
      // (PyWorkspace and commutator::Workspace each keep one reference).
      if (it->use_count() == 1 && (*it)->model_space().use_count() <= 2) {
        dead.push_back(std::move(*it));
        it = c.entries.erase(it);
      } else {
        ++it;
      }
    }
  }
}

void ClearWorkspaceCache() {
  auto& c = GetCache();
  std::vector<std::shared_ptr<PyWorkspace>> old;
  {
    std::lock_guard lock(c.mutex);
    old.swap(c.entries);
  }
}

namespace {

using commutator::Term;
using commutator::TermMask;

/// Binds the commutator Term enum and its name functions.
void BindTerm(nb::module_& m) {
  nb::enum_<Term>(m, "Term",
                  "The scalar commutator terms, named as in imsrg++ "
                  "(`commABCss`: A-body part of X and B-body part of Y to the "
                  "C-body part of [X, Y]); IMSRG(2), then the IMSRG(3) terms "
                  "(N7 first; the N8/N9 terms are not implemented).")
      .value("Comm110ss", Term::kComm110ss)
      .value("Comm220ss", Term::kComm220ss)
      .value("Comm111ss", Term::kComm111ss)
      .value("Comm121ss", Term::kComm121ss)
      .value("Comm221ss", Term::kComm221ss)
      .value("Comm122ss", Term::kComm122ss)
      .value("Comm222PpHhss", Term::kComm222PpHhss)
      .value("Comm222Phss", Term::kComm222Phss)
      .value("Comm133ss", Term::kComm133ss)
      .value("Comm330ss", Term::kComm330ss)
      .value("Comm223ss", Term::kComm223ss)
      .value("Comm232ss", Term::kComm232ss)
      .value("Comm331ss", Term::kComm331ss)
      .value("Comm231ss", Term::kComm231ss)
      .value("Comm132ss", Term::kComm132ss)
      .value("Comm233PpHhss", Term::kComm233PpHhss)
      .value("Comm233Phss", Term::kComm233Phss)
      .value("Comm332PpphHhhpss", Term::kComm332PpphHhhpss)
      .value("Comm332Pphhss", Term::kComm332Pphhss)
      .value("Comm333PppHhhss", Term::kComm333PppHhhss)
      .value("Comm333PphHhpss", Term::kComm333PphHhpss);
  m.def(
      "term_name", [](Term t) { return std::string(commutator::TermName(t)); },
      nb::arg("term"), "The imsrg++ name of a term (\"comm110ss\", ...).");
  m.def(
      "term_from_name",
      [](const std::string& name) { return commutator::TermFromName(name); },
      nb::arg("name"), "Inverse of term_name; None for an unknown name.");
}

/// Binds TermMask.
void BindTermMask(nb::module_& m) {
  nb::class_<TermMask>(m, "TermMask", "A set of commutator Terms (value type).")
      .def(nb::init<>(), "The empty set.")
      .def_static("none", &TermMask::None, "The empty set.")
      .def_static("imsrg2", &TermMask::Imsrg2,
                  "All scalar IMSRG(2) terms (the default).")
      .def_static("imsrg3_n7", &TermMask::Imsrg3N7,
                  "The IMSRG(2) terms and the seven N7 three-body terms.")
      .def_static("only", &TermMask::Only, nb::arg("term"),
                  "The set holding only `term`.")
      .def_static(
          "of",
          [](const std::vector<Term>& ts) {
            TermMask mask;
            for (const Term t : ts) mask.set(t);
            return mask;
          },
          nb::arg("terms"), "The set of the given terms.")
      .def("test", &TermMask::test, nb::arg("term"),
           "True if `term` is in the set.")
      .def(
          "set", [](TermMask& mk, Term t, bool on) { mk.set(t, on); },
          nb::arg("term"), nb::arg("on") = true, "Adds or removes `term`.")
      .def(
          "reset", [](TermMask& mk, Term t) { mk.reset(t); }, nb::arg("term"),
          "Removes `term`.")
      .def("empty", &TermMask::empty, "True for the empty set.")
      .def_prop_ro("bits", &TermMask::bits)
      .def(
          "terms",
          [](const TermMask& mk) {
            std::vector<Term> out;
            for (int i = 0; i < commutator::kNumTerms; ++i)
              if (mk.test(static_cast<Term>(i)))
                out.push_back(static_cast<Term>(i));
            return out;
          },
          "The terms of the set in imsrg++ call order.")
      .def("__or__", [](TermMask a, TermMask b) { return a | b; })
      .def("__and__", [](TermMask a, TermMask b) { return a & b; })
      .def("__eq__", [](TermMask a, TermMask b) { return a == b; })
      .def("__hash__", [](TermMask a) { return a.bits(); })
      .def("__repr__", [](const TermMask& mk) {
        std::string s = "TermMask([";
        bool first = true;
        for (int i = 0; i < commutator::kNumTerms; ++i) {
          if (!mk.test(static_cast<Term>(i))) continue;
          s += (first ? "" : ", ");
          s += std::string(commutator::TermName(static_cast<Term>(i)));
          first = false;
        }
        return s + "])";
      });
}

/// Binds CommutatorOptions and FactorizedOptions.
void BindCommutatorOptions(nb::module_& m) {
  nb::class_<commutator::CommutatorOptions>(
      m, "CommutatorOptions", "Options of the commutator (term selection).")
      .def(
          "__init__",
          [](commutator::CommutatorOptions* self, TermMask terms) {
            new (self) commutator::CommutatorOptions{terms};
          },
          nb::kw_only(), nb::arg("terms") = TermMask::Imsrg2())
      .def_rw("terms", &commutator::CommutatorOptions::terms,
              "The TermMask of the terms to evaluate (default: all IMSRG(2) "
              "terms).");

  nb::class_<commutator::FactorizedOptions>(
      m, "FactorizedOptions",
      "Topologies of the factorized double commutator (IMSRG(3f2), D32).")
      .def(
          "__init__",
          [](commutator::FactorizedOptions* self, bool one_body,
             bool two_body) {
            new (self) commutator::FactorizedOptions{one_body, two_body};
          },
          nb::kw_only(), nb::arg("one_body_intermediates") = true,
          nb::arg("two_body_intermediates") = false)
      .def_rw("one_body_intermediates",
              &commutator::FactorizedOptions::one_body_intermediates)
      .def_rw("two_body_intermediates",
              &commutator::FactorizedOptions::two_body_intermediates,
              "Not implemented: True raises NotImplementedError.");
}

/// Binds Workspace and its cache.
void BindWorkspace(nb::module_& m) {
  nb::class_<PyWorkspace>(
      m, "CommutatorWorkspace",
      "Precomputed tables and scratch buffers of the commutator kernels for "
      "one model space. Functions build and cache one by themselves; make "
      "one explicitly to control its lifetime. Not thread safe (calls on one "
      "workspace serialize).")
      .def(
          "__init__",
          [](PyWorkspace* self, const PyModelSpace& ms) {
            EnsureRuntime();
            new (self) PyWorkspace(ms.model_space());
          },
          nb::arg("model_space"))
      .def_prop_ro(
          "model_space",
          [](PyWorkspace& w) {
            w.ws();  // raises StaleHostError if released
            return PyModelSpace{w.model_space()};
          },
          "The model space the workspace was built for.");

  m.def("clear_workspace_cache", &ClearWorkspaceCache,
        "Drops the workspaces cached by commutator() and the BCH functions.");
}

/// Binds commutator and factorized_double_commutator.
void BindCommutatorFunctions(nb::module_& m) {
  m.def(
      "commutator",
      [](const PyOperator& x, const PyOperator& y,
         const commutator::CommutatorOptions& options,
         std::shared_ptr<PyWorkspace> workspace) {
        const op::Operator& xi = Input(x);
        const op::Operator& yi = Input(y);
        RequireSameModelSpace(xi, yi);
        if (!workspace) workspace = WorkspaceFor(xi.model_space_ptr());
        GilFree r;
        std::lock_guard lock(workspace->mutex());
        if (!workspace->ws().Compatible(xi))
          throw ModelSpaceError(
              "the workspace was built for another model space");
        return Result(commutator::Commutator(xi, yi, options, workspace->ws()));
      },
      nb::arg("x"), nb::arg("y"), nb::kw_only(),
      nb::arg("options") = commutator::CommutatorOptions{},
      nb::arg("workspace").none() = nb::none(),
      "Z = [X, Y] restricted to the terms of `options.terms` (IMSRG(2): Z "
      "has no three-body part). X and Y must be scalar, number conserving "
      "and of particle rank <= 2; the hermiticity of Z follows from those "
      "of X and Y. NonHermitian operands raise NotImplementedError in this "
      "version of the core. Complexity: O(sum over channels of N^3).");

  m.def(
      "factorized_double_commutator",
      [](const PyOperator& eta, const PyOperator& gamma,
         const commutator::FactorizedOptions& options,
         std::shared_ptr<PyWorkspace> workspace) {
        const op::Operator& e = Input(eta);
        const op::Operator& g = Input(gamma);
        RequireSameModelSpace(e, g);
        if (!workspace) workspace = WorkspaceFor(e.model_space_ptr());
        GilFree r;
        std::lock_guard lock(workspace->mutex());
        if (!workspace->ws().Compatible(e))
          throw ModelSpaceError(
              "the workspace was built for another model space");
        op::Operator z;
        commutator::FactorizedDoubleCommutator(e, g, z, options,
                                               workspace->ws());
        return Result(std::move(z));
      },
      nb::arg("eta"), nb::arg("gamma"), nb::kw_only(),
      nb::arg("options") = commutator::FactorizedOptions{},
      nb::arg("workspace").none() = nb::none(),
      "The factorized double commutator F(eta, gamma) of IMSRG(3f2): the "
      "part of [eta, [eta, gamma]_3b] that imsrg++ evaluates with one-body "
      "intermediates. eta must be anti-Hermitian.");
}

}  // namespace

void BindCommutator(nb::module_& m) {
  BindTerm(m);
  BindTermMask(m);
  BindCommutatorOptions(m);
  BindWorkspace(m);
  BindCommutatorFunctions(m);
}

}  // namespace imsrg::py
