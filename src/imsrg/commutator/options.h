// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_COMMUTATOR_OPTIONS_H_
#define IMSRG_COMMUTATOR_OPTIONS_H_

/// @file
/// Explicit options of the commutator module (design/00 D10, design/13 section
/// 6): the term selection `TermMask` (an enum bitset, no string keys, no
/// globals) and `CommutatorOptions`.

#include <cstdint>
#include <optional>
#include <string_view>
#include <type_traits>

namespace imsrg::commutator {

/// The scalar commutator terms, named as in imsrg++ (`Commutator.cc`,
/// `IMSRG3Commutators.cc`): `commABCss` takes the A-body part of X and the
/// B-body part of Y (plus the X <-> Y exchange) to the C-body part of Z = [X,
/// Y]. The enumerator order is the call order of imsrg++
/// `CommutatorScalarScalar` (Commutator.cc:359-454): the IMSRG(2) terms, then
/// the IMSRG(3) block, the seven N7 terms first and then the six N8/N9 terms of
/// the full IMSRG(3), which are not implemented (requesting them throws
/// NotImplementedError).
enum class Term : std::int8_t {
  kComm110ss = 0,  ///< [X1, Y1] -> Z0.
  kComm220ss,      ///< [X2, Y2] -> Z0.
  kComm111ss,      ///< [X1, Y1] -> Z1.
  kComm121ss,      ///< [X1, Y2] + [X2, Y1] -> Z1.
  kComm221ss,      ///< [X2, Y2] -> Z1.
  kComm122ss,      ///< [X1, Y2] + [X2, Y1] -> Z2.
  kComm222PpHhss,  ///< [X2, Y2] -> Z2, particle-particle and hole-hole
                   ///< ladders.
  kComm222Phss,    ///< [X2, Y2] -> Z2, particle-hole (Pandya-transformed) part.
  kComm133ss,      ///< [X1, Y3] + [X3, Y1] -> Z3 (N7).
  kComm330ss,      ///< [X3, Y3] -> Z0 (N7).
  kComm223ss,      ///< [X2, Y2] -> Z3 (N7).
  kComm232ss,      ///< [X2, Y3] + [X3, Y2] -> Z2 (N7).
  kComm331ss,      ///< [X3, Y3] -> Z1 (N7).
  kComm231ss,      ///< [X2, Y3] + [X3, Y2] -> Z1 (N7).
  kComm132ss,      ///< [X1, Y3] + [X3, Y1] -> Z2 (N7).
  kComm233PpHhss,  ///< [X2, Y3] + [X3, Y2] -> Z3, pp/hh (N8, planned).
  kComm233Phss,    ///< [X2, Y3] + [X3, Y2] -> Z3, ph (N8, planned).
  kComm332PpphHhhpss,  ///< [X3, Y3] -> Z2, ppph/hhhp (N8, planned).
  kComm332Pphhss,      ///< [X3, Y3] -> Z2, pphh (N8, planned).
  kComm333PppHhhss,    ///< [X3, Y3] -> Z3, ppp/hhh (N9, planned).
  kComm333PphHhpss,    ///< [X3, Y3] -> Z3, pph/hhp (N9, planned).
};

/// Number of `Term` values.
inline constexpr int kNumTerms = 21;
/// Number of IMSRG(2) terms (the enumerators kComm110ss..kComm222Phss).
inline constexpr int kNumImsrg2Terms = 8;
/// Number of N7 terms (the enumerators kComm133ss..kComm132ss).
inline constexpr int kNumN7Terms = 7;

/// The imsrg++ name of a term ("comm110ss", ...).
std::string_view TermName(Term t) noexcept;

/// Inverse of `TermName`; nullopt for an unknown name.
std::optional<Term> TermFromName(std::string_view name) noexcept;

/// A set of `Term`s (bitset over the enumerators). Value type. Presets:
/// `Imsrg2()` (imsrg++ with IMSRG(3) off), `Imsrg3N7()` (imsrg++ `IMSRG3=true
/// imsrg3_n7=true`: the IMSRG(2) terms and the seven N7 terms,
/// `SetUseIMSRG3N7`, Commutator.cc:111-129) and `Imsrg3Full()` (every term,
/// `SetUseIMSRG3`; the N8/N9 terms are not implemented).
class TermMask {
 public:
  /// The empty set.
  constexpr TermMask() noexcept = default;

  TermMask(const TermMask&) = default;
  TermMask& operator=(const TermMask&) = default;
  TermMask(TermMask&&) = default;
  TermMask& operator=(TermMask&&) = default;

  /// The empty set.
  static constexpr TermMask None() noexcept { return TermMask(); }

  /// All scalar IMSRG(2) terms (the imsrg++ default with IMSRG(3) off).
  static constexpr TermMask Imsrg2() noexcept {
    return TermMask((std::uint64_t{1} << kNumImsrg2Terms) - 1);
  }

  /// The IMSRG(2) terms and the seven N7 three-body terms (IMSRG(3)-N7).
  static constexpr TermMask Imsrg3N7() noexcept {
    return TermMask((std::uint64_t{1} << (kNumImsrg2Terms + kNumN7Terms)) - 1);
  }

  /// Every term (full IMSRG(3); the N8/N9 terms are not implemented).
  static constexpr TermMask Imsrg3Full() noexcept {
    return TermMask((std::uint64_t{1} << kNumTerms) - 1);
  }

  /// The IMSRG(3) terms of the set (the enumerators after the IMSRG(2) ones).
  constexpr TermMask ThreeBodyTerms() const noexcept {
    return TermMask(bits_ & ~((std::uint64_t{1} << kNumImsrg2Terms) - 1));
  }

  /// The set holding only `t`.
  static constexpr TermMask Only(Term t) noexcept { return TermMask(Bit(t)); }

  /// True if `t` is in the set.
  constexpr bool test(Term t) const noexcept { return (bits_ & Bit(t)) != 0; }

  /// Adds (`on`) or removes `t`; returns *this.
  constexpr TermMask& set(Term t, bool on = true) noexcept {
    bits_ = on ? (bits_ | Bit(t)) : (bits_ & ~Bit(t));
    return *this;
  }

  /// Removes `t`; returns *this.
  constexpr TermMask& reset(Term t) noexcept { return set(t, false); }

  /// True for the empty set.
  constexpr bool empty() const noexcept { return bits_ == 0; }

  /// The raw bits (bit i = enumerator value i).
  constexpr std::uint64_t bits() const noexcept { return bits_; }

  /// Set equality.
  friend constexpr bool operator==(TermMask a, TermMask b) noexcept = default;

  /// Union.
  friend constexpr TermMask operator|(TermMask a, TermMask b) noexcept {
    return TermMask(a.bits_ | b.bits_);
  }

  /// Intersection.
  friend constexpr TermMask operator&(TermMask a, TermMask b) noexcept {
    return TermMask(a.bits_ & b.bits_);
  }

 private:
  constexpr explicit TermMask(std::uint64_t bits) noexcept : bits_(bits) {}
  static constexpr std::uint64_t Bit(Term t) noexcept {
    return std::uint64_t{1} << static_cast<int>(t);
  }
  std::uint64_t bits_ = 0;
};
static_assert(std::is_trivially_copyable_v<TermMask>,
              "TermMask is passed by value into kernels and options");

/// Options of `Commutator()`. Passed explicitly to every call (no global state,
/// design/00 D10).
struct CommutatorOptions {
  /// The terms to evaluate; default: all IMSRG(2) terms. A set with an
  /// IMSRG(3) term (`ThreeBodyTerms()` not empty) selects the IMSRG(3)
  /// truncation: Z gets a three-body part (imsrg++ `use_imsrg3`).
  TermMask terms = TermMask::Imsrg2();
  /// The IMSRG(3) terms are evaluated only if ||X|| > threebody_threshold and
  /// ||Y|| > threebody_threshold (full operator norms, `op::Norm`; imsrg++
  /// `threebody_threshold`, Commutator.cc:396). Default 0: always, unless an
  /// operand is exactly zero.
  double threebody_threshold = 0.0;
  /// Scalar-tensor commutators (M15, `TensorCommutator`): 9j symbols with
  /// |9j| below this are skipped in the particle-hole term (imsrg-devel
  /// 1e-10; the 79e9330 fork used 1e-8, design/50 section 5).
  double tensor_ninej_threshold = 1.0e-10;
};

}  // namespace imsrg::commutator

#endif  // IMSRG_COMMUTATOR_OPTIONS_H_
