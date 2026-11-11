#pragma once

/// @file
/// Explicit options of the commutator module (design/00 D10, design/13 section 6): the term
/// selection `TermMask` (an enum bitset, no string keys, no globals) and `CommutatorOptions`.

#include <cstdint>
#include <optional>
#include <string_view>

namespace imsrg::commutator {

/// The scalar IMSRG(2) commutator terms, named as in imsrg++ (`Commutator.cc`): `commABCss`
/// takes the A-body part of X and the B-body part of Y (plus the X <-> Y exchange) to the
/// C-body part of Z = [X, Y]. The enumerator order is the call order of imsrg++
/// `CommutatorScalarScalar`.
enum class Term : std::uint8_t {
  comm110ss = 0,    ///< [X1, Y1] -> Z0.
  comm220ss,        ///< [X2, Y2] -> Z0.
  comm111ss,        ///< [X1, Y1] -> Z1.
  comm121ss,        ///< [X1, Y2] + [X2, Y1] -> Z1.
  comm221ss,        ///< [X2, Y2] -> Z1.
  comm122ss,        ///< [X1, Y2] + [X2, Y1] -> Z2.
  comm222_pp_hhss,  ///< [X2, Y2] -> Z2, particle-particle and hole-hole ladders.
  comm222_phss,     ///< [X2, Y2] -> Z2, particle-hole (Pandya-transformed) part.
};

/// Number of `Term` values.
inline constexpr int kNumTerms = 8;

/// The imsrg++ name of a term ("comm110ss", ...).
std::string_view term_name(Term t) noexcept;

/// Inverse of `term_name`; nullopt for an unknown name.
std::optional<Term> term_from_name(std::string_view name) noexcept;

/// A set of `Term`s (bitset over the enumerators). Value type; the IMSRG(3) terms of M10 will
/// add enumerators (the 64-bit storage leaves room) and presets `imsrg3_n7`, `imsrg3_full`.
class TermMask {
 public:
  /// The empty set.
  constexpr TermMask() noexcept = default;

  /// The empty set.
  static constexpr TermMask none() noexcept { return TermMask(); }

  /// All scalar IMSRG(2) terms (the imsrg++ default with IMSRG(3) off).
  static constexpr TermMask imsrg2() noexcept {
    return TermMask((std::uint64_t{1} << kNumTerms) - 1);
  }

  /// The set holding only `t`.
  static constexpr TermMask only(Term t) noexcept { return TermMask(bit(t)); }

  /// True if `t` is in the set.
  constexpr bool test(Term t) const noexcept { return (bits_ & bit(t)) != 0; }

  /// Adds (`on`) or removes `t`; returns *this.
  constexpr TermMask& set(Term t, bool on = true) noexcept {
    bits_ = on ? (bits_ | bit(t)) : (bits_ & ~bit(t));
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
  static constexpr std::uint64_t bit(Term t) noexcept {
    return std::uint64_t{1} << static_cast<std::uint8_t>(t);
  }
  std::uint64_t bits_ = 0;
};

/// Options of `commutator()`. Passed explicitly to every call (no global state, design/00 D10).
struct CommutatorOptions {
  /// The terms to evaluate; default: all IMSRG(2) terms.
  TermMask terms = TermMask::imsrg2();
};

}  // namespace imsrg::commutator
