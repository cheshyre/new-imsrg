// SPDX-License-Identifier: GPL-3.0-or-later
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

/// The scalar IMSRG(2) commutator terms, named as in imsrg++ (`Commutator.cc`):
/// `commABCss` takes the A-body part of X and the B-body part of Y (plus the X
/// <-> Y exchange) to the C-body part of Z = [X, Y]. The enumerator order is
/// the call order of imsrg++ `CommutatorScalarScalar`.
enum class Term : std::uint8_t {
  kComm110ss = 0,  ///< [X1, Y1] -> Z0.
  kComm220ss,      ///< [X2, Y2] -> Z0.
  kComm111ss,      ///< [X1, Y1] -> Z1.
  kComm121ss,      ///< [X1, Y2] + [X2, Y1] -> Z1.
  kComm221ss,      ///< [X2, Y2] -> Z1.
  kComm122ss,      ///< [X1, Y2] + [X2, Y1] -> Z2.
  kComm222PpHhss,  ///< [X2, Y2] -> Z2, particle-particle and hole-hole
                   ///< ladders.
  kComm222Phss,    ///< [X2, Y2] -> Z2, particle-hole (Pandya-transformed) part.
};

/// Number of `Term` values.
inline constexpr int kNumTerms = 8;

/// The imsrg++ name of a term ("comm110ss", ...).
std::string_view TermName(Term t) noexcept;

/// Inverse of `TermName`; nullopt for an unknown name.
std::optional<Term> TermFromName(std::string_view name) noexcept;

/// A set of `Term`s (bitset over the enumerators). Value type; the IMSRG(3)
/// terms of M10 will add enumerators (the 64-bit storage leaves room) and
/// presets `imsrg3_n7`, `imsrg3_full`.
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
    return TermMask((std::uint64_t{1} << kNumTerms) - 1);
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
    return std::uint64_t{1} << static_cast<std::uint8_t>(t);
  }
  std::uint64_t bits_ = 0;
};
static_assert(std::is_trivially_copyable_v<TermMask>,
              "TermMask is passed by value into kernels and options");

/// Options of `Commutator()`. Passed explicitly to every call (no global state,
/// design/00 D10).
struct CommutatorOptions {
  /// The terms to evaluate; default: all IMSRG(2) terms.
  TermMask terms = TermMask::Imsrg2();
};

}  // namespace imsrg::commutator

#endif  // IMSRG_COMMUTATOR_OPTIONS_H_
