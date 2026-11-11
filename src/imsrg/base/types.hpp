#pragma once
/// @file
/// Strong types for indices and quantum numbers (design/00 D9, design/10 section 3.1).
///
/// Mixing up J and 2J, or local and global ket indices, is the most common bug class in this kind
/// of code, so public APIs never take raw `int`s for these. Every type here is trivially copyable,
/// standard layout, `constexpr`, and usable in device code (`IMSRG_INLINE_FUNCTION`); a
/// `static_assert` next to each definition guards this (design/12 section 5 rule 3).
///
/// Conventions: angular momenta and their projections are stored doubled (`TwoJ{3}` is j = 3/2);
/// isospin projections follow imsrg++ (proton tz2 = -1, neutron tz2 = +1).

#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>

#include "imsrg/base/device.hpp"

namespace imsrg {

/// An integer index with a tag type, so that indices of different kinds cannot be mixed.
///
/// Construction from the raw representation is explicit; the only arithmetic is increment and
/// decrement (use `value()` for anything else, which makes index arithmetic visible in the code).
/// A default-constructed index has value 0. `Rep` must be an integral type.
template <class Tag, class Rep = std::int32_t>
class StrongIndex {
  static_assert(std::is_integral_v<Rep>, "StrongIndex needs an integral representation");

 public:
  using rep_type = Rep;
  using tag_type = Tag;

  /// Index 0.
  StrongIndex() = default;
  /// Wraps the raw value `v`.
  IMSRG_INLINE_FUNCTION constexpr explicit StrongIndex(Rep v) noexcept : value_(v) {}

  /// The raw value.
  IMSRG_INLINE_FUNCTION constexpr Rep value() const noexcept { return value_; }
  /// The raw value as `std::size_t`, for indexing arrays. Precondition: `value() >= 0`.
  IMSRG_INLINE_FUNCTION constexpr std::size_t to_size() const noexcept {
    return static_cast<std::size_t>(value_);
  }

  /// Pre-increment.
  IMSRG_INLINE_FUNCTION constexpr StrongIndex& operator++() noexcept {
    ++value_;
    return *this;
  }
  /// Post-increment.
  IMSRG_INLINE_FUNCTION constexpr StrongIndex operator++(int) noexcept {
    StrongIndex old = *this;
    ++value_;
    return old;
  }
  /// Pre-decrement.
  IMSRG_INLINE_FUNCTION constexpr StrongIndex& operator--() noexcept {
    --value_;
    return *this;
  }
  /// Post-decrement.
  IMSRG_INLINE_FUNCTION constexpr StrongIndex operator--(int) noexcept {
    StrongIndex old = *this;
    --value_;
    return old;
  }

  // Comparisons are spelled out (instead of a defaulted <=>) so that they are plain
  // __host__ __device__ functions on CUDA/HIP.
  friend IMSRG_INLINE_FUNCTION constexpr bool operator==(StrongIndex a, StrongIndex b) noexcept {
    return a.value_ == b.value_;
  }
  friend IMSRG_INLINE_FUNCTION constexpr bool operator!=(StrongIndex a, StrongIndex b) noexcept {
    return a.value_ != b.value_;
  }
  friend IMSRG_INLINE_FUNCTION constexpr bool operator<(StrongIndex a, StrongIndex b) noexcept {
    return a.value_ < b.value_;
  }
  friend IMSRG_INLINE_FUNCTION constexpr bool operator<=(StrongIndex a, StrongIndex b) noexcept {
    return a.value_ <= b.value_;
  }
  friend IMSRG_INLINE_FUNCTION constexpr bool operator>(StrongIndex a, StrongIndex b) noexcept {
    return a.value_ > b.value_;
  }
  friend IMSRG_INLINE_FUNCTION constexpr bool operator>=(StrongIndex a, StrongIndex b) noexcept {
    return a.value_ >= b.value_;
  }

 private:
  Rep value_ = 0;
};

/// Single-particle orbit index in imsrg++ order (design/00 D11).
using OrbitIndex = StrongIndex<struct OrbitTag>;
/// Two-body channel (J, parity, Tz) index.
using ChannelIndex = StrongIndex<struct ChannelTag>;
/// Global two-body ket index |pq> (p <= q).
using KetIndex = StrongIndex<struct KetTag>;
/// Row/column index inside one channel block.
using LocalKetIndex = StrongIndex<struct LocalKetTag>;
/// Three-body channel index.
using Channel3Index = StrongIndex<struct Channel3Tag>;
/// Three-body ket index.
using Ket3Index = StrongIndex<struct Ket3Tag>;

static_assert(std::is_trivially_copyable_v<OrbitIndex> && std::is_standard_layout_v<OrbitIndex>);
static_assert(sizeof(OrbitIndex) == sizeof(std::int32_t));
static_assert(!std::is_convertible_v<int, OrbitIndex>, "construction must be explicit");
static_assert(!std::is_convertible_v<OrbitIndex, KetIndex>, "index kinds must not mix");

// Comparison operators for the doubled quantum numbers below (all compare `.twice`). A macro
// instead of a CRTP base keeps the types aggregates, so `TwoJ{3}` works.
#define IMSRG_DETAIL_TWICE_COMPARISONS(T)                                     \
  friend IMSRG_INLINE_FUNCTION constexpr bool operator==(T a, T b) noexcept { \
    return a.twice == b.twice;                                                \
  }                                                                           \
  friend IMSRG_INLINE_FUNCTION constexpr bool operator!=(T a, T b) noexcept { \
    return a.twice != b.twice;                                                \
  }                                                                           \
  friend IMSRG_INLINE_FUNCTION constexpr bool operator<(T a, T b) noexcept {  \
    return a.twice < b.twice;                                                 \
  }                                                                           \
  friend IMSRG_INLINE_FUNCTION constexpr bool operator<=(T a, T b) noexcept { \
    return a.twice <= b.twice;                                                \
  }                                                                           \
  friend IMSRG_INLINE_FUNCTION constexpr bool operator>(T a, T b) noexcept {  \
    return a.twice > b.twice;                                                 \
  }                                                                           \
  friend IMSRG_INLINE_FUNCTION constexpr bool operator>=(T a, T b) noexcept { \
    return a.twice >= b.twice;                                                \
  }

/// Twice an angular momentum: 2j of an orbit, or 2J of a coupled state. `TwoJ{3}` is j = 3/2.
/// Construct explicitly (`TwoJ{2 * J}` or `TwoJ::from_integer(J)`); there is no conversion from
/// int.
struct TwoJ {
  std::int32_t twice = 0;

  IMSRG_DETAIL_TWICE_COMPARISONS(TwoJ)

  /// 2J for an integer J.
  IMSRG_INLINE_FUNCTION static constexpr TwoJ from_integer(std::int32_t j) noexcept {
    return TwoJ{2 * j};
  }
};

/// Twice an angular-momentum projection: 2m or 2M. `TwoM{-1}` is m = -1/2.
struct TwoM {
  std::int32_t twice = 0;

  IMSRG_DETAIL_TWICE_COMPARISONS(TwoM)
};

/// Twice an isospin projection (imsrg++ convention): proton tz2 = -1, neutron tz2 = +1.
struct TwoTz {
  std::int32_t twice = 0;

  IMSRG_DETAIL_TWICE_COMPARISONS(TwoTz)

  /// tz2 of a proton (-1).
  IMSRG_INLINE_FUNCTION static constexpr TwoTz proton() noexcept { return TwoTz{-1}; }
  /// tz2 of a neutron (+1).
  IMSRG_INLINE_FUNCTION static constexpr TwoTz neutron() noexcept { return TwoTz{+1}; }
};

static_assert(std::is_trivially_copyable_v<TwoJ> && std::is_standard_layout_v<TwoJ>);
static_assert(std::is_trivially_copyable_v<TwoM> && std::is_standard_layout_v<TwoM>);
static_assert(std::is_trivially_copyable_v<TwoTz> && std::is_standard_layout_v<TwoTz>);
static_assert(sizeof(TwoJ) == 4 && sizeof(TwoM) == 4 && sizeof(TwoTz) == 4);
static_assert(!std::is_convertible_v<int, TwoJ> && !std::is_convertible_v<TwoJ, TwoM>);
static_assert(std::is_aggregate_v<TwoJ> && TwoJ{3}.twice == 3);

#undef IMSRG_DETAIL_TWICE_COMPARISONS

/// Degeneracy 2j + 1 of an angular momentum.
IMSRG_INLINE_FUNCTION constexpr std::int32_t degeneracy(TwoJ j) noexcept { return j.twice + 1; }
/// The angular momentum j itself (e.g. 1.5 for `TwoJ{3}`).
IMSRG_INLINE_FUNCTION constexpr double as_double(TwoJ j) noexcept { return 0.5 * j.twice; }
/// The projection m itself (e.g. -0.5 for `TwoM{-1}`).
IMSRG_INLINE_FUNCTION constexpr double as_double(TwoM m) noexcept { return 0.5 * m.twice; }
/// True if j is half-integer (2j odd).
IMSRG_INLINE_FUNCTION constexpr bool is_half_integer(TwoJ j) noexcept { return (j.twice % 2) != 0; }

/// Parity of a state, (-1)^l for an orbit. Combining parities multiplies them (`operator*`).
enum class Parity : std::uint8_t { Even = 0, Odd = 1 };

/// Parity (-1)^l of orbital angular momentum l (l >= 0).
IMSRG_INLINE_FUNCTION constexpr Parity parity_of_l(std::int32_t l) noexcept {
  return (l % 2 == 0) ? Parity::Even : Parity::Odd;
}
/// Product of two parities.
IMSRG_INLINE_FUNCTION constexpr Parity operator*(Parity a, Parity b) noexcept {
  return static_cast<Parity>(static_cast<std::uint8_t>(a) ^ static_cast<std::uint8_t>(b));
}
/// +1 for even, -1 for odd parity.
IMSRG_INLINE_FUNCTION constexpr std::int32_t sign(Parity p) noexcept {
  return p == Parity::Even ? 1 : -1;
}

/// Hermiticity of an operator: O^dagger = O, O^dagger = -O, or neither.
enum class Hermiticity : std::uint8_t { Hermitian, AntiHermitian, NonHermitian };

static_assert(std::is_trivially_copyable_v<Parity> && sizeof(Parity) == 1);
static_assert(std::is_trivially_copyable_v<Hermiticity> && sizeof(Hermiticity) == 1);

}  // namespace imsrg

/// Hash support, so strong indices can be keys of unordered containers (host code).
template <class Tag, class Rep>
struct std::hash<imsrg::StrongIndex<Tag, Rep>> {
  std::size_t operator()(imsrg::StrongIndex<Tag, Rep> i) const noexcept {
    return std::hash<Rep>{}(i.value());
  }
};
