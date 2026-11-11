#pragma once
/// @file
/// IEEE-754 binary16 ("half") <-> binary32 conversion on raw bit patterns, for compact storage of
/// matrix elements. Usable in host and device code, constexpr, no exceptions.

#include <bit>
#include <cstdint>

#include "imsrg/base/device.hpp"

namespace imsrg {

/// Converts binary16 bits to float. Exact for every input: +-0, subnormals, normals, +-inf; NaNs
/// stay NaN (payload shifted, sign kept). O(1).
IMSRG_INLINE_FUNCTION constexpr float half_to_float(std::uint16_t bits) noexcept {
  const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
  const std::uint32_t exp = (bits >> 10) & 0x1Fu;
  std::uint32_t man = bits & 0x3FFu;
  std::uint32_t out = 0;
  if (exp == 0x1Fu) {  // inf / NaN
    out = sign | 0x7F800000u | (man << 13);
  } else if (exp != 0) {  // normal
    out = sign | ((exp + 112u) << 23) | (man << 13);
  } else if (man == 0) {  // zero
    out = sign;
  } else {                   // subnormal half: normalize
    std::uint32_t e = 113u;  // float biased exponent of 2^-14
    while ((man & 0x400u) == 0) {
      man <<= 1;
      --e;
    }
    man &= 0x3FFu;
    out = sign | (e << 23) | (man << 13);
  }
  return std::bit_cast<float>(out);
}

/// Converts float to binary16 bits with round-to-nearest-even. Overflow (after rounding) gives
/// +-inf, NaN gives a quiet NaN with the input's sign, values below half the smallest subnormal
/// round to +-0. O(1).
IMSRG_INLINE_FUNCTION constexpr std::uint16_t float_to_half(float value) noexcept {
  const std::uint32_t x = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t sign = (x >> 16) & 0x8000u;
  const std::uint32_t absx = x & 0x7FFFFFFFu;
  if (absx > 0x7F800000u) return static_cast<std::uint16_t>(sign | 0x7E00u);   // NaN
  if (absx >= 0x477FF000u) return static_cast<std::uint16_t>(sign | 0x7C00u);  // overflow / inf
  if (absx < 0x38800000u) {  // below the smallest normal half (2^-14): subnormal or zero
    if (absx < 0x33000000u) return static_cast<std::uint16_t>(sign);  // < 2^-25
    const std::uint32_t exp = absx >> 23;                             // 102..112
    const std::uint32_t man = (absx & 0x7FFFFFu) | 0x800000u;
    const std::uint32_t shift = 126u - exp;  // 14..24
    std::uint32_t h = man >> shift;
    const std::uint32_t rem = man & ((1u << shift) - 1u);
    const std::uint32_t halfway = 1u << (shift - 1u);
    if (rem > halfway || (rem == halfway && (h & 1u) != 0)) ++h;
    return static_cast<std::uint16_t>(sign | h);
  }
  // Normal: rebias the exponent, round the 13 dropped bits to nearest even (carry propagates).
  std::uint32_t h = (absx - 0x38000000u) >> 13;
  const std::uint32_t rem = absx & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) != 0)) ++h;
  return static_cast<std::uint16_t>(sign | h);
}

}  // namespace imsrg
