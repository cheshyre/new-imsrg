// STUB (M8a task T3): replaced by the legacy translator.
#include "imsrg/driver/legacy/legacy_parameters.hpp"

#include "imsrg/base/error.hpp"

namespace imsrg::driver {

std::span<const LegacyKeyInfo> legacy_keys() noexcept { return {}; }
const LegacyKeyInfo* find_legacy_key(std::string_view) noexcept { return nullptr; }
LegacyTranslation translate_legacy(std::span<const std::string>) {
  throw NotImplementedError("translate_legacy");
}
std::string legacy_keys_text() { throw NotImplementedError("legacy_keys_text"); }

}  // namespace imsrg::driver
