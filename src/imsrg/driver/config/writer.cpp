// STUB (M8a task T1): replaced by the resolved-config writer.
#include "imsrg/driver/config/writer.hpp"

#include "imsrg/base/error.hpp"

namespace imsrg::driver {

std::string to_toml(const RunConfig&, bool) { throw NotImplementedError("to_toml"); }
void write_resolved_config(const RunConfig&, const std::filesystem::path&) {
  throw NotImplementedError("write_resolved_config");
}

}  // namespace imsrg::driver
