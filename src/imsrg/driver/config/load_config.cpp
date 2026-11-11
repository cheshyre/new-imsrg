// STUB (M8a task T2): replaced by the loader.
#include "imsrg/driver/config/load_config.hpp"

#include "imsrg/base/error.hpp"

namespace imsrg::driver {

Expected<RunConfig, ConfigIssues> load_config(const ConfigSources&, const LoadOptions&) {
  throw NotImplementedError("load_config");
}

}  // namespace imsrg::driver
