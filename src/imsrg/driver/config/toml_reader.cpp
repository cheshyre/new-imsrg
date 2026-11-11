// STUB (M8a task T1): replaced by the toml++ based reader.
#include "imsrg/driver/config/toml_reader.hpp"

#include "imsrg/base/error.hpp"

namespace imsrg::driver {

ConfigIssues read_toml_file(const std::filesystem::path&, RawTree&) {
  throw NotImplementedError("read_toml_file");
}
ConfigIssues read_toml_string(std::string_view, std::string_view, RawTree&) {
  throw NotImplementedError("read_toml_string");
}
ConfigIssues apply_set_override(std::string_view, RawTree&) {
  throw NotImplementedError("apply_set_override");
}

}  // namespace imsrg::driver
