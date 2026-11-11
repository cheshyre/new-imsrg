// The pipeline stub of the `imsrg` driver (design/16 section 6.1).
#include "imsrg/driver/run.hpp"

#include <filesystem>
#include <string>

#include "imsrg/base/error.hpp"
#include "imsrg/driver/config/writer.hpp"

namespace imsrg::driver {

void run(const RunConfig& config) {
  const std::filesystem::path file =
      std::filesystem::path(config.output.directory) / "resolved_config.toml";
  write_resolved_config(config, file);
  // M8 replaces this throw with the pipeline (HF, flow, operators, output) and creates the
  // Kokkos/MPI runtime in main.
  throw ConfigError(
      "E-PLANNED: the imsrg pipeline (HF, IMSRG flow, operators, output) is not implemented yet "
      "in this build (milestone M8); the resolved configuration was written to '" +
      file.string() + "'");
}

}  // namespace imsrg::driver
