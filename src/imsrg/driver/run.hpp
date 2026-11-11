#pragma once
/// @file
/// The pipeline entry point of the `imsrg` executable and the Python bindings.

#include "imsrg/driver/config/schema.hpp"

namespace imsrg::driver {

/// Runs the calculation described by the resolved `config` (design/16 section 6.1). First writes
/// `<output.directory>/resolved_config.toml` (directory created if missing), so that a run that
/// fails later still leaves it (design/16 section 1.7); then runs the stages.
///
/// Until milestone M8 wires the pipeline (HF, flow, operators, output), it stops after writing the
/// resolved config with a `ConfigError` whose message starts with "E-PLANNED".
/// @throws InputError if the resolved config cannot be written; ConfigError (see above).
void run(const RunConfig& config);

}  // namespace imsrg::driver
