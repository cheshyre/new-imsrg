// Placeholder: replaced by pipeline stage 1 (M8 task P1).
#include "imsrg/driver/pipeline/hamiltonian.hpp"

#include "imsrg/base/error.hpp"

namespace imsrg::driver::pipeline {

PreparedHamiltonian prepare_hamiltonian(const RunConfig& /*config*/) {
  throw ConfigError("E-PLANNED: prepare_hamiltonian is not implemented yet");
}

op::Operator prepare_operator(const OperatorRequest& /*request*/,
                              const PreparedHamiltonian& /*ph*/, const RunConfig& /*config*/) {
  throw ConfigError("E-PLANNED: prepare_operator is not implemented yet");
}

}  // namespace imsrg::driver::pipeline
