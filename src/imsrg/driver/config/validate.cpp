// STUB (M8a task T2): replaced by the validation rules.
#include "imsrg/base/error.hpp"
#include "imsrg/driver/config/resolve.hpp"

namespace imsrg::driver {

ConfigIssues validate(const RunConfig&, const RawTree&, const ResolveOptions&) {
  throw NotImplementedError("validate");
}

}  // namespace imsrg::driver
