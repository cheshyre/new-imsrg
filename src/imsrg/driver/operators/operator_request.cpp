// STUB (M8a task T4): replaced by the operator catalog and parser.
#include "imsrg/driver/operators/operator_request.hpp"

#include "imsrg/base/error.hpp"

namespace imsrg::driver {

std::span<const OperatorCatalogEntry> operator_catalog() noexcept { return {}; }
Expected<std::vector<OperatorRequest>, ConfigIssues> parse_operator_list(std::span<const RawValue>,
                                                                         const ModelSpaceInfo&,
                                                                         std::string_view) {
  throw NotImplementedError("parse_operator_list");
}
Expected<std::vector<OperatorRequest>, ConfigIssues> parse_operator_spec(std::string_view,
                                                                         const ModelSpaceInfo&) {
  throw NotImplementedError("parse_operator_spec");
}
std::string operator_catalog_text() { throw NotImplementedError("operator_catalog_text"); }

}  // namespace imsrg::driver
