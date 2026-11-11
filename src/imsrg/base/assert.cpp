#include "imsrg/base/assert.hpp"

#include <string>

#include "imsrg/base/error.hpp"

namespace imsrg::detail {

void check_failed(const char* condition, const char* file, int line, std::string_view message) {
  std::string what = "check failed: ";
  what += condition;
  what += " at ";
  what += file;
  what += ':';
  what += std::to_string(line);
  if (!message.empty()) {
    what += ": ";
    what += message;
  }
  throw InvariantError(what);
}

}  // namespace imsrg::detail
