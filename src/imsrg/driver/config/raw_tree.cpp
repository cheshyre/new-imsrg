#include "imsrg/driver/config/raw_tree.hpp"

namespace imsrg::driver {

std::string_view RawValue::type_name() const noexcept {
  switch (data.index()) {
    case 0:
      return "boolean";
    case 1:
      return "integer";
    case 2:
      return "float";
    case 3:
      return "string";
    case 4:
      return "array";
    case 5:
      return "table";
    default:
      return "unknown";
  }
}

void RawTree::set(std::string path, RawValue value) {
  entries_.insert_or_assign(std::move(path), std::move(value));
}

const RawValue* RawTree::find(std::string_view path) const {
  const auto it = entries_.find(path);
  return it == entries_.end() ? nullptr : &it->second;
}

bool RawTree::contains(std::string_view path) const {
  return entries_.find(path) != entries_.end();
}

bool RawTree::erase(std::string_view path) {
  const auto it = entries_.find(path);
  if (it == entries_.end()) return false;
  entries_.erase(it);
  return true;
}

bool RawTree::has_table(std::string_view table) const {
  std::string prefix(table);
  prefix += '.';
  const auto it = entries_.lower_bound(prefix);
  return it != entries_.end() && it->first.starts_with(prefix);
}

void RawTree::merge(const RawTree& overlay) {
  for (const auto& [path, value] : overlay.entries_) entries_.insert_or_assign(path, value);
}

}  // namespace imsrg::driver
