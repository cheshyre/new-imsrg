// Explicit instantiations of DeviceBuffer for the element types of the library.
#include "imsrg/exec/device_buffer.hpp"

#include "imsrg/exec/detail/device_buffer_impl.hpp"

namespace imsrg::exec {

template class DeviceBuffer<double>;
template class DeviceBuffer<float>;
template class DeviceBuffer<std::int32_t>;
template class DeviceBuffer<std::int64_t>;
template class DeviceBuffer<std::uint16_t>;
template class DeviceBuffer<std::uint8_t>;

}  // namespace imsrg::exec
