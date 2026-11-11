#include "imsrg/exec/runtime.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "imsrg/base/error.hpp"

namespace imsrg::exec {

namespace {

void validate(const RuntimeOptions& o) {
  if (o.num_threads == 0 || o.num_threads < -1) {
    throw ConfigError("exec::Runtime: num_threads must be -1 (default) or >= 1, got " +
                      std::to_string(o.num_threads));
  }
  if (o.device_id < -1) {
    throw ConfigError("exec::Runtime: device_id must be -1 (default) or >= 0, got " +
                      std::to_string(o.device_id));
  }
}

void require_fresh_kokkos() {
  if (Kokkos::is_initialized()) {
    throw RuntimeError("exec::Runtime: Kokkos is already initialized");
  }
  if (Kokkos::is_finalized()) {
    throw RuntimeError("exec::Runtime: Kokkos was finalized and cannot be initialized again");
  }
}

Kokkos::InitializationSettings make_settings(const RuntimeOptions& o) {
  Kokkos::InitializationSettings s;
  if (o.num_threads >= 0) {
    s.set_num_threads(o.num_threads);
  }
  if (o.device_id >= 0) {
    s.set_device_id(o.device_id);
  }
  if (o.print_configuration) {
    s.set_print_configuration(true);
  }
  return s;
}

bool has_flag(const std::vector<char*>& args, const std::string& prefix) {
  for (const char* a : args) {
    if (a != nullptr && std::string(a).rfind(prefix, 0) == 0) {
      return true;
    }
  }
  return false;
}

}  // namespace

Runtime::Runtime(const RuntimeOptions& options) {
  validate(options);
  require_fresh_kokkos();
  Kokkos::initialize(make_settings(options));
}

Runtime::Runtime(int& argc, char** argv, const RuntimeOptions& options) {
  validate(options);
  require_fresh_kokkos();
  // Kokkos::initialize(argc, argv) cannot be combined with settings, so options are turned into
  // arguments (unless the command line already sets them) appended to a working copy.
  std::vector<char*> args(argv, argv + argc);
  std::vector<std::string> extra;
  if (options.num_threads >= 0 && !has_flag(args, "--kokkos-num-threads")) {
    extra.push_back("--kokkos-num-threads=" + std::to_string(options.num_threads));
  }
  if (options.device_id >= 0 && !has_flag(args, "--kokkos-device-id")) {
    extra.push_back("--kokkos-device-id=" + std::to_string(options.device_id));
  }
  if (options.print_configuration && !has_flag(args, "--kokkos-print-configuration")) {
    extra.push_back("--kokkos-print-configuration");
  }
  for (std::string& e : extra) {
    args.push_back(e.data());
  }
  int n = static_cast<int>(args.size());
  args.push_back(nullptr);
  Kokkos::initialize(n, args.data());
  // Kokkos removed the arguments it consumed, including every appended --kokkos-* flag (whose
  // strings die with `extra`), so at most the caller's arguments remain; hand them back.
  bool only_caller_args = n <= argc;
  for (int i = 0; i < n && only_caller_args; ++i) {
    only_caller_args =
        std::find(argv, argv + argc, args[static_cast<std::size_t>(i)]) != argv + argc;
  }
  if (!only_caller_args) {
    Kokkos::finalize();
    throw RuntimeError("exec::Runtime: Kokkos left appended --kokkos-* arguments unconsumed");
  }
  for (int i = 0; i < n; ++i) {
    argv[i] = args[static_cast<std::size_t>(i)];
  }
  if (n < argc) {
    argv[n] = nullptr;
  }
  argc = n;
}

Runtime::~Runtime() {
  if (Kokkos::is_initialized()) {
    Kokkos::fence("imsrg::exec::Runtime::~Runtime");
    Kokkos::finalize();
  }
}

bool Runtime::is_initialized() noexcept { return Kokkos::is_initialized(); }

std::string Runtime::execution_space_name() { return Kokkos::DefaultExecutionSpace::name(); }

int Runtime::concurrency() { return Kokkos::DefaultExecutionSpace().concurrency(); }

bool default_memory_is_host_accessible() noexcept {
  return Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                    Kokkos::DefaultExecutionSpace::memory_space>::accessible;
}

void push_profiling_region(const char* name) { Kokkos::Tools::pushRegion(std::string(name)); }

void pop_profiling_region() { Kokkos::Tools::popRegion(); }

}  // namespace imsrg::exec
