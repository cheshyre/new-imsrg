# ImsrgCompilerFlags.cmake -- compiler launcher, global instrumentation flags, and the warning set.
#
# Global flags (all targets, including bundled dependencies): ccache, sanitizers, native arch.
# Per-target flags: link `imsrg::warnings` PRIVATE (imsrg_add_module and imsrg_add_test do this) to
# get -Wall -Wextra -Wpedantic -Wshadow -Wconversion (design/10 section 5), plus -Werror when
# IMSRG_WERROR is ON. Third-party code never sees these warnings: it is added as SYSTEM.

include_guard(GLOBAL)

# ---- ccache ---------------------------------------------------------------------------------------
if(IMSRG_USE_CCACHE AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
  find_program(IMSRG_CCACHE_PROGRAM ccache)
  if(IMSRG_CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${IMSRG_CCACHE_PROGRAM}")
  endif()
endif()

# ---- sanitizers -----------------------------------------------------------------------------------
string(REPLACE "," ";" _imsrg_sanitizers "${IMSRG_SANITIZE}")
list(REMOVE_ITEM _imsrg_sanitizers "none" "")
if(_imsrg_sanitizers)
  foreach(_s IN LISTS _imsrg_sanitizers)
    if(NOT _s MATCHES "^(address|undefined|thread)$")
      message(FATAL_ERROR "IMSRG_SANITIZE: unknown sanitizer '${_s}' (address, undefined, thread)")
    endif()
  endforeach()
  if("thread" IN_LIST _imsrg_sanitizers AND "address" IN_LIST _imsrg_sanitizers)
    message(FATAL_ERROR "IMSRG_SANITIZE: address and thread cannot be combined")
  endif()
  list(JOIN _imsrg_sanitizers "," _imsrg_sanitize_flag)
  add_compile_options(-fsanitize=${_imsrg_sanitize_flag} -fno-omit-frame-pointer)
  add_link_options(-fsanitize=${_imsrg_sanitize_flag})
  if("undefined" IN_LIST _imsrg_sanitizers)
    add_compile_options(-fno-sanitize-recover=undefined)  # UB fails the test instead of a warning
  endif()
endif()

# ---- native architecture ----------------------------------------------------------------------------
if(IMSRG_NATIVE_ARCH)
  include(CheckCXXCompilerFlag)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64)$")
    set(_imsrg_native_flag "-mcpu=native")
  else()
    set(_imsrg_native_flag "-march=native")
  endif()
  check_cxx_compiler_flag("${_imsrg_native_flag}" IMSRG_HAVE_NATIVE_ARCH_FLAG)
  if(IMSRG_HAVE_NATIVE_ARCH_FLAG)
    add_compile_options("${_imsrg_native_flag}")
  else()
    message(WARNING "IMSRG_NATIVE_ARCH: the compiler does not accept ${_imsrg_native_flag}")
  endif()
endif()

# ---- warnings (per target) ------------------------------------------------------------------------
add_library(imsrg_warnings INTERFACE)
add_library(imsrg::warnings ALIAS imsrg_warnings)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU|IntelLLVM")
  target_compile_options(imsrg_warnings INTERFACE
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion)
  if(IMSRG_WERROR)
    target_compile_options(imsrg_warnings INTERFACE -Werror)
  endif()
endif()
