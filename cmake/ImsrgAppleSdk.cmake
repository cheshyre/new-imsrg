# ImsrgAppleSdk.cmake -- choose a macOS SDK that the installed linker can actually use.
#
# Why: on the development machine (Command Line Tools as of 2026-11) xcrun selects MacOSX27.0.sdk by
# default, but its .tbd stubs list the architecture "arm64e.x1-macos", which ld-1267 cannot parse
# ("tapi error: malformed file ... unknown architecture"). Every link then fails, including CMake's
# own compiler check. MacOSX.sdk (-> MacOSX26.5.sdk) links fine.
#
# What: on macOS, when neither CMAKE_OSX_SYSROOT nor $SDKROOT is set, link a trivial C++ program with
# the default SDK. If that fails, try the CommandLineTools/Xcode SDKs (MacOSX.sdk first, then the
# versioned ones, newest first) and cache CMAKE_OSX_SYSROOT as the first one that links. Nothing
# changes when the default SDK works, so the workaround switches itself off once the toolchain is
# fixed. Set -DIMSRG_SKIP_SDK_PROBE=ON to disable it.
#
# Use: include() it before project() in the top-level CMakeLists.txt, or preload it into another
# CMake project with `cmake -C cmake/ImsrgAppleSdk.cmake ...` (tools/bootstrap_deps.sh does this for
# the Kokkos build) so that every build on the machine uses the same SDK.

if(NOT CMAKE_HOST_APPLE OR CMAKE_OSX_SYSROOT OR IMSRG_SKIP_SDK_PROBE)
  return()
endif()
if(NOT "$ENV{SDKROOT}" STREQUAL "")
  return()
endif()

if(CMAKE_CXX_COMPILER)
  set(_imsrg_sdk_cxx "${CMAKE_CXX_COMPILER}")
elseif(NOT "$ENV{CXX}" STREQUAL "")
  set(_imsrg_sdk_cxx "$ENV{CXX}")
else()
  set(_imsrg_sdk_cxx "c++")
endif()

if(CMAKE_BINARY_DIR)
  set(_imsrg_sdk_dir "${CMAKE_BINARY_DIR}/CMakeFiles/imsrg_sdk_probe")
else()
  string(RANDOM LENGTH 8 _imsrg_sdk_rand)
  set(_imsrg_sdk_dir "/tmp/imsrg_sdk_probe_${_imsrg_sdk_rand}")
endif()
file(MAKE_DIRECTORY "${_imsrg_sdk_dir}")
file(WRITE "${_imsrg_sdk_dir}/probe.cpp"
     "#include <string>\nint main() { std::string s(\"ok\"); return static_cast<int>(s.size()) - 2; }\n")

# Returns TRUE in <out> if a program links with the given SDK ("" = the compiler's default).
function(_imsrg_sdk_links sdk out)
  set(_args "${_imsrg_sdk_dir}/probe.cpp" -o "${_imsrg_sdk_dir}/probe")
  if(sdk)
    list(PREPEND _args -isysroot "${sdk}")
  endif()
  execute_process(COMMAND "${_imsrg_sdk_cxx}" ${_args}
                  RESULT_VARIABLE _rc OUTPUT_QUIET ERROR_QUIET)
  if(_rc EQUAL 0)
    set(${out} TRUE PARENT_SCOPE)
  else()
    set(${out} FALSE PARENT_SCOPE)
  endif()
endfunction()

_imsrg_sdk_links("" _imsrg_sdk_default_ok)
if(NOT _imsrg_sdk_default_ok)
  set(_imsrg_sdk_roots "/Library/Developer/CommandLineTools/SDKs"
      "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs")
  set(_imsrg_sdk_candidates "")
  foreach(_root IN LISTS _imsrg_sdk_roots)
    if(EXISTS "${_root}/MacOSX.sdk")
      list(APPEND _imsrg_sdk_candidates "${_root}/MacOSX.sdk")
    endif()
    file(GLOB _versioned LIST_DIRECTORIES TRUE "${_root}/MacOSX[0-9]*.sdk")
    list(SORT _versioned COMPARE NATURAL ORDER DESCENDING)
    list(APPEND _imsrg_sdk_candidates ${_versioned})
  endforeach()
  set(_imsrg_sdk_found "")
  foreach(_sdk IN LISTS _imsrg_sdk_candidates)
    _imsrg_sdk_links("${_sdk}" _ok)
    if(_ok)
      set(_imsrg_sdk_found "${_sdk}")
      break()
    endif()
  endforeach()
  if(_imsrg_sdk_found)
    message(STATUS "imsrg: the default macOS SDK does not link; using ${_imsrg_sdk_found} "
                   "(cmake/ImsrgAppleSdk.cmake)")
    set(CMAKE_OSX_SYSROOT "${_imsrg_sdk_found}" CACHE PATH
        "macOS SDK, auto-selected by cmake/ImsrgAppleSdk.cmake because the default SDK does not link")
  else()
    message(WARNING "imsrg: no macOS SDK links with ${_imsrg_sdk_cxx}; set CMAKE_OSX_SYSROOT "
                    "or SDKROOT by hand")
  endif()
endif()
file(REMOVE_RECURSE "${_imsrg_sdk_dir}")
