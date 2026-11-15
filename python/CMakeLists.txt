# nanobind extension imsrg._core (design/18 section 2). Python is a consumer of the library
# modules, not a module of the DAG. nanobind comes from the extern/nanobind submodule.
set(Python_EXECUTABLE "${Python3_EXECUTABLE}")
# Python >= 3.12 (D29). Development.SABIModule (CMake >= 3.26) provides Python::SABIModule, which
# nanobind needs for a stable-ABI ('abi3') build of _core; with CMake 3.25 the module is built
# for the running interpreter only.
set(_imsrg_py_dev Development.Module)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.26)
  list(APPEND _imsrg_py_dev Development.SABIModule)
endif()
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter ${_imsrg_py_dev})
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/nanobind" "${CMAKE_BINARY_DIR}/extern/nanobind")

nanobind_add_module(_core NB_STATIC STABLE_ABI
  src/module.cc src/runtime.cc src/basis.cc src/operator.cc src/config.cc)
# Binding-local headers are included by their path from the repository root
# ("python/src/bindings.h").
target_include_directories(_core PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(_core PRIVATE imsrg::operator imsrg::driver imsrg::exec imsrg::comm)
target_compile_features(_core PRIVATE cxx_std_20)

# Development layout: <build>/python/imsrg/{__init__.py,_core*.so}, so that
# PYTHONPATH=<build>/python works (tools/ci.sh, tests/python).
set(_pkg_dir "${CMAKE_CURRENT_BINARY_DIR}/imsrg")
set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${_pkg_dir}")
# Remove extension files with another suffix (e.g. a version-specific _core.cpython-3xx-*.so from
# before the stable-ABI switch): Python's import prefers the version-specific name, so a stale file
# would silently shadow the freshly built module.
get_target_property(_core_suffix _core SUFFIX)
file(GLOB _stale_core "${_pkg_dir}/_core.*")
foreach(f ${_stale_core})
  get_filename_component(_n "${f}" NAME)
  if(NOT _n STREQUAL "_core${_core_suffix}")
    file(REMOVE "${f}")
  endif()
endforeach()
foreach(cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY_${cfg} "${_pkg_dir}")
endforeach()
file(GLOB _py_files "${CMAKE_CURRENT_SOURCE_DIR}/imsrg/*.py" "${CMAKE_CURRENT_SOURCE_DIR}/imsrg/py.typed")
foreach(f ${_py_files})
  get_filename_component(_n "${f}" NAME)
  configure_file("${f}" "${_pkg_dir}/${_n}" COPYONLY)
endforeach()

# scikit-build-core install (pip install -e . --no-build-isolation).
install(TARGETS _core LIBRARY DESTINATION imsrg)
install(FILES ${_py_files} DESTINATION imsrg)
