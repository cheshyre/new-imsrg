# nanobind extension imsrg._core (design/18 section 2). Python is a consumer of the library
# modules, not a module of the DAG. nanobind comes from the extern/nanobind submodule.
set(Python_EXECUTABLE "${Python3_EXECUTABLE}")
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/nanobind" "${CMAKE_BINARY_DIR}/extern/nanobind")

nanobind_add_module(_core NB_STATIC
  src/module.cc src/runtime.cc src/basis.cc src/operator.cc src/config.cc)
target_include_directories(_core PRIVATE src)
target_link_libraries(_core PRIVATE imsrg::operator imsrg::driver imsrg::exec imsrg::comm)
target_compile_features(_core PRIVATE cxx_std_20)

# Development layout: <build>/python/imsrg/{__init__.py,_core*.so}, so that
# PYTHONPATH=<build>/python works (tools/ci.sh, tests/python).
set(_pkg_dir "${CMAKE_CURRENT_BINARY_DIR}/imsrg")
set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${_pkg_dir}")
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
