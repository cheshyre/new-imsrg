# ImsrgModule.cmake -- the one way to define an imsrg library module (design/20 section 2).
#
#   imsrg_add_module(NAME <module>
#                    [SOURCES <file>...]       # .cpp files, kernels/ included; none = header-only
#                    [HEADERS <file>...]       # optional, for IDEs
#                    [DEPS <module>...]        # imsrg modules (short names), linked PUBLIC
#                    [LIBS <target>...]        # external targets, PUBLIC (e.g. fmt::fmt)
#                    [PRIVATE_LIBS <target>...]) # external targets, PRIVATE (e.g. MPI::MPI_CXX)
#
# Defines the static library imsrg_<module> with alias imsrg::<module>, include root src/ (and the
# generated-header root <build>/src) so headers are included as "imsrg/<module>/x.hpp", the warning
# set of imsrg::warnings (with -Werror under IMSRG_WERROR), and records the module's DEPS for
# tools/check_layering.py (imsrg_write_module_deps). A module may only list modules that the DAG in
# design/10-architecture.md section 2 allows; the layering test fails otherwise.

include_guard(GLOBAL)

function(imsrg_add_module)
  cmake_parse_arguments(PARSE_ARGV 0 ARG "" "NAME" "SOURCES;HEADERS;DEPS;LIBS;PRIVATE_LIBS")
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "imsrg_add_module: unexpected arguments '${ARG_UNPARSED_ARGUMENTS}'")
  endif()
  if(NOT ARG_NAME MATCHES "^[a-z][a-z0-9_]*$")
    message(FATAL_ERROR "imsrg_add_module: NAME '${ARG_NAME}' must be snake_case")
  endif()
  set(_target "imsrg_${ARG_NAME}")
  if(TARGET ${_target})
    message(FATAL_ERROR "imsrg_add_module: module '${ARG_NAME}' is defined twice")
  endif()

  if(ARG_SOURCES)
    add_library(${_target} STATIC ${ARG_SOURCES} ${ARG_HEADERS})
    set(_scope PUBLIC)
    target_link_libraries(${_target} PRIVATE imsrg::warnings)
    if(IMSRG_UNITY_BUILD)
      set_target_properties(${_target} PROPERTIES UNITY_BUILD ON)
    endif()
  else()
    add_library(${_target} INTERFACE ${ARG_HEADERS})
    set(_scope INTERFACE)
    if(ARG_PRIVATE_LIBS)
      message(FATAL_ERROR "imsrg_add_module(${ARG_NAME}): PRIVATE_LIBS needs SOURCES")
    endif()
  endif()
  add_library(imsrg::${ARG_NAME} ALIAS ${_target})
  target_compile_features(${_target} ${_scope} cxx_std_20)
  target_include_directories(${_target} ${_scope}
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>"
    "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>")
  foreach(_dep IN LISTS ARG_DEPS)
    target_link_libraries(${_target} ${_scope} imsrg::${_dep})
  endforeach()
  if(ARG_LIBS)
    target_link_libraries(${_target} ${_scope} ${ARG_LIBS})
  endif()
  if(ARG_PRIVATE_LIBS)
    target_link_libraries(${_target} PRIVATE ${ARG_PRIVATE_LIBS})
  endif()

  file(RELATIVE_PATH _dir "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
  set_property(GLOBAL APPEND PROPERTY IMSRG_MODULES "${ARG_NAME}")
  set_property(GLOBAL PROPERTY "IMSRG_MODULE_DEPS_${ARG_NAME}" "${ARG_DEPS}")
  set_property(GLOBAL PROPERTY "IMSRG_MODULE_DIR_${ARG_NAME}" "${_dir}")
endfunction()

# Writes the modules recorded by imsrg_add_module as JSON for tools/check_layering.py:
#   {"version": 1, "modules": [{"name": "base", "deps": [], "dir": "src/imsrg/base"}, ...]}
function(imsrg_write_module_deps file)
  get_property(_modules GLOBAL PROPERTY IMSRG_MODULES)
  set(_entries "")
  foreach(_m IN LISTS _modules)
    get_property(_deps GLOBAL PROPERTY "IMSRG_MODULE_DEPS_${_m}")
    get_property(_dir GLOBAL PROPERTY "IMSRG_MODULE_DIR_${_m}")
    set(_quoted "")
    foreach(_d IN LISTS _deps)
      list(APPEND _quoted "\"${_d}\"")
    endforeach()
    list(JOIN _quoted ", " _deps_json)
    list(APPEND _entries "    {\"name\": \"${_m}\", \"deps\": [${_deps_json}], \"dir\": \"${_dir}\"}")
  endforeach()
  list(JOIN _entries ",\n" _body)
  if(_body STREQUAL "")
    set(_json "{\"version\": 1, \"modules\": []}\n")
  else()
    set(_json "{\"version\": 1, \"modules\": [\n${_body}\n]}\n")
  endif()
  file(CONFIGURE OUTPUT "${file}" CONTENT "${_json}" @ONLY)
endfunction()
