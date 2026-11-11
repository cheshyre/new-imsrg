# Adds every module directory src/imsrg/<module>/ that has a CMakeLists.txt; each one calls
# imsrg_add_module() (cmake/ImsrgModule.cmake). Discovery instead of a hand-kept list means that
# milestones adding modules in parallel never edit this file. tools/check_layering.py rejects
# directories that are not modules of the DAG in design/10-architecture.md section 2.
file(GLOB _imsrg_module_lists CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/imsrg/*/CMakeLists.txt")
foreach(_list IN LISTS _imsrg_module_lists)
  get_filename_component(_dir "${_list}" DIRECTORY)
  add_subdirectory("${_dir}")
endforeach()
