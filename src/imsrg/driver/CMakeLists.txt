# driver: the run configuration of the `imsrg` executable -- TOML schema, legacy key=value
# translation, operator requests, resolved-config writer (design/16-config-and-cli.md) -- and,
# from M8, the pipeline (design/10-architecture.md section 2, layer 6).
#
# toml++ is PRIVATE and included only by config/toml_reader.cc; public headers include neither
# toml++ nor CLI11. M8a declares only the modules the configuration layer uses (base; basis for
# nucleus names, named valence spaces and orbit labels); M8 adds the pipeline modules (io, opslib,
# hf, commutator, flow) for pipeline/*.cc and run.cc.
imsrg_add_module(NAME driver
  SOURCES
    config/config_issue.cc
    config/raw_tree.cc
    config/schema_table.cc
    config/toml_reader.cc
    config/resolve.cc
    config/validate.cc
    config/writer.cc
    config/load_config.cc
    legacy/legacy_parameters.cc
    operators/operator_request.cc
    pipeline/charge_radius.cc
    pipeline/checkpoint.cc
    pipeline/hamiltonian.cc
    pipeline/mpi_sync.cc
    run.cc
  DEPS base exec comm basis operator io opslib hf mbpt commutator flow
  PRIVATE_LIBS tomlplusplus::tomlplusplus)

# Version and git hash for the provenance record, on the one file that uses them (a new commit
# changes the hash only at the next configure).
find_package(Git QUIET)
set(_imsrg_git_hash "unknown")
if(GIT_FOUND)
  execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse --short=12 HEAD
                  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
                  OUTPUT_VARIABLE _imsrg_git_out OUTPUT_STRIP_TRAILING_WHITESPACE
                  RESULT_VARIABLE _imsrg_git_rc ERROR_QUIET)
  if(_imsrg_git_rc EQUAL 0 AND NOT _imsrg_git_out STREQUAL "")
    set(_imsrg_git_hash "${_imsrg_git_out}")
  endif()
endif()
set_property(SOURCE config/load_config.cc APPEND PROPERTY COMPILE_DEFINITIONS
             "IMSRG_VERSION_STRING=\"${PROJECT_VERSION}\"" "IMSRG_GIT_HASH=\"${_imsrg_git_hash}\"")
