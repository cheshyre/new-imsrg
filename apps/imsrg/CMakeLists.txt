# The `imsrg` executable: CLI11 command line -> imsrg::driver (design/16-config-and-cli.md 2.12).
# The only place that includes CLI11.
add_executable(imsrg main.cpp)
target_link_libraries(imsrg PRIVATE imsrg::driver CLI11::CLI11 imsrg::warnings)
