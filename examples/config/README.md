# Example configurations

Verbatim copies of the examples in `design/16-config-and-cli.md` section 2.11
(`tests/unit/driver/test_golden_equivalence.cc` checks that they stay identical).

Relative paths in a TOML config resolve against the directory of the config
file (design/16 section 1.8): the examples name their input files without a
directory, so copy a config next to the interaction files (or edit the paths)
and run `imsrg --config <file>`. The equivalent legacy command lines of the
document resolve the same names against the working directory.
`output.directory = "output"` in example (a) is relative to the config file too;
without the key the output directory defaults to `output` in the working
directory. `resolved_config.toml` records absolute paths.
