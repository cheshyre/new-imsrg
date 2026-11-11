# Golden data for the valence-space writers (M5, io)

Reference version: `imsrg-devel src-sha256:ddcbf1aff166414e746113d5f3de2ddc1e72783fb7c201829cd58e317717f40f`
(see `../README.md`). Regenerate with

    REFHARNESS=$PWD/build/refharness/refharness tools/reference/gen_golden_io.sh

which runs `refharness write-valence ../hf/emax2_O16_2N/HNO.dump valence_space=sd-shell out=PREFIX
[mode=op]`: the dump (O16, emax 2, HF basis, 2N only) is read into the old `Operator` on the O16 /
sd-shell model space, then `ReadWrite::WriteTokyo` (ReadWrite.cc:5841), `WriteNuShellX_int` /
`WriteNuShellX_intfile` (:3610) and `WriteNuShellX_sps` (:3753) write the files.

| files | content |
|---|---|
| `O16_sdshell.{snt,int,sp}` | Hamiltonian mode (mode "" / "int"): matrix elements as stored |
| `O16_sdshell_op.{snt,int,sp}` | mode "op": proton-neutron matrix elements averaged with their isospin partners |

Valence orbits are numbered protons first (d5/2, d3/2, s1/2), then neutrons. The first comment line
of every file carries the build version string of the old code run. The `.int` comment records
`Aref: 0 Zref: 0` because imsrg++ leaves those members unset when only the writer is called. The
test (`tests/unit/io/test_valence_writers.cpp`) ignores the first line, compares the other comments
exactly and numbers within 1e-10.
