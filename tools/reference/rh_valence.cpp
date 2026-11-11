// rh_valence.cpp -- golden data for the valence-space interaction writers (M5 io).
//
//   refharness write-valence IN.dump valence_space=NAME out=PREFIX [mode=op]
//
// Reads an OperatorDump v1 into the old imsrg++ Operator on the model space of the dump's
// reference with the given valence space, then calls ReadWrite::WriteTokyo (mode ""),
// WriteNuShellX_int and WriteNuShellX_sps (ReadWrite.cc:5841, 3610, 3753), producing
// PREFIX.snt, PREFIX.int and PREFIX.sp.

#include <iostream>

#include "IMSRG.hh"
#include "ReadWrite.hh"
#include "rh_common.hh"

namespace rh {

/// Writes the tokyo (.snt) and NuShellX (.int, .sp) files of IN with the old code.
int cmd_write_valence(const Args& a) {
  if (a.positional.size() != 1)
    throw std::runtime_error("usage: refharness write-valence IN.dump valence_space=NAME out=PREFIX");
  if (!a.has("valence_space") || !a.has("out"))
    throw std::runtime_error("write-valence needs valence_space=NAME and out=PREFIX");
  const std::string& in = a.positional[0];
  auto ms = modelspace_for_dump(in, a);
  Operator op = read_opdump(in, *ms);
  const std::string prefix = a.str("out", "");
  ReadWrite rw;
  const std::string mode = a.str("mode", "");  // "" (Hamiltonian) or "op" (isospin-averaged)
  rw.WriteTokyo(op, prefix + ".snt", mode);
  if (mode.empty()) rw.WriteNuShellX_int(op, prefix + ".int");
  else rw.WriteNuShellX_intfile(op, prefix + ".int", mode);
  rw.WriteNuShellX_sps(op, prefix + ".sp");
  return 0;
}

}  // namespace rh
