// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/operator/tensor_dump.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <istream>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "imsrg/base/error.h"
#include "imsrg/basis/two_body_channels.h"
#include "imsrg/operator/operator_spec.h"

namespace imsrg::op {

namespace {

std::string Fmt17(double v) {
  char buf[40];
  std::snprintf(buf, sizeof(buf), "%.17g", v);
  return buf;
}

[[noreturn]] void Fail(std::string_view name, std::int64_t line,
                       const std::string& msg) {
  throw InputError(std::string(name) + ":" + std::to_string(line) + ": " + msg);
}

/// Splits a line into whitespace-separated tokens.
std::vector<std::string> Tokens(const std::string& line) {
  std::istringstream is(line);
  std::vector<std::string> t;
  std::string w;
  while (is >> w) t.push_back(w);
  return t;
}

std::int32_t ToInt(std::string_view name, std::int64_t line,
                   const std::string& s) {
  try {
    std::size_t pos = 0;
    const int v = std::stoi(s, &pos);
    if (pos != s.size()) Fail(name, line, "bad integer '" + s + "'");
    return v;
  } catch (const std::logic_error&) {
    Fail(name, line, "bad integer '" + s + "'");
  }
}

double ToDouble(std::string_view name, std::int64_t line,
                const std::string& s) {
  try {
    std::size_t pos = 0;
    const double v = std::stod(s, &pos);
    if (pos != s.size()) Fail(name, line, "bad number '" + s + "'");
    return v;
  } catch (const std::logic_error&) {
    Fail(name, line, "bad number '" + s + "'");
  }
}

}  // namespace

Operator ReadTensorDump(std::istream& in, std::string_view name,
                        std::shared_ptr<const basis::ModelSpace> ms) {
  if (!ms) throw InvariantError("ReadTensorDump: null model space");
  const auto& basis = ms->basis();
  const auto& tb = ms->two_body();
  std::string line;
  std::int64_t ln = 0;
  bool have_spec = false;
  OperatorSpec spec;
  std::vector<OrbitIndex> map;  // dump orbit -> model-space orbit
  Operator op;
  ChannelIndex ch_bra{-1};
  ChannelIndex ch_ket{-1};
  bool header_ok = false;
  bool ended = false;
  while (std::getline(in, line)) {
    ++ln;
    if (ln == 1) {
      if (line.rfind("# imsrg-tensor-dump raw v0", 0) != 0)
        Fail(name, ln, "not an imsrg-tensor-dump raw v0 file");
      header_ok = true;
      continue;
    }
    const auto t = Tokens(line);
    if (t.empty()) continue;
    const std::string& key = t[0];
    if (key == "spec") {
      // spec rank_J L parity p delta_Tz t hermiticity H particle_rank n
      // reduced r
      if (t.size() != 13) Fail(name, ln, "malformed spec record");
      spec.rank_J = ToInt(name, ln, t[2]);
      spec.parity = ToInt(name, ln, t[4]) == 0 ? Parity::kEven : Parity::kOdd;
      spec.delta_Tz = ToInt(name, ln, t[6]);
      if (t[8].size() != 1) Fail(name, ln, "bad hermiticity code");
      spec.hermiticity = HermiticityFromCode(t[8][0]);
      spec.particle_rank = ToInt(name, ln, t[10]);
      const std::int32_t reduced = ToInt(name, ln, t[12]);
      if (reduced != (spec.is_reduced() ? 1 : 0))
        Fail(name, ln, "reduced flag does not match the spec");
      Validate(spec);
      have_spec = true;
    } else if (key == "norbits") {
      if (t.size() != 2) Fail(name, ln, "malformed norbits record");
      map.reserve(static_cast<std::size_t>(ToInt(name, ln, t[1])));
    } else if (key == "orbit") {
      if (t.size() != 7) Fail(name, ln, "malformed orbit record");
      if (ToInt(name, ln, t[1]) != static_cast<std::int32_t>(map.size()))
        Fail(name, ln, "orbits out of order");
      basis::Orbit o;
      o.n = ToInt(name, ln, t[2]);
      o.l = ToInt(name, ln, t[3]);
      o.j2 = TwoJ{ToInt(name, ln, t[4])};
      o.tz2 = TwoTz{ToInt(name, ln, t[5])};
      const auto idx = basis.find(o);
      if (!idx) Fail(name, ln, "orbit not in the model space");
      map.push_back(*idx);
    } else if (key == "zero_body") {
      if (!have_spec) Fail(name, ln, "zero_body before spec");
      if (t.size() != 2) Fail(name, ln, "malformed zero_body record");
      op = Operator(ms, spec);
      op.zero_body() = ToDouble(name, ln, t[1]);
    } else if (key == "one_body") {
      if (op.empty()) Fail(name, ln, "one_body before zero_body");
    } else if (key == "1b") {
      if (op.empty() || t.size() != 4) Fail(name, ln, "malformed 1b record");
      const auto a = static_cast<std::size_t>(ToInt(name, ln, t[1]));
      const auto b = static_cast<std::size_t>(ToInt(name, ln, t[2]));
      if (a >= map.size() || b >= map.size())
        Fail(name, ln, "orbit index out of range");
      op.one_body().set_element(map[a], map[b], ToDouble(name, ln, t[3]));
    } else if (key == "two_body") {
      if (op.empty()) Fail(name, ln, "two_body before zero_body");
    } else if (key == "block") {
      if (op.empty() || t.size() != 10) Fail(name, ln, "malformed block");
      ch_bra = ChannelIndex{ToInt(name, ln, t[1])};
      ch_ket = ChannelIndex{ToInt(name, ln, t[5])};
      if (ch_bra.value() < 0 || ch_bra.value() >= tb.num_channels() ||
          ch_ket.value() < 0 || ch_ket.value() >= tb.num_channels())
        Fail(name, ln, "channel out of range");
      const auto& cb = tb.channel(ch_bra);
      const auto& ck = tb.channel(ch_ket);
      if (cb.J() != ToInt(name, ln, t[2]) || cb.Tz() != ToInt(name, ln, t[4]) ||
          ck.J() != ToInt(name, ln, t[6]) || ck.Tz() != ToInt(name, ln, t[8]) ||
          static_cast<std::int32_t>(cb.parity) != ToInt(name, ln, t[3]) ||
          static_cast<std::int32_t>(ck.parity) != ToInt(name, ln, t[7]))
        Fail(name, ln, "channel labels do not match the model space");
      if (!op.two_body().block_index(ch_bra, ch_ket))
        Fail(name, ln, "block not stored for this spec");
    } else if (key == "2b") {
      if (op.empty() || t.size() != 6 || ch_bra.value() < 0)
        Fail(name, ln, "malformed 2b record");
      std::size_t o[4];
      for (int k = 0; k < 4; ++k) {
        o[k] = static_cast<std::size_t>(
            ToInt(name, ln, t[static_cast<std::size_t>(1 + k)]));
        if (o[k] >= map.size()) Fail(name, ln, "orbit index out of range");
      }
      const auto bra =
          tb.local_index(ch_bra, tb.ket_index(map[o[0]], map[o[1]]));
      const auto ket =
          tb.local_index(ch_ket, tb.ket_index(map[o[2]], map[o[3]]));
      if (!bra || !ket || map[o[0]] > map[o[1]] || map[o[2]] > map[o[3]])
        Fail(name, ln, "ket not in its channel");
      op.two_body().set(ch_bra, ch_ket, *bra, *ket, ToDouble(name, ln, t[5]));
    } else if (key == "end") {
      ended = true;
      break;
    }
  }
  if (!header_ok) Fail(name, ln, "empty file");
  if (op.empty()) Fail(name, ln, "no zero_body record");
  if (!ended) Fail(name, ln, "missing end record");
  op.SyncToDevice();
  return op;
}

Operator ReadTensorDump(const std::filesystem::path& path,
                        std::shared_ptr<const basis::ModelSpace> ms) {
  std::ifstream in(path);
  if (!in) throw InputError("cannot read " + path.string());
  return ReadTensorDump(in, path.string(), std::move(ms));
}

void WriteTensorDump(std::ostream& out, const Operator& op,
                     const std::string& operator_name) {
  Operator src(op);
  src.SyncToHost();
  const Operator& c = src;
  const auto& ms = c.model_space();
  const auto& basis = ms.basis();
  const auto& tb = ms.two_body();
  const OperatorSpec& s = c.spec();
  out << "# imsrg-tensor-dump raw v0 (written by the imsrg rewrite)\n";
  out << "operator " << operator_name << "\n";
  out << "spec rank_J " << s.rank_J << " parity " << static_cast<int>(s.parity)
      << " delta_Tz " << s.delta_Tz << " hermiticity "
      << HermiticityCode(s.hermiticity) << " particle_rank " << s.particle_rank
      << " reduced " << (s.is_reduced() ? 1 : 0) << "\n";
  const std::int32_t n = basis.num_orbits();
  out << "norbits " << n << "\n";
  for (std::int32_t i = 0; i < n; ++i) {
    const auto& o = basis.orbit(OrbitIndex{i});
    out << "orbit " << i << " " << o.n << " " << o.l << " " << o.j2.twice << " "
        << o.tz2.twice << " " << Fmt17(ms.reference().occupation(OrbitIndex{i}))
        << "\n";
  }
  out << "zero_body " << Fmt17(c.zero_body()) << "\n";
  std::vector<std::string> l1;
  for (std::int32_t a = 0; a < n; ++a)
    for (std::int32_t b = 0; b < n; ++b) {
      const double v = c.one_body().get(OrbitIndex{a}, OrbitIndex{b});
      if (v != 0.0)
        l1.push_back("1b " + std::to_string(a) + " " + std::to_string(b) + " " +
                     Fmt17(v));
    }
  out << "one_body " << l1.size() << "\n";
  for (const auto& x : l1) out << x << "\n";
  std::vector<std::string> l2;
  std::int64_t nblocks = 0;
  if (c.two_body().allocated()) {
    for (std::size_t k = 0; k < c.two_body().matrix().num_blocks(); ++k) {
      const auto& info = c.two_body().matrix().block_info(k);
      const ChannelIndex cb{info.row_block.value()};
      const ChannelIndex ck{info.col_block.value()};
      const auto blk = c.two_body().matrix().host_block(k);
      std::vector<std::string> lines;
      for (std::int64_t i = 0; i < info.rows; ++i)
        for (std::int64_t j = 0; j < info.cols; ++j) {
          const double v = blk(i, j);
          if (v == 0.0) continue;
          const auto& bra = tb.ket(
              tb.global_ket(cb, LocalKetIndex{static_cast<std::int32_t>(i)}));
          const auto& ket = tb.ket(
              tb.global_ket(ck, LocalKetIndex{static_cast<std::int32_t>(j)}));
          lines.push_back("2b " + std::to_string(bra.p.value()) + " " +
                          std::to_string(bra.q.value()) + " " +
                          std::to_string(ket.p.value()) + " " +
                          std::to_string(ket.q.value()) + " " + Fmt17(v));
        }
      if (lines.empty()) continue;
      const auto& chb = tb.channel(cb);
      const auto& chk = tb.channel(ck);
      l2.push_back("block " + std::to_string(cb.value()) + " " +
                   std::to_string(chb.J()) + " " +
                   std::to_string(static_cast<int>(chb.parity)) + " " +
                   std::to_string(chb.Tz()) + " " + std::to_string(ck.value()) +
                   " " + std::to_string(chk.J()) + " " +
                   std::to_string(static_cast<int>(chk.parity)) + " " +
                   std::to_string(chk.Tz()) + " " +
                   std::to_string(lines.size()));
      for (auto& x : lines) l2.push_back(std::move(x));
      ++nblocks;
    }
  }
  const std::int64_t count = static_cast<std::int64_t>(l2.size()) - nblocks;
  out << "two_body " << count << " blocks " << nblocks << "\n";
  for (const auto& x : l2) out << x << "\n";
  out << "end\n";
}

void WriteTensorDump(const std::filesystem::path& path, const Operator& op,
                     const std::string& operator_name) {
  std::ofstream out(path);
  if (!out) throw InputError("cannot write " + path.string());
  WriteTensorDump(out, op, operator_name);
  if (!out) throw InputError("error writing " + path.string());
}

}  // namespace imsrg::op
