// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include "tests/unit/basis/golden_basis_dump.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace imsrg::basis::testing {

namespace {

// Line-oriented reader; every failure carries the file name and the 1-based
// line number.
class Reader {
 public:
  explicit Reader(const std::filesystem::path& file) : file_(file) {
    std::ifstream in(file);
    if (!in) throw std::runtime_error(file.string() + ": cannot open file");
    std::string line;
    while (std::getline(in, line)) lines_.push_back(line);
  }

  [[noreturn]] void Fail(const std::string& why) const {
    throw std::runtime_error(file_.filename().string() + ":" +
                             std::to_string(line_no_) + ": " + why);
  }

  // Advances to the next line, splits it into tokens and requires the first one
  // to be `keyword`.
  std::vector<std::string> Record(const std::string& keyword) {
    if (pos_ >= lines_.size()) {
      line_no_ = static_cast<int>(lines_.size()) + 1;
      Fail("unexpected end of file, expected record '" + keyword + "'");
    }
    line_no_ = static_cast<int>(pos_) + 1;
    const std::string& line = lines_[pos_++];
    std::istringstream is(line);
    std::vector<std::string> tok;
    for (std::string t; is >> t;) tok.push_back(t);
    if (tok.empty() || tok[0] != keyword) {
      Fail("expected record '" + keyword + "', found '" + line + "'");
    }
    return tok;
  }

  const std::string& raw_line() const { return lines_[pos_ - 1]; }
  bool at_end() const { return pos_ >= lines_.size(); }
  bool empty_file() const { return lines_.empty(); }
  void FailTrailing() {
    line_no_ = static_cast<int>(pos_) + 1;
    Fail("unexpected content after 'end'");
  }

  int ToInt(const std::string& s) const {
    char* end = nullptr;
    errno = 0;
    const long v = std::strtol(s.c_str(), &end, 10);
    if (s.empty() || *end != '\0' || errno != 0 || v < -2147483647L ||
        v > 2147483647L) {
      Fail("not an integer: '" + s + "'");
    }
    return static_cast<int>(v);
  }
  long long ToLl(const std::string& s) const {
    char* end = nullptr;
    errno = 0;
    const long long v = std::strtoll(s.c_str(), &end, 10);
    if (s.empty() || *end != '\0' || errno != 0)
      Fail("not an integer: '" + s + "'");
    return v;
  }
  double ToDouble(const std::string& s) const {
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (s.empty() || *end != '\0') Fail("not a number: '" + s + "'");
    return v;
  }

  void ExpectSize(const std::vector<std::string>& tok, std::size_t n) const {
    if (tok.size() != n) {
      Fail("record '" + tok[0] + "' has " + std::to_string(tok.size()) +
           " fields, expected " + std::to_string(n));
    }
  }
  void ExpectWord(const std::vector<std::string>& tok, std::size_t at,
                  const std::string& name) const {
    if (tok.size() <= at || tok[at] != name) {
      Fail("record '" + tok[0] + "': expected field name '" + name +
           "' at position " + std::to_string(at));
    }
  }

  // `<count> <indices...>` starting at tok[first]; the count must equal the
  // number of indices.
  std::vector<int> CountedList(const std::vector<std::string>& tok,
                               std::size_t first) const {
    if (tok.size() <= first) Fail("record '" + tok[0] + "': missing count");
    const int count = ToInt(tok[first]);
    if (count < 0) Fail("record '" + tok[0] + "': negative count");
    if (tok.size() != first + 1 + static_cast<std::size_t>(count)) {
      Fail("record '" + tok[0] + "': count " + std::to_string(count) + " but " +
           std::to_string(tok.size() - first - 1) + " indices");
    }
    std::vector<int> v;
    for (std::size_t i = first + 1; i < tok.size(); ++i)
      v.push_back(ToInt(tok[i]));
    return v;
  }

  // Count record `<keyword> N`.
  int CountRecord(const std::string& keyword) {
    const auto tok = Record(keyword);
    ExpectSize(tok, 2);
    const int n = ToInt(tok[1]);
    if (n < 0) Fail("negative count");
    return n;
  }

 private:
  std::filesystem::path file_;
  std::vector<std::string> lines_;
  std::size_t pos_ = 0;
  int line_no_ = 1;
};

const char* const kSubsetNames[9] = {"pp", "hh", "ph", "cc", "vc",
                                     "qc", "vv", "qv", "qq"};

}  // namespace

GoldenBasisDump ReadGoldenBasisDump(const std::filesystem::path& file) {
  Reader r(file);
  GoldenBasisDump d;

  if (r.empty_file()) r.Fail("empty file");
  r.Record("#");
  if (r.raw_line() != "# imsrg-basis-dump v1")
    r.Fail("first line must be '# imsrg-basis-dump v1'");

  r.Record("source");
  d.source = r.raw_line().size() > 7 ? r.raw_line().substr(7) : "";

  {
    const auto t = r.Record("basis");
    r.ExpectSize(t, 11);
    r.ExpectWord(t, 1, "emax");
    r.ExpectWord(t, 3, "e2max");
    r.ExpectWord(t, 5, "e3max");
    r.ExpectWord(t, 7, "lmax");
    r.ExpectWord(t, 9, "hw");
    d.emax = r.ToInt(t[2]);
    d.e2max = r.ToInt(t[4]);
    d.e3max = r.ToInt(t[6]);
    d.lmax = r.ToInt(t[8]);
    d.hw = r.ToDouble(t[10]);
  }
  {
    const auto t = r.Record("reference");
    r.ExpectSize(t, 4);
    r.ExpectWord(t, 2, "valence_space");
    d.reference = t[1];
    d.valence_space = t[3];
  }
  {
    const auto t = r.Record("counts");
    r.ExpectSize(t, 13);
    const char* names[6] = {"Aref",  "Zref",        "Acore",
                            "Zcore", "target_mass", "target_Z"};
    for (std::size_t i = 0; i < 6; ++i) r.ExpectWord(t, 1 + 2 * i, names[i]);
    d.Aref = r.ToDouble(t[2]);
    d.Zref = r.ToDouble(t[4]);
    d.Acore = r.ToDouble(t[6]);
    d.Zcore = r.ToDouble(t[8]);
    d.target_mass = r.ToInt(t[10]);
    d.target_Z = r.ToInt(t[12]);
  }

  const int norb = r.CountRecord("norbits");
  for (int i = 0; i < norb; ++i) {
    const auto t = r.Record("orbit");
    r.ExpectSize(t, 8);
    GoldenOrbit o;
    o.index = r.ToInt(t[1]);
    o.n = r.ToInt(t[2]);
    o.l = r.ToInt(t[3]);
    o.j2 = r.ToInt(t[4]);
    o.tz2 = r.ToInt(t[5]);
    o.occupation = r.ToDouble(t[6]);
    o.cvq = r.ToInt(t[7]);
    d.orbits.push_back(o);
  }
  d.holes = r.CountedList(r.Record("holes"), 1);
  d.particles = r.CountedList(r.Record("particles"), 1);
  d.core = r.CountedList(r.Record("core"), 1);
  d.valence = r.CountedList(r.Record("valence"), 1);
  d.qspace = r.CountedList(r.Record("qspace"), 1);

  const int nkets = r.CountRecord("nkets");
  for (int i = 0; i < nkets; ++i) {
    const auto t = r.Record("ket");
    r.ExpectSize(t, 4);
    if (r.ToInt(t[1]) != i) {
      r.Fail("ket index " + t[1] + " out of order, expected " +
             std::to_string(i));
    }
    d.kets.emplace_back(r.ToInt(t[2]), r.ToInt(t[3]));
  }

  const int nch = r.CountRecord("ntwobody_channels");
  for (int i = 0; i < nch; ++i) {
    GoldenChannel c;
    auto t = r.Record("channel");
    r.ExpectSize(t, 6);
    c.ch = r.ToInt(t[1]);
    c.J = r.ToInt(t[2]);
    c.parity = r.ToInt(t[3]);
    c.Tz = r.ToInt(t[4]);
    c.nkets = r.ToInt(t[5]);
    if (c.ch != i)
      r.Fail("channel index out of order, expected " + std::to_string(i));
    t = r.Record("channel_kets");
    if (t.size() < 2 || r.ToInt(t[1]) != c.ch)
      r.Fail("channel_kets for the wrong channel");
    for (std::size_t k = 2; k < t.size(); ++k) c.kets.push_back(r.ToInt(t[k]));
    if (static_cast<int>(c.kets.size()) != c.nkets) {
      r.Fail("channel " + std::to_string(c.ch) + " declares " +
             std::to_string(c.nkets) + " kets but channel_kets lists " +
             std::to_string(c.kets.size()));
    }
    for (const char* name : kSubsetNames) {
      t = r.Record("channel_subset");
      if (t.size() < 4 || r.ToInt(t[1]) != c.ch)
        r.Fail("channel_subset for the wrong channel");
      if (t[2] != name) {
        r.Fail(std::string("expected subset '") + name + "', found '" + t[2] +
               "'");
      }
      c.subsets[name] = r.CountedList(t, 3);
    }
    d.channels.push_back(std::move(c));
  }

  const int nph = r.CountRecord("nph_channels");
  for (int i = 0; i < nph; ++i) {
    GoldenPhChannel c;
    auto t = r.Record("ph_channel");
    r.ExpectSize(t, 6);
    c.ch = r.ToInt(t[1]);
    c.J = r.ToInt(t[2]);
    c.parity = r.ToInt(t[3]);
    c.Tz = r.ToInt(t[4]);
    c.nkets = r.ToInt(t[5]);
    if (c.ch != i)
      r.Fail("ph_channel index out of order, expected " + std::to_string(i));
    t = r.Record("ph_channel_kets");
    if (t.size() < 2 || r.ToInt(t[1]) != c.ch)
      r.Fail("ph_channel_kets for the wrong channel");
    for (std::size_t k = 2; k < t.size(); ++k) c.kets.push_back(r.ToInt(t[k]));
    if (static_cast<int>(c.kets.size()) != c.nkets) {
      r.Fail("ph_channel " + std::to_string(c.ch) + " declares " +
             std::to_string(c.nkets) + " kets but ph_channel_kets lists " +
             std::to_string(c.kets.size()));
    }
    d.ph_channels.push_back(std::move(c));
  }

  const int n3 = r.CountRecord("nthreebody_channels");
  for (int i = 0; i < n3; ++i) {
    const auto t = r.Record("threebody_channel");
    r.ExpectSize(t, 6);
    GoldenChannel3 c;
    c.ch = r.ToInt(t[1]);
    c.twoJ = r.ToInt(t[2]);
    c.parity = r.ToInt(t[3]);
    c.twoTz = r.ToInt(t[4]);
    c.nkets = r.ToInt(t[5]);
    if (c.ch != i)
      r.Fail("threebody_channel index out of order, expected " +
             std::to_string(i));
    d.channels3.push_back(c);
  }
  {
    const auto t = r.Record("nkets3");
    r.ExpectSize(t, 4);
    r.ExpectWord(t, 2, "sum_channel_kets3");
    d.nkets3 = r.ToLl(t[1]);
    d.sum_channel_kets3 = r.ToLl(t[3]);
  }
  r.ExpectSize(r.Record("end"), 1);
  if (!r.at_end()) r.FailTrailing();
  return d;
}

std::filesystem::path GoldenBasisDir() {
  return std::filesystem::path(IMSRG_GOLDEN_DIR) / "basis";
}

std::vector<std::string> GoldenBasisFiles() {
  std::vector<std::string> names;
  for (const char* e : {"emax2", "emax4"}) {
    for (const char* s :
         {"He4", "Ne20", "Ne20_sd-shell", "O16", "p-shell", "sd-shell"}) {
      names.push_back(std::string(e) + "_" + s + ".txt");
    }
  }
  names.push_back("emax4_O16_e3max7.txt");
  names.push_back("emax4_Ne20_sd-shell_lmax2_e3max8.txt");
  std::sort(names.begin(), names.end());
  return names;
}

}  // namespace imsrg::basis::testing
