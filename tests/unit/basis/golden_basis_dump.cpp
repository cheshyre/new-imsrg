#include "golden_basis_dump.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace imsrg::basis::testing {

namespace {

// Line-oriented reader; every failure carries the file name and the 1-based line number.
class Reader {
 public:
  explicit Reader(const std::filesystem::path& file) : file_(file) {
    std::ifstream in(file);
    if (!in) throw std::runtime_error(file.string() + ": cannot open file");
    std::string line;
    while (std::getline(in, line)) lines_.push_back(line);
  }

  [[noreturn]] void fail(const std::string& why) const {
    throw std::runtime_error(file_.filename().string() + ":" + std::to_string(line_no_) + ": " +
                             why);
  }

  // Advances to the next line, splits it into tokens and requires the first one to be `keyword`.
  std::vector<std::string> record(const std::string& keyword) {
    if (pos_ >= lines_.size()) {
      line_no_ = static_cast<int>(lines_.size()) + 1;
      fail("unexpected end of file, expected record '" + keyword + "'");
    }
    line_no_ = static_cast<int>(pos_) + 1;
    const std::string& line = lines_[pos_++];
    std::istringstream is(line);
    std::vector<std::string> tok;
    for (std::string t; is >> t;) tok.push_back(t);
    if (tok.empty() || tok[0] != keyword) {
      fail("expected record '" + keyword + "', found '" + line + "'");
    }
    return tok;
  }

  const std::string& raw_line() const { return lines_[pos_ - 1]; }
  bool at_end() const { return pos_ >= lines_.size(); }
  bool empty_file() const { return lines_.empty(); }
  void fail_trailing() {
    line_no_ = static_cast<int>(pos_) + 1;
    fail("unexpected content after 'end'");
  }

  int to_int(const std::string& s) const {
    char* end = nullptr;
    errno = 0;
    const long v = std::strtol(s.c_str(), &end, 10);
    if (s.empty() || *end != '\0' || errno != 0 || v < -2147483647L || v > 2147483647L) {
      fail("not an integer: '" + s + "'");
    }
    return static_cast<int>(v);
  }
  long long to_ll(const std::string& s) const {
    char* end = nullptr;
    errno = 0;
    const long long v = std::strtoll(s.c_str(), &end, 10);
    if (s.empty() || *end != '\0' || errno != 0) fail("not an integer: '" + s + "'");
    return v;
  }
  double to_double(const std::string& s) const {
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (s.empty() || *end != '\0') fail("not a number: '" + s + "'");
    return v;
  }

  void expect_size(const std::vector<std::string>& tok, std::size_t n) const {
    if (tok.size() != n) {
      fail("record '" + tok[0] + "' has " + std::to_string(tok.size()) + " fields, expected " +
           std::to_string(n));
    }
  }
  void expect_word(const std::vector<std::string>& tok, std::size_t at,
                   const std::string& name) const {
    if (tok.size() <= at || tok[at] != name) {
      fail("record '" + tok[0] + "': expected field name '" + name + "' at position " +
           std::to_string(at));
    }
  }

  // `<count> <indices...>` starting at tok[first]; the count must equal the number of indices.
  std::vector<int> counted_list(const std::vector<std::string>& tok, std::size_t first) const {
    if (tok.size() <= first) fail("record '" + tok[0] + "': missing count");
    const int count = to_int(tok[first]);
    if (count < 0) fail("record '" + tok[0] + "': negative count");
    if (tok.size() != first + 1 + static_cast<std::size_t>(count)) {
      fail("record '" + tok[0] + "': count " + std::to_string(count) + " but " +
           std::to_string(tok.size() - first - 1) + " indices");
    }
    std::vector<int> v;
    for (std::size_t i = first + 1; i < tok.size(); ++i) v.push_back(to_int(tok[i]));
    return v;
  }

  // Count record `<keyword> N`.
  int count_record(const std::string& keyword) {
    const auto tok = record(keyword);
    expect_size(tok, 2);
    const int n = to_int(tok[1]);
    if (n < 0) fail("negative count");
    return n;
  }

 private:
  std::filesystem::path file_;
  std::vector<std::string> lines_;
  std::size_t pos_ = 0;
  int line_no_ = 1;
};

const char* const kSubsetNames[9] = {"pp", "hh", "ph", "cc", "vc", "qc", "vv", "qv", "qq"};

}  // namespace

GoldenBasisDump read_golden_basis_dump(const std::filesystem::path& file) {
  Reader r(file);
  GoldenBasisDump d;

  if (r.empty_file()) r.fail("empty file");
  r.record("#");
  if (r.raw_line() != "# imsrg-basis-dump v1") r.fail("first line must be '# imsrg-basis-dump v1'");

  r.record("source");
  d.source = r.raw_line().size() > 7 ? r.raw_line().substr(7) : "";

  {
    const auto t = r.record("basis");
    r.expect_size(t, 11);
    r.expect_word(t, 1, "emax");
    r.expect_word(t, 3, "e2max");
    r.expect_word(t, 5, "e3max");
    r.expect_word(t, 7, "lmax");
    r.expect_word(t, 9, "hw");
    d.emax = r.to_int(t[2]);
    d.e2max = r.to_int(t[4]);
    d.e3max = r.to_int(t[6]);
    d.lmax = r.to_int(t[8]);
    d.hw = r.to_double(t[10]);
  }
  {
    const auto t = r.record("reference");
    r.expect_size(t, 4);
    r.expect_word(t, 2, "valence_space");
    d.reference = t[1];
    d.valence_space = t[3];
  }
  {
    const auto t = r.record("counts");
    r.expect_size(t, 13);
    const char* names[6] = {"Aref", "Zref", "Acore", "Zcore", "target_mass", "target_Z"};
    for (std::size_t i = 0; i < 6; ++i) r.expect_word(t, 1 + 2 * i, names[i]);
    d.Aref = r.to_double(t[2]);
    d.Zref = r.to_double(t[4]);
    d.Acore = r.to_double(t[6]);
    d.Zcore = r.to_double(t[8]);
    d.target_mass = r.to_int(t[10]);
    d.target_Z = r.to_int(t[12]);
  }

  const int norb = r.count_record("norbits");
  for (int i = 0; i < norb; ++i) {
    const auto t = r.record("orbit");
    r.expect_size(t, 8);
    GoldenOrbit o;
    o.index = r.to_int(t[1]);
    o.n = r.to_int(t[2]);
    o.l = r.to_int(t[3]);
    o.j2 = r.to_int(t[4]);
    o.tz2 = r.to_int(t[5]);
    o.occupation = r.to_double(t[6]);
    o.cvq = r.to_int(t[7]);
    d.orbits.push_back(o);
  }
  d.holes = r.counted_list(r.record("holes"), 1);
  d.particles = r.counted_list(r.record("particles"), 1);
  d.core = r.counted_list(r.record("core"), 1);
  d.valence = r.counted_list(r.record("valence"), 1);
  d.qspace = r.counted_list(r.record("qspace"), 1);

  const int nkets = r.count_record("nkets");
  for (int i = 0; i < nkets; ++i) {
    const auto t = r.record("ket");
    r.expect_size(t, 4);
    if (r.to_int(t[1]) != i) {
      r.fail("ket index " + t[1] + " out of order, expected " + std::to_string(i));
    }
    d.kets.emplace_back(r.to_int(t[2]), r.to_int(t[3]));
  }

  const int nch = r.count_record("ntwobody_channels");
  for (int i = 0; i < nch; ++i) {
    GoldenChannel c;
    auto t = r.record("channel");
    r.expect_size(t, 6);
    c.ch = r.to_int(t[1]);
    c.J = r.to_int(t[2]);
    c.parity = r.to_int(t[3]);
    c.Tz = r.to_int(t[4]);
    c.nkets = r.to_int(t[5]);
    if (c.ch != i) r.fail("channel index out of order, expected " + std::to_string(i));
    t = r.record("channel_kets");
    if (t.size() < 2 || r.to_int(t[1]) != c.ch) r.fail("channel_kets for the wrong channel");
    for (std::size_t k = 2; k < t.size(); ++k) c.kets.push_back(r.to_int(t[k]));
    if (static_cast<int>(c.kets.size()) != c.nkets) {
      r.fail("channel " + std::to_string(c.ch) + " declares " + std::to_string(c.nkets) +
             " kets but channel_kets lists " + std::to_string(c.kets.size()));
    }
    for (const char* name : kSubsetNames) {
      t = r.record("channel_subset");
      if (t.size() < 4 || r.to_int(t[1]) != c.ch) r.fail("channel_subset for the wrong channel");
      if (t[2] != name) {
        r.fail(std::string("expected subset '") + name + "', found '" + t[2] + "'");
      }
      c.subsets[name] = r.counted_list(t, 3);
    }
    d.channels.push_back(std::move(c));
  }

  const int nph = r.count_record("nph_channels");
  for (int i = 0; i < nph; ++i) {
    GoldenPhChannel c;
    auto t = r.record("ph_channel");
    r.expect_size(t, 6);
    c.ch = r.to_int(t[1]);
    c.J = r.to_int(t[2]);
    c.parity = r.to_int(t[3]);
    c.Tz = r.to_int(t[4]);
    c.nkets = r.to_int(t[5]);
    if (c.ch != i) r.fail("ph_channel index out of order, expected " + std::to_string(i));
    t = r.record("ph_channel_kets");
    if (t.size() < 2 || r.to_int(t[1]) != c.ch) r.fail("ph_channel_kets for the wrong channel");
    for (std::size_t k = 2; k < t.size(); ++k) c.kets.push_back(r.to_int(t[k]));
    if (static_cast<int>(c.kets.size()) != c.nkets) {
      r.fail("ph_channel " + std::to_string(c.ch) + " declares " + std::to_string(c.nkets) +
             " kets but ph_channel_kets lists " + std::to_string(c.kets.size()));
    }
    d.ph_channels.push_back(std::move(c));
  }

  const int n3 = r.count_record("nthreebody_channels");
  for (int i = 0; i < n3; ++i) {
    const auto t = r.record("threebody_channel");
    r.expect_size(t, 6);
    GoldenChannel3 c;
    c.ch = r.to_int(t[1]);
    c.twoJ = r.to_int(t[2]);
    c.parity = r.to_int(t[3]);
    c.twoTz = r.to_int(t[4]);
    c.nkets = r.to_int(t[5]);
    if (c.ch != i) r.fail("threebody_channel index out of order, expected " + std::to_string(i));
    d.channels3.push_back(c);
  }
  {
    const auto t = r.record("nkets3");
    r.expect_size(t, 4);
    r.expect_word(t, 2, "sum_channel_kets3");
    d.nkets3 = r.to_ll(t[1]);
    d.sum_channel_kets3 = r.to_ll(t[3]);
  }
  r.expect_size(r.record("end"), 1);
  if (!r.at_end()) r.fail_trailing();
  return d;
}

std::filesystem::path golden_basis_dir() {
  return std::filesystem::path(IMSRG_GOLDEN_DIR) / "basis";
}

std::vector<std::string> golden_basis_files() {
  std::vector<std::string> names;
  for (const char* e : {"emax2", "emax4"}) {
    for (const char* s : {"He4", "Ne20", "Ne20_sd-shell", "O16", "p-shell", "sd-shell"}) {
      names.push_back(std::string(e) + "_" + s + ".txt");
    }
  }
  std::sort(names.begin(), names.end());
  return names;
}

}  // namespace imsrg::basis::testing
