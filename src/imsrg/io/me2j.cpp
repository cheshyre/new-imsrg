#include "imsrg/io/me2j.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "imsrg/base/error.hpp"
#include "imsrg/base/types.hpp"
#include "imsrg/io/input_stream.hpp"

namespace imsrg::io {

namespace {

constexpr double kSqrt2 = 1.4142135623730950488016887242097;

/// A single-particle orbit label of the file (the isospin label is implicit).
struct FileOrbit {
  std::int32_t n;
  std::int32_t l;
  std::int32_t j2;
  std::int32_t e() const noexcept { return 2 * n + l; }
};

/// The truncation with the imsrg++ defaults resolved (a non-positive value selects the default).
struct Resolved {
  std::int32_t e1max;
  std::int32_t e2max;
  std::int32_t lmax;
};

Resolved resolve(const Me2jTruncation& t) {
  if (t.e1max < 0) throw InputError("me2j: file2e1max must be >= 0");
  return {t.e1max, t.e2max > 0 ? t.e2max : 2 * t.e1max, t.lmax > 0 ? t.lmax : t.e1max};
}

/// The file's orbit list for shells e <= emax: e ascending, then l ascending, then 2j ascending.
std::vector<FileOrbit> file_orbits(std::int32_t emax, std::int32_t lmax) {
  std::vector<FileOrbit> out;
  for (std::int32_t e = 0; e <= emax; ++e) {
    for (std::int32_t l = e % 2; l <= std::min(e, lmax); l += 2) {
      const std::int32_t n = (e - l) / 2;
      for (std::int32_t j2 = std::abs(2 * l - 1); j2 <= 2 * l + 1; j2 += 2) {
        out.push_back({n, l, j2});
      }
    }
  }
  return out;
}

/// Calls `f(nlj1, nlj2, nlj3, nlj4, J)` for every (abcd, J) record of the file prefix with shells
/// up to `emax_eff`, in file order (the loop structure of imsrg++ ReadBareTBME_Darmstadt).
template <class F>
void for_each_record(const std::vector<FileOrbit>& orbs, std::int32_t e2max, F&& f) {
  const std::size_t count = orbs.size();
  for (std::size_t i1 = 0; i1 < count; ++i1) {
    const FileOrbit& o1 = orbs[i1];
    for (std::size_t i2 = 0; i2 <= i1; ++i2) {
      const FileOrbit& o2 = orbs[i2];
      if (o1.e() + o2.e() > e2max) break;
      for (std::size_t i3 = 0; i3 <= i1; ++i3) {
        const FileOrbit& o3 = orbs[i3];
        const std::size_t i4max = (i3 == i1) ? i2 : i3;
        for (std::size_t i4 = 0; i4 <= i4max; ++i4) {
          const FileOrbit& o4 = orbs[i4];
          if (o3.e() + o4.e() > e2max) break;
          if ((o1.l + o2.l + o3.l + o4.l) % 2 != 0) continue;
          const std::int32_t jmin = std::max(std::abs(o1.j2 - o2.j2), std::abs(o3.j2 - o4.j2)) / 2;
          const std::int32_t jmax = std::min(o1.j2 + o2.j2, o3.j2 + o4.j2) / 2;
          for (std::int32_t J = jmin; J <= jmax; ++J) f(i1, i2, i3, i4, J);
        }
      }
    }
  }
}

std::int32_t effective_emax(const Resolved& r, std::int32_t emax_read) {
  return (emax_read < 0) ? r.e1max : std::min(emax_read, r.e1max);
}

std::int64_t count_values(const Resolved& r, std::int32_t emax_read) {
  const auto orbs = file_orbits(effective_emax(r, emax_read), r.lmax);
  std::int64_t records = 0;
  for_each_record(
      orbs, r.e2max,
      [&](std::size_t, std::size_t, std::size_t, std::size_t, std::int32_t) { ++records; });
  return 4 * records;
}

/// Proton and neutron model-space orbits of a file orbit (both absent if not in the space).
struct PnOrbit {
  OrbitIndex p;
  OrbitIndex n;
  bool present = false;
};

PnOrbit locate(const basis::SingleParticleBasis& b, const FileOrbit& o) {
  const basis::Orbit proton{o.n, o.l, TwoJ{o.j2}, TwoTz{-1}};
  const basis::Orbit neutron{o.n, o.l, TwoJ{o.j2}, TwoTz{1}};
  const auto p = b.find(proton);
  const auto n = b.find(neutron);
  if (!p || !n) return {};
  return {*p, *n, true};
}

}  // namespace

std::int64_t me2j_value_count(const Me2jTruncation& file, std::int32_t emax_read) {
  return count_values(resolve(file), emax_read);
}

op::Operator read_me2j(const std::filesystem::path& path, const Me2jReadOptions& options,
                       std::shared_ptr<const basis::ModelSpace> ms, Me2jReadReport* report) {
  const auto t0 = std::chrono::steady_clock::now();
  if (!ms) throw InputError("read_me2j: null model space");
  const Resolved r = resolve(options.file);
  const basis::SingleParticleBasis& sp = ms->basis();

  // Prefix of the file the model space needs (the whole file when verifying it).
  const std::int32_t emax_read = options.read_whole_file ? r.e1max : sp.emax();
  const auto orbs = file_orbits(effective_emax(r, emax_read), r.lmax);
  std::vector<PnOrbit> pn;
  pn.reserve(orbs.size());
  for (const FileOrbit& o : orbs) pn.push_back(locate(sp, o));

  op::Operator H(ms, op::OperatorSpec::scalar(2, Hermiticity::Hermitian));
  op::TwoBody& tb = H.two_body();

  TextInput in(path);
  in.skip_line();  // header line

  const std::int64_t expected = count_values(r, emax_read);
  std::int64_t read = 0;

  const auto next = [&]() {
    float v = 0.0F;
    if (!in.next_float(v)) {
      throw InputError("'" + path.string() + "': data ended after " + std::to_string(read) +
                       " values; " + std::to_string(expected) +
                       " are needed for the given file truncation (check file2e1max, file2e2max "
                       "and file2lmax)");
    }
    ++read;
    return static_cast<double>(v);
  };

  for_each_record(
      orbs, r.e2max,
      [&](std::size_t i1, std::size_t i2, std::size_t i3, std::size_t i4, std::int32_t J) {
        // File order: T=0 pn, nn, T=1 pn, pp (widened from float).
        const double v00 = next();
        const double vnn = next();
        const double v10 = next();
        const double vpp = next();
        const PnOrbit& a = pn[i1];
        const PnOrbit& b = pn[i2];
        const PnOrbit& c = pn[i3];
        const PnOrbit& d = pn[i4];
        if (!(a.present && b.present && c.present && d.present)) return;

        // The file is unnormalized: normalize identical pairs (imsrg++ order of ops).
        double norm = 1.0;
        const bool ab = (i1 == i2);
        const bool cd = (i3 == i4);
        if (ab) norm /= kSqrt2;
        if (cd) norm /= kSqrt2;
        const TwoJ tj{2 * J};

        if (norm > 0.9 || J % 2 == 0) {
          tb.set_normalized(a.p, b.p, c.p, d.p, tj, vpp * norm);
          tb.set_normalized(a.n, b.n, c.n, d.n, tj, vnn * norm);
        }
        // Set_pn_TBME_from_iso for T = 1 (even J for identical pairs) and T = 0.
        const auto add_iso = [&](double value, int phase) {
          double t = value * norm;
          t *= 0.5;
          if (ab) t *= kSqrt2;
          if (cd) t *= kSqrt2;
          tb.add_normalized(a.p, b.n, c.p, d.n, tj, t);
          if (i3 != i4) tb.add_normalized(a.p, b.n, c.n, d.p, tj, t * phase);
          if (i1 != i2 && i3 != i4) tb.add_normalized(a.n, b.p, c.n, d.p, tj, t);
          if (i1 != i2 && (i1 != i3 || i2 != i4)) {
            tb.add_normalized(a.n, b.p, c.p, d.n, tj, t * phase);
          }
        };
        if (norm > 0.9 || J % 2 == 0) add_iso(v10, +1);
        if (norm > 0.9 || J % 2 != 0) add_iso(v00, -1);
      });

  if (options.read_whole_file) {
    // Everything beyond the expected values means the truncation does not describe the file.
    float extra = 0.0F;
    std::int64_t found = read;
    while (in.next_float(extra)) ++found;
    if (found != expected) {
      throw InputError("'" + path.string() + "': found " + std::to_string(found) +
                       " values but expected " + std::to_string(expected) +
                       " for the given file truncation; check file2e1max, file2e2max and "
                       "file2lmax");
    }
  }

  if (report != nullptr) {
    report->values_read = read;
    report->seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }
  H.sync_to_device();  // the elements were set through the host mirror (no-op on host backends)
  return H;
}

}  // namespace imsrg::io
