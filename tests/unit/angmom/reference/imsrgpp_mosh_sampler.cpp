// Prints imsrg++ AngMom::Moshinsky and AngMom::TalmiB values for the angmom tests of the rewrite;
// its output is tests/unit/angmom/data/moshinsky_imsrgpp_samples.txt. NOT part of the build: it
// links against the imsrg-devel reference library (M1 build, outside the repo):
//   SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk clang++ -std=c++17 -O1
//     -I<new-imsrg-reference>/imsrg-devel-src/src imsrgpp_mosh_sampler.cpp
//     -L<new-imsrg-reference>/build -lIMSRG -Wl,-rpath,<new-imsrg-reference>/build -o sampler
//   ./sampler > moshinsky_imsrgpp_samples.txt
// Deterministic: all brackets with E <= 4, then an LCG sample of allowed brackets with E <= 16;
// all Talmi B with n <= 3, l <= 6, then an LCG sample with n <= 6, l <= 12.
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "AngMom.hh"

namespace {
std::uint64_t state = 0x2545F4914F6CDD1DULL;
int next_int(int n) {  // uniform-ish in [0, n)
  state = state * 6364136223846793005ULL + 1442695040888963407ULL;
  return static_cast<int>((state >> 33) % static_cast<std::uint64_t>(n));
}
bool tri(int a, int b, int c) { return std::abs(a - b) <= c && c <= a + b; }
void print_mosh(int N, int L, int n, int l, int n1, int l1, int n2, int l2, int lam) {
  std::printf("mosh %d %d %d %d %d %d %d %d %d %.17g\n", N, L, n, l, n1, l1, n2, l2, lam,
              AngMom::Moshinsky(N, L, n, l, n1, l1, n2, l2, lam));
}
}  // namespace

int main() {
  AngMom::FillFactorialLists(170);
  std::printf(
      "# imsrg++ AngMom::Moshinsky(N,L,n,l,n1,l1,n2,l2,lam) and AngMom::TalmiB(na,la,nb,lb,p)\n");
  std::printf(
      "# printed with %%.17g by mosh_sampler.cpp linked against libIMSRG (M1 reference build)\n");
  // Exhaustive E <= 4.
  for (int E = 0; E <= 4; ++E)
    for (int N = 0; 2 * N <= E; ++N)
      for (int L = 0; 2 * N + L <= E; ++L)
        for (int n = 0; 2 * N + L + 2 * n <= E; ++n) {
          const int l = E - 2 * N - L - 2 * n;
          for (int n1 = 0; 2 * n1 <= E; ++n1)
            for (int l1 = 0; 2 * n1 + l1 <= E; ++l1)
              for (int n2 = 0; 2 * n1 + l1 + 2 * n2 <= E; ++n2) {
                const int l2 = E - 2 * n1 - l1 - 2 * n2;
                if ((L + l - l1 - l2) % 2 != 0) continue;
                for (int lam = 0; lam <= E; ++lam)
                  if (tri(L, l, lam) && tri(l1, l2, lam))
                    print_mosh(N, L, n, l, n1, l1, n2, l2, lam);
              }
        }
  // Random allowed brackets with 5 <= E <= 16.
  int count = 0;
  while (count < 3000) {
    const int E = 5 + next_int(12);
    const int e_cm = next_int(E + 1);
    const int N = next_int(e_cm / 2 + 1), L = e_cm - 2 * N;
    const int n = next_int((E - e_cm) / 2 + 1), l = E - e_cm - 2 * n;
    const int e1 = next_int(E + 1);
    const int n1 = next_int(e1 / 2 + 1), l1 = e1 - 2 * n1;
    const int n2 = next_int((E - e1) / 2 + 1), l2 = E - e1 - 2 * n2;
    if ((L + l - l1 - l2) % 2 != 0) continue;
    const int lo = std::abs(L - l) > std::abs(l1 - l2) ? std::abs(L - l) : std::abs(l1 - l2);
    const int hi = L + l < l1 + l2 ? L + l : l1 + l2;
    if (lo > hi) continue;
    print_mosh(N, L, n, l, n1, l1, n2, l2, lo + next_int(hi - lo + 1));
    ++count;
  }
  // Talmi B.
  for (int na = 0; na <= 3; ++na)
    for (int nb = 0; nb <= 3; ++nb)
      for (int la = 0; la <= 6; ++la)
        for (int lb = la % 2; lb <= 6; lb += 2)
          for (int p = (la + lb) / 2; p <= (la + lb) / 2 + na + nb; ++p)
            std::printf("talmib %d %d %d %d %d %.17g\n", na, la, nb, lb, p,
                        AngMom::TalmiB(na, la, nb, lb, p));
  for (int i = 0; i < 1000; ++i) {
    const int na = next_int(7), nb = next_int(7), la = next_int(13);
    const int lb = la % 2 + 2 * next_int(7 - la % 2);
    const int p = (la + lb) / 2 + next_int(na + nb + 1);
    std::printf("talmib %d %d %d %d %d %.17g\n", na, la, nb, lb, p,
                AngMom::TalmiB(na, la, nb, lb, p));
  }
  return 0;
}
