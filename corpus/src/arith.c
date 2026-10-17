/* Integer arithmetic and comparisons. */
#include "cc.h"

int CC add_ab(int a, int b) { return a + b; }
int CC add_ba(int a, int b) { return b + a; }

int CC mul2(int x, int y) { return x * 2 + y; }
int CC shl1(int x, int y) { return (int)((unsigned)x << 1) + y; }

/* branch vs. branch-free minimum */
int CC min_branch(int x, int y) {
  if (x < y) return x;
  return y;
}
int CC min_bits(int x, int y) { return y ^ ((x ^ y) & -(x < y)); }

/* off-by-one mutation */
int CC lt10(int x, int y) { return x < 10 ? y : -y; }
int CC le10(int x, int y) { return x <= 10 ? y : -y; }

/* signed vs. unsigned comparison */
int CC cmp_signed(int x, int y) { return x < y ? 7 : 3; }
int CC cmp_unsigned(int x, int y) { return (unsigned)x < (unsigned)y ? 7 : 3; }

/* differs for exactly one value of x */
int CC rare_base(int x, int y) { return y * 3 + 1; }
int CC rare_mut(int x, int y) { return x == 0x5eed1234 ? y * 3 + 2 : y * 3 + 1; }

/* used for compiler/optimisation-level cross builds */
int CC clamp(int x, int hi) {
  if (hi < 0) hi = 0;
  if (x < 0) return 0;
  if (x > hi) return hi;
  return x;
}
int CC absdiff(int a, int b) { return a > b ? a - b : b - a; }

unsigned CC udivmod(unsigned a, unsigned b) {
  if (b == 0) return 0;
  return a / b + a % b;
}

/* 64-bit results: edx:eax on x86 */
long long CC mul64_ab(int a, int b) { return (long long)a * b; }
long long CC mul64_ba(int a, int b) { return (long long)b * a; }
long long CC mul64_trunc(int a, int b) { return (long long)(int)((unsigned)a * (unsigned)b); }

/* variable trip count without memory accesses */
int CC tri(int n, int k) {
  int s = 0;
  for (int i = 0; i < n; i++) s += i * k + 1;
  return s;
}
