/* Memory effects through pointer arguments and globals. */
#include "cc.h"

struct pair { int a, b, c; };

void CC set2(struct pair *p, int v) {
  p->a = v;
  p->b = v + 1;
}
void CC set2_rev(struct pair *p, int v) {
  p->b = v + 1;
  p->a = v;
}
/* dropped write */
void CC set1(struct pair *p, int v) { p->a = v; }

void CC bump(int x, int y) {
  g_counter += x;
  g_table[y & 15] = x;
}

int CC sum4_rolled(const int *p, int n) {
  int s = 0;
  for (int i = 0; i < 4; i++) s += p[i];
  return s + n;
}
int CC sum4_unrolled(const int *p, int n) { return p[0] + p[1] + p[2] + p[3] + n; }

/* variable trip count: only provable up to an unroll bound */
int CC sum_n(const int *p, int n) {
  int s = 0;
  for (int i = 0; i < n; i++) s += p[i];
  return s;
}

/* reads through a pointer that may alias the write */
int CC alias_rw(int *p, int *q) {
  *p = 1;
  return *q;
}
