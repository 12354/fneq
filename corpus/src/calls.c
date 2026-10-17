/* External calls. */
#include "cc.h"

int CC call_ab(int a, int b) { return ext_g2(a, b) + 1; }
/* swapped argument order */
int CC call_ba(int a, int b) { return ext_g2(b, a) + 1; }

int CC call_chain(int a, int b) {
  int t = ext_g1(a);
  ext_sink(t, b, 3);
  return t + b;
}
