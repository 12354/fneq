/* Floating point: SSE (and x87 on x86 unless -mfpmath=sse). */
#include "cc.h"

double CC fmul2(double x, double y) { return x * 2.0 + y; }
double CC fadd2(double x, double y) { return x + x + y; }

/* x87 precision control: same product, but one variant computes it with the
 * precision-control field set to 24 bits. */
double CC x87_mul(double x, double y) {
  long double r = (long double)x * (long double)y;
  return (double)r;
}
double CC x87_mul24(double x, double y) {
  unsigned short cw, cw24;
  __asm__ volatile("fnstcw %0" : "=m"(cw));
  cw24 = cw & ~0x300;
  __asm__ volatile("fldcw %0" : : "m"(cw24));
  long double r = (long double)x * (long double)y;
  volatile double out = (double)r;
  __asm__ volatile("fldcw %0" : : "m"(cw));
  return out;
}
