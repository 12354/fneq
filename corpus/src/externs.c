/* Callees and globals shared by all corpus functions. fneq treats the
 * callees as opaque; their bodies only exist so that the binaries link. */
#include "cc.h"

int g_counter;
int g_table[16];

int CC ext_g2(int a, int b) { return a * 31 + b; }
int CC ext_g1(int a) { return a ^ 0x55; }
void CC ext_sink(int a, int b, int c) { g_counter = a + b + c; }

void fneq_entry(void) {}
