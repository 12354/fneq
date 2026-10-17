/* Calling-convention selection for corpus functions. The build passes
 * -DFNEQ_CC_<name>; every corpus function and callee is declared with CC. */
#ifndef FNEQ_CC_H
#define FNEQ_CC_H

#if defined(FNEQ_CC_cdecl)
#define CC __attribute__((cdecl))
#elif defined(FNEQ_CC_stdcall)
#define CC __attribute__((stdcall))
#elif defined(FNEQ_CC_fastcall)
#define CC __attribute__((fastcall))
#elif defined(FNEQ_CC_thiscall)
#define CC __attribute__((thiscall))
#elif defined(FNEQ_CC_sysv)
#define CC __attribute__((sysv_abi))
#elif defined(FNEQ_CC_ms)
#define CC __attribute__((ms_abi))
#else
#error "define FNEQ_CC_<convention>"
#endif

/* Opaque callees and globals, defined in externs.c. */
extern int CC ext_g2(int a, int b);
extern int CC ext_g1(int a);
extern void CC ext_sink(int a, int b, int c);
extern int g_counter;
extern int g_table[16];

#endif
