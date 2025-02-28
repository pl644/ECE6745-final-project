//========================================================================
// ubmark-accum-xcel
//========================================================================

#include "ubmark-accum.h"
#include "ubmark-accum-xcel.h"

//------------------------------------------------------------------------
// accum_xcel
//------------------------------------------------------------------------

#ifdef _RISCV

int ubmark_accum_xcel( int* src, int size )
{
  int result = 0;

  __asm__ (
    "csrw 0x7e1, %[src]; \n"
    "csrw 0x7e2, %[size];\n"
    "csrw 0x7e0, x0     ;\n"
    "csrr %[result], 0x7e0;\n"

    // Outputs from the inline assembly block

    : [result] "=r"(result)

    // Inputs to the inline assembly block

    : [src]    "r"(src),
      [size]   "r"(size)

    // Tell the compiler this accelerator read/writes memory

    : "memory"
  );

  return result;
}

#else

int ubmark_accum_xcel( int* src, int size )
{
  return ubmark_accum( src, size );
}

#endif

