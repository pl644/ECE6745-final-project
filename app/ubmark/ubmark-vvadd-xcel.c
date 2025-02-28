//========================================================================
// ubmark-vvadd-xcel
//========================================================================

#include "ubmark-vvadd.h"
#include "ubmark-vvadd-xcel.h"

//------------------------------------------------------------------------
// vvadd_xcel
//------------------------------------------------------------------------

#ifdef _RISCV

void ubmark_vvadd_xcel( int* dest, int* src0, int* src1, int size )
{
  __asm__ (
    "csrw 0x7e1, %[src0];\n"
    "csrw 0x7e2, %[src1];\n"
    "csrw 0x7e3, %[dest];\n"
    "csrw 0x7e4, %[size];\n"
    "csrw 0x7e0, x0     ;\n"
    "csrr x0,    0x7e0  ;\n"

    // Outputs from the inline assembly block

    :

    // Inputs to the inline assembly block

    : [src0] "r"(src0),
      [src1] "r"(src1),
      [dest] "r"(dest),
      [size] "r"(size)

    // Tell the compiler this accelerator read/writes memory

    : "memory"
  );
}

#else

void ubmark_vvadd_xcel( int* dest, int* src0, int* src1, int size )
{
  ubmark_vvadd( dest, src0, src1, size );
}

#endif

