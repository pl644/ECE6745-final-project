//========================================================================
// ubmark-cmult
//========================================================================

#include "ubmark-cmult.h"

void ubmark_cmult( complex_t* dest, complex_t* src0,
                   complex_t* src1, int size )
{
  for ( int i = 0; i < size; i++ ) {
    dest[i].re = (src0[i].re * src1[i].re) - (src0[i].im * src1[i].im);
    dest[i].im = (src0[i].re * src1[i].im) + (src0[i].im * src1[i].re);
  }
}

