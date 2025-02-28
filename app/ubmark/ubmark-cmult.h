//========================================================================
// ubmark-cmult
//========================================================================
// This microbenchmark performs elementwise complex multiplication across
// two input arrays and stores the result in the destination array.

#ifndef UBMARK_CMULT_H
#define UBMARK_CMULT_H

typedef struct
{
  int re;
  int im;
}
complex_t;

void ubmark_cmult( complex_t* dest, complex_t* src0,
                   complex_t* src1, int size );

#endif /* UBMARK_CMULT_H */

