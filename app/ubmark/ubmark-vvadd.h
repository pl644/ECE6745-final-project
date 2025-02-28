//========================================================================
// ubmark-vvadd
//========================================================================
// This microbenchmark does an element-wise addition across two arrays
// and stores the result in a third array.

#ifndef UBMARK_VVADD_H
#define UBMARK_VVADD_H

void ubmark_vvadd( int* dest, int* src0, int* src1, int size );

#endif /* UBMARK_VVADD_H */

