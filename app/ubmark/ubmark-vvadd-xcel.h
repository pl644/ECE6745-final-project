//========================================================================
// ubmark-vvadd-xcel
//========================================================================
// This microbenchmark does an element-wise addition across two arrays
// and stores the result in a third array.

#ifndef UBMARK_VVADD_XCEL_H
#define UBMARK_VVADD_XCEL_H

void ubmark_vvadd_xcel( int* dest, int* src0, int* src1, int size );

#endif /* UBMARK_VVADD_XCEL_H */

