//========================================================================
// ubmark-sort
//========================================================================
// This microbenchmark sorts an array of integers.

#ifndef UBMARK_SORT_H
#define UBMARK_SORT_H

//''' LAB TASK '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
// Declare helper functions here so you can unit test them
//''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''\/

// Helper functions

void ubmark_sort_swap( int* x_ptr, int* y_ptr );
int  ubmark_sort_partition( int* x, int first, int last );
int  ubmark_sort_median( int x, int y, int z );

//''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''/\

void ubmark_sort( int* x, int size );

#endif /* UBMARK_SORT_H */

