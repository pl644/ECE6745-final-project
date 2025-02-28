//========================================================================
// ubmark-sort
//========================================================================

#include "ubmark-sort.h"

//''' LAB TASK '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
// Implement helper functions here
//''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''\/

//------------------------------------------------------------------------
// ubmark_sort_swap
//------------------------------------------------------------------------
// Helper function to swap two values

void ubmark_sort_swap( int* x_ptr, int* y_ptr )
{
  int temp = *x_ptr;
  *x_ptr = *y_ptr;
  *y_ptr = temp;
}

//------------------------------------------------------------------------
// ubmark_sort_partition
//------------------------------------------------------------------------
// Helper function to partition an array with last as the pivot

int ubmark_sort_partition( int* x, int first, int last )
{
  int pivot = x[last-1];
  int idx   = first;

  for ( int i = first; i < last; i++ ) {
    if ( x[i] <= pivot ) {
      ubmark_sort_swap( &x[i], &x[idx] );
      idx += 1;
    }
  }

  return idx - 1;
}

//------------------------------------------------------------------------
// ubmark_sort_h
//------------------------------------------------------------------------
// Quick sort recursive helper function

void ubmark_sort_h( int* x, int first, int last )
{
  // Base case

  int size = last - first;
  if ( size <= 1 )
    return;

  // Find partition element by median of 3, place pivot last

  int lo = first;
  int md = first + ( size / 2 );
  int hi = first + ( size - 1 );

  // lo is median

  if ( ( x[hi] >= x[lo] && x[lo] >= x[md] ) ||
       ( x[md] >= x[lo] && x[lo] >= x[hi] ) )
    ubmark_sort_swap( &x[lo], &x[hi] );

  // md is median

  else if ( ( x[lo] >= x[md] && x[md] >= x[hi] ) ||
            ( x[hi] >= x[md] && x[md] >= x[lo] ) )
    ubmark_sort_swap( &x[md], &x[hi] );

  // Partition array

  int p = ubmark_sort_partition( x, first, last );

  // Recursive case

  ubmark_sort_h( x, first, p    );
  ubmark_sort_h( x, p,     last );
}

//''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''/\

//------------------------------------------------------------------------
// ubmark_sort
//------------------------------------------------------------------------

void ubmark_sort( int* x, int size )
{
  //''' LAB TASK '''''''''''''''''''''''''''''''''''''''''''''''''''''''''
  // Implement single-threaded sorting algorithm here
  //''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''\/
  //: // These macros allow the function to compile. Once you have
  //: // implemented your algorithm you should remove these macros.
  //:
  //: ECE6745_UNUSED_PTR(x);
  //: ECE6745_UNUSED_INT(size);

  ubmark_sort_h( x, 0, size );

  //''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''/\

}

