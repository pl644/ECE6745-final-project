//========================================================================
// ubmark-accum
//========================================================================

#include "ubmark-accum.h"

int ubmark_accum( int* src, int size )
{
  // ''' SECTION TASK ''''''''''''''''''''''''''''''''''''''''''''''''''''
  // Implement a simple C function to add all of the elements in the
  // source array and then return this result.
  // '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''\/

  int sum = 0;
  for ( int i = 0; i < size; i++ )
    sum += src[i];
  return sum;

  // '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''/\

}

