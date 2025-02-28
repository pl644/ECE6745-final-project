//========================================================================
// ubmark-mfilt
//========================================================================

#include "ubmark-mfilt.h"

// Helper functions for setting and getting pixels in image

inline
void set( int* image, int size, int row, int col, int v )
{
  image[ row*size + col ] = v;
}

inline
int get( int* image, int size, int row, int col )
{
  return image[ row*size + col ];
}

void ubmark_mfilt( int* dest, int* mask, int* src, int size )
{
  int coeff0 = 8;
  int coeff1 = 6;

  // norm is calculated as coeff0 + 4*coeff1. Because it is a power of 2,
  // we can represent it as a shift

  int norm_shamt = 5;

  for ( int i = 0; i < size; i++ ) {
    for ( int j = 0; j < size; j++ ) {

      // Set pixels at edge to be zero

      if ( (i == 0) || (i == size-1) || (j == 0) || (j == size-1) )
        set( dest, size, i, j, 0 );

      else {

        // If mask is set, then do the filter ...

        if ( get( mask, size, i, j ) != 0 ) {
          int out0 = get( src, size, i-1, j   ) * coeff1;
          int out1 = get( src, size, i,   j-1 ) * coeff1;
          int out2 = get( src, size, i,   j   ) * coeff0;
          int out3 = get( src, size, i,   j+1 ) * coeff1;
          int out4 = get( src, size, i+1, j   ) * coeff1;
          int out  = out0 + out1 + out2 + out3 + out4;
          set( dest, size, i, j, out >> norm_shamt );
        }

        // ... otherwise copy src to dest

        else {
          int out = get( src, size, i, j );
          set( dest, size, i, j, out );
        }

      }

    }
  }
}

