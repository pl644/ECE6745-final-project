//========================================================================
// ubmark-bsearch
//========================================================================

#include "ubmark-bsearch.h"

void ubmark_bsearch( int* idxs, int* search_keys, int search_keys_size,
                     int* sorted_keys, int sorted_keys_size )
{
  for ( int i = 0; i < search_keys_size; i++ ) {

    int key     = search_keys[i];
    int idx_min = 0;
    int idx_mid = sorted_keys_size/2;
    int idx_max = sorted_keys_size-1;

    int done = 0;
    idxs[i] = -1;
    do {
      int midkey = sorted_keys[idx_mid];

      if ( key == midkey ) {
        idxs[i] = idx_mid;
        done = 1;
      }

      if ( key > midkey )
        idx_min = idx_mid + 1;
      else if ( key < midkey )
        idx_max = idx_mid - 1;

      idx_mid = ( idx_min + idx_max ) / 2;

    } while ( !done && (idx_min <= idx_max) );

  }
}

