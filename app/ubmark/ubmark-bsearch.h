//========================================================================
// ubmark-bsearch
//========================================================================
// This microbenchmark performs a binary search in an array of sorted
// keys and returns the corresponding index.

#ifndef UBMARK_BSEARCH_H
#define UBMARK_BSEARCH_H

void ubmark_bsearch( int* idxs, int* search_keys, int search_keys_size,
                     int* sorted_keys, int sorted_keys_size );

#endif /* UBMARK_BSEARCH_H */

