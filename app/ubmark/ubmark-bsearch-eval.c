//========================================================================
// ubmark-bsearch-eval
//========================================================================

#include "ece6745.h"
#include "ubmark-bsearch.h"
#include "ubmark-bsearch.dat"

int main( void )
{
  // Allocate destination array for results

  int* idxs = ece6745_malloc( eval_search_keys_size * (int)sizeof(int) );

  // Run the evaluation

  ece6745_stats_on();

  ubmark_bsearch( idxs, eval_search_keys, eval_search_keys_size,
                  eval_sorted_keys, eval_sorted_keys_size );

  ece6745_stats_off();

  // Verify the results

  for ( int i = 0; i < eval_search_keys_size; i++ ) {
    if ( idxs[i] != eval_ref[i] ) {
      ece6745_wprintf( L"\n FAILED: idxs[%d] != eval_ref[%d] (%d != %d)\n\n",
                       i, i, idxs[i], eval_ref[i] );
      ece6745_exit(1);
    }
  }

  // Free destination array

  ece6745_free(idxs);

  // Check for no memory leaks

  if ( ece6745_get_heap_usage() != 0 ) {
    ece6745_wprintf( L"\n FAILED: memory leak of %d bytes!\n\n",
                     ece6745_get_heap_usage() );
    ece6745_exit(1);
  }

  // Otherwise we passed

  ece6745_wprintf( L"\n **PASSED** \n\n" );

  return 0;
}

