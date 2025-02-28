//========================================================================
// ubmark-mfilt-eval
//========================================================================

#include "ece6745.h"
#include "ubmark-mfilt.h"
#include "ubmark-mfilt.dat"

int main( void )
{
  // Allocate destination array for results

  int* dest = ece6745_malloc( eval_size*eval_size * (int)sizeof(int) );

  // Run the evaluation

  ece6745_stats_on();
  ubmark_mfilt( dest, eval_mask, eval_src, eval_size );
  ece6745_stats_off();

  // Verify the results

  for ( int i = 0; i < eval_size*eval_size; i++ ) {
    if ( dest[i] != eval_ref[i] ) {
      ece6745_wprintf( L"\n FAILED: dest[%d] != eval_ref[%d] (%d != %d)\n\n",
                       i, i, dest[i], eval_ref[i] );
      ece6745_exit(1);
    }
  }

  // Free destination array

  ece6745_free(dest);

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

