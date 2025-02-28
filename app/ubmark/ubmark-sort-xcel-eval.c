//========================================================================
// ubmark-sort-xcel-eval
//========================================================================

#include "ece6745.h"
#include "ubmark-sort-xcel.h"
#include "ubmark-sort.dat"

int main( void )
{
  // Run the evaluation

  ece6745_stats_on();
  ubmark_sort_xcel( eval_src, eval_size );
  ece6745_stats_off();

  // Verify the results

  for ( int i = 0; i < eval_size; i++ ) {
    if ( eval_src[i] != eval_ref[i] ) {
      ece6745_wprintf( L"\n FAILED: eval_src[%d] != eval_ref[%d] (%d != %d)\n\n",
                       i, i, eval_src[i], eval_ref[i] );
      ece6745_exit(1);
    }
  }

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

