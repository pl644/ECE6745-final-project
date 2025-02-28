//========================================================================
// ubmark-accum-eval
//========================================================================

#include "ece6745.h"
#include "ubmark-accum.h"
#include "ubmark-accum.dat"

int main( void )
{
  // Run the evaluation

  ece6745_stats_on();
  int result = ubmark_accum( eval_src, eval_size );
  ece6745_stats_off();

  // Verify the results

  for ( int i = 0; i < eval_size; i++ ) {
    if ( result != eval_ref ) {
      ece6745_wprintf( L"\n FAILED: result != eval_ref (%d != %d)\n\n",
                       result, eval_ref );
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

