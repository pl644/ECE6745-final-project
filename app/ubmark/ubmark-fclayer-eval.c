//========================================================================
// ubmark-sort-eval
//========================================================================

#include "ece6745.h"
#include "ubmark-fclayer.h"
#include "ubmark-fclayer.dat"
// #include <stdlib.h> 


int abs(int x) {
  return x < 0 ? -x : x;
}

int main( void )
{
  // Run the evaluation
  int* output = ece6745_malloc( eval_size * (int)sizeof(int) );
  ece6745_stats_on();
  ubmark_fclayer_fixed( eval_input,eval_weight, eval_bias,output,8,32,32 );
  ece6745_stats_off();

  // Verify the results

  for ( int i = 0; i < eval_size; i++ ) {
    if ( abs(output[i] - eval_ref[i]) > 20 ) {
      ece6745_wprintf( L"\n FAILED: eval_src[%d]: %d,  eval_ref[%d]: %d\n\n",
                       i,output[i], i, eval_ref[i] );
      ece6745_exit(1);
    }
  }

  ece6745_free(output);
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

