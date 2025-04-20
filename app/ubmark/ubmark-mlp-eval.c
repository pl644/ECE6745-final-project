//========================================================================
// ubmark-sort-eval
//========================================================================

#include "ece6745.h"
#include "ubmark-mlp.h"
#include "ubmark-mlp.dat"
// #include <stdlib.h> 


int abs(int x) {
  return x < 0 ? -x : x;
}

int batch_size = 1;                      // Batch size
int input_size = 4;
int hidden1_size = 2;
int hidden2_size =2;
int output_size = 2;
int main( void )
{
  // Run the evaluation
  int* output = ece6745_malloc( eval_size * (int)sizeof(int) );
  ece6745_stats_on();
  mnist_inference( eval_input,eval_weight_0, eval_bias_0,eval_weight_1,eval_bias_1,eval_weight_2,eval_bias_2, output,batch_size,input_size,hidden1_size, hidden2_size,output_size);
  ece6745_stats_off();

  // Verify the results

  for ( int i = 0; i < eval_size; i++ ) {
    if ( abs(output[i] - eval_ref[i]) > 20 ) {
      ece6745_wprintf( L"\n FAILED: output[%d]: %d,  eval_ref[%d]: %d\n\n",
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

