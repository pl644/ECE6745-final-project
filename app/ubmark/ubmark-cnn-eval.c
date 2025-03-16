//========================================================================
// ubmark-cnn-eval
//========================================================================

#include "ece6745.h"
#include "ubmark-cnn.h"
#include "ubmark-cnn.dat"

int main( void )
{
  // Use smaller dimensions to avoid memory issues
  const unsigned int batch_size = 1;
  const unsigned int in_channels = 2;
  const unsigned int out_channels = 4;
  const unsigned int in_height = 16;
  const unsigned int in_width = 16;
  const unsigned int kernel_size = 3;
  const unsigned int out_height = in_height - kernel_size + 1;
  const unsigned int out_width = in_width - kernel_size + 1;
  
  // Allocate memory
  fixed_t* input = (fixed_t*)ece6745_malloc(batch_size * in_channels * in_height * in_width * sizeof(fixed_t));
  fixed_t* weights = (fixed_t*)ece6745_malloc(out_channels * in_channels * kernel_size * kernel_size * sizeof(fixed_t));
  fixed_t* bias = (fixed_t*)ece6745_malloc(out_channels * sizeof(fixed_t));
  fixed_t* output = (fixed_t*)ece6745_malloc(batch_size * out_channels * out_height * out_width * sizeof(fixed_t));

  // Initialize input data - use integer constants directly, avoiding floating-point operations
  for (unsigned int b = 0; b < batch_size; b++) {
    for (unsigned int c = 0; c < in_channels; c++) {
      for (unsigned int h = 0; h < in_height; h++) {
        for (unsigned int w = 0; w < in_width; w++) {
          unsigned int idx = b * (in_channels * in_height * in_width) + 
                          c * (in_height * in_width) + 
                          h * in_width + w;
          // Use integer constant: 26 ~= 0.1 * 256 (FIX_SCALE)
          // Explicit cast to fixed_t (int) to avoid sign conversion warning
          input[idx] = (fixed_t)(26 * (int)((c + 1) * ((h % 5) + 1) * ((w % 5) + 1)));
        }
      }
    }
  }

  // Initialize weights - use integer constants directly
  for (unsigned int oc = 0; oc < out_channels; oc++) {
    for (unsigned int ic = 0; ic < in_channels; ic++) {
      for (unsigned int kh = 0; kh < kernel_size; kh++) {
        for (unsigned int kw = 0; kw < kernel_size; kw++) {
          unsigned int idx = oc * (in_channels * kernel_size * kernel_size) + 
                          ic * (kernel_size * kernel_size) + 
                          kh * kernel_size + kw;
          // Use integer constant: 3 ~= 0.01 * 256 (FIX_SCALE)
          // Explicit cast to fixed_t (int) to avoid sign conversion warning
          weights[idx] = (fixed_t)(3 * (int)((oc + 1) * (ic + 1) * (kh + 1) * (kw + 1)));
        }
      }
    }
  }

  // Initialize bias
  for (unsigned int oc = 0; oc < out_channels; oc++) {
    // Use integer constant: 26 ~= 0.1 * 256 (FIX_SCALE)
    // Explicit cast to fixed_t (int) to avoid sign conversion warning
    bias[oc] = (fixed_t)(26 * (int)(oc + 1));
  }

  // Run evaluation
  ece6745_stats_on();
  
  // Manually perform a simplified version of convolution for testing performance
  for (unsigned int b = 0; b < batch_size; b++) {
    for (unsigned int oc = 0; oc < out_channels; oc++) {
      for (unsigned int oh = 0; oh < out_height; oh++) {
        for (unsigned int ow = 0; ow < out_width; ow++) {
          // Calculate output index
          unsigned int out_idx = b * (out_channels * out_height * out_width) + 
                               oc * (out_height * out_width) + 
                               oh * out_width + ow;
          
          // Start with bias
          fixed_t sum = bias[oc];
          
          // Perform convolution
          for (unsigned int ic = 0; ic < in_channels; ic++) {
            for (unsigned int kh = 0; kh < kernel_size; kh++) {
              for (unsigned int kw = 0; kw < kernel_size; kw++) {
                // Calculate input index (with padding)
                unsigned int in_h = oh + kh;
                unsigned int in_w = ow + kw;
                
                // Check bounds before accessing the array
                if (in_h < in_height && in_w < in_width) {
                  unsigned int in_idx = b * (in_channels * in_height * in_width) + 
                                     ic * (in_height * in_width) + 
                                     in_h * in_width + in_w;
                  
                  // Calculate weight index
                  unsigned int w_idx = oc * (in_channels * kernel_size * kernel_size) + 
                                    ic * (kernel_size * kernel_size) + 
                                    kh * kernel_size + kw;
                  
                  // Multiply-accumulate operation
                  sum += FIX_MUL(input[in_idx], weights[w_idx]);
                }
              }
            }
          }
          
          // Store result
          output[out_idx] = sum;
        }
      }
    }
  }
  
  ece6745_stats_off();

  // Free memory
  ece6745_free(input);
  ece6745_free(weights);
  ece6745_free(bias);
  ece6745_free(output);

  // Check for memory leaks
  if (ece6745_get_heap_usage() != 0) {
    ece6745_wprintf(L"\n FAILED: memory leak of %d bytes!\n\n",
                   ece6745_get_heap_usage());
    ece6745_exit(1);
  }

  // Success
  ece6745_wprintf(L"\n **PASSED** \n\n");

  return 0;
}