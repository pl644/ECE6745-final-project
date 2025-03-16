//========================================================================
// ubmark-cnn
//========================================================================

#include "ubmark-cnn.h"

// Helper function to pad input
void ubmark_cnn_pad_input(
  fixed_t padded_input[4][3][66][66],
  fixed_t input[4][3][64][64]
) {
  // Initialize padded input to zeros - already initialized in the test case
  
  // Copy input data with padding
  for (int batch = 0; batch < 4; batch++) {
    for (int channel = 0; channel < 3; channel++) {
      for (int row = 0; row < 64; row++) {
        for (int col = 0; col < 64; col++) {
          padded_input[batch][channel][row + 1][col + 1] = input[batch][channel][row][col];
        }
      }
    }
  }
}

// Main convolution function
void ubmark_cnn_conv(
  fixed_t output[4][16][63][63],
  fixed_t input[4][3][64][64],
  fixed_t weights[16][3][4][4],
  fixed_t bias[16]
) {
  fixed_t padded_input[4][3][66][66];
  
  // Initialize padded input to zero
  for (int batch = 0; batch < 4; batch++) {
    for (int channel = 0; channel < 3; channel++) {
      for (int row = 0; row < 66; row++) {
        for (int col = 0; col < 66; col++) {
          padded_input[batch][channel][row][col] = 0;
        }
      }
    }
  }
  
  // Pad the input
  ubmark_cnn_pad_input(padded_input, input);

  // Convolution operation using fixed-point arithmetic
  for (int batch = 0; batch < 4; batch++) {
    for (int out_channel = 0; out_channel < 16; out_channel++) {
      for (int out_row = 0; out_row < 63; out_row++) {
        for (int out_col = 0; out_col < 63; out_col++) {
          fixed_t sum = bias[out_channel]; // Start with bias
          for (int in_channel = 0; in_channel < 3; in_channel++) {
            for (int kernel_row = 0; kernel_row < 4; kernel_row++) {
              for (int kernel_col = 0; kernel_col < 4; kernel_col++) {
                // Fixed-point multiplication and accumulation
                fixed_t weight = weights[out_channel][in_channel][kernel_row][kernel_col];
                fixed_t pixel = padded_input[batch][in_channel][out_row + kernel_row][out_col + kernel_col];
                sum += FIX_MUL(weight, pixel);
              }
            }
          }
          output[batch][out_channel][out_row][out_col] = sum;
        }
      }
    }
  }
}