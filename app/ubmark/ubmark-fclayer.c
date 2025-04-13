#include "ubmark-fclayer.h"

// Define Q4.7 fixed-point format
#define FIXED_POINT_BITS 7
#define FIXED_POINT_SCALE (1 << FIXED_POINT_BITS)  // 128
#define FIXED_MULT(a, b) (((a) * (b)) >> FIXED_POINT_BITS)

void ubmark_fclayer_fixed(int* input, int* weights, int* bias, int* output,
    int batch, int channel_in, int channel_out) {
    // Process each input vector
    for (int i = 0; i < batch; i++) {
        // Initialize output with bias values
        for (int j = 0; j < channel_out; j++) {
            output[i * channel_out + j] = bias[j];
        }

        // Perform matrix multiplication
        for (int k = 0; k < channel_in; k++) {
            int input_val = input[i * channel_in + k];

            for (int j = 0; j < channel_out; j++) {
                // Use fixed-point multiplication
                output[i * channel_out + j] += FIXED_MULT(input_val, weights[k * channel_out + j]);
            }
        }
    }
}