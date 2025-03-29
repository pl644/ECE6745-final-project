
#include "ubmark-fclayer.h"
void linear_fclayer(float* input, float* weights, float* bias, float* output,
    int batch, int channel_in, int channel_out) {
// Process each input vector
for (int i = 0; i < batch; i++) {
// Initialize output with bias values
for (int j = 0; j < channel_out; j++) {
output[i * channel_out + j] = bias[j];
}

// Perform matrix multiplication
for (int k = 0; k < channel_in; k++) {
float input_val = input[i * channel_in + k];

for (int j = 0; j < channel_out; j++) {
output[i * channel_out + j] += input_val * weights[k * channel_out + j];
}
}
}
}