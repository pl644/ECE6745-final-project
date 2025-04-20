#include <stdio.h>
#include <stdlib.h>
#include "ubmark-mlp.h"

// Define Q4.7 fixed-point format
#define FIXED_POINT_BITS 7
#define FIXED_POINT_SCALE (1 << FIXED_POINT_BITS)  // 128
#define FIXED_MULT(a, b) (((a) * (b) + (FIXED_POINT_SCALE >> 1)) >> FIXED_POINT_BITS)

// Network dimensions


// Simple MNIST inference function
void mnist_inference(
    int* input,                          // Input data [batch_size][INPUT_SIZE]
    int* fc1_weights,                    // FC1 weights [INPUT_SIZE][HIDDEN1_SIZE]
    int* fc1_bias,                       // FC1 bias [HIDDEN1_SIZE]
    int* fc2_weights,                    // FC2 weights [HIDDEN1_SIZE][HIDDEN2_SIZE]
    int* fc2_bias,                       // FC2 bias [HIDDEN2_SIZE]
    int* fc3_weights,                    // FC3 weights [HIDDEN2_SIZE][OUTPUT_SIZE]
    int* fc3_bias,                       // FC3 bias [OUTPUT_SIZE]
    int* output,                         // Output buffer [batch_size][OUTPUT_SIZE]
    int batch_size,                       // Batch size
    int input_size,
    int hidden1_size,
    int hidden2_size,
    int output_size
) {

    int hidden1_output[10]; // batch_size * hidden1_size = 1 * 3
    int hidden2_output[10];
    // Process each sample in the batch
    for (int b = 0; b < batch_size; b++) {
        // First fully connected layer: input -> hidden1
        for (int h1 = 0; h1 < hidden1_size; h1++) {
            // Start with bias
            hidden1_output[b * hidden1_size + h1] = fc1_bias[h1];
            
            // Add weighted sum of inputs
            for (int i = 0; i < input_size; i++) {
                hidden1_output[b * hidden1_size + h1] += 
                    FIXED_MULT(input[b * input_size + i], fc1_weights[i * hidden1_size + h1]);
            }
            
            // Apply ReLU
            if (hidden1_output[b * hidden1_size + h1] < 0) {
                hidden1_output[b * hidden1_size + h1] = 0;
            }
        }
        
        // Second fully connected layer: hidden1 -> hidden2
        for (int h2 = 0; h2 < hidden2_size; h2++) {
            // Start with bias
            hidden2_output[b * hidden2_size + h2] = fc2_bias[h2];
            
            // Add weighted sum of hidden1 outputs
            for (int h1 = 0; h1 < hidden1_size; h1++) {
                hidden2_output[b * hidden2_size + h2] += 
                    FIXED_MULT(hidden1_output[b * hidden1_size + h1], fc2_weights[h1 * hidden2_size + h2]);
            }
            
            // Apply ReLU
            if (hidden2_output[b * hidden2_size + h2] < 0) {
                hidden2_output[b * hidden2_size + h2] = 0;
            }
        }
        
        // Third fully connected layer: hidden2 -> output
        for (int o = 0; o < output_size; o++) {
            // Start with bias
            output[b * output_size + o] = fc3_bias[o];
            
            // Add weighted sum of hidden2 outputs
            for (int h2 = 0; h2 < hidden2_size; h2++) {
                output[b * output_size + o] += 
                    FIXED_MULT(hidden2_output[b * hidden2_size + h2], fc3_weights[h2 * output_size + o]);
            }
            // No activation function on the output layer (raw logits)
        }
    }

}