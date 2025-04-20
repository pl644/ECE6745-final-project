//========================================================================
// ubmark-cnn
//========================================================================
// This microbenchmark implements a simple CNN with convolution layer
// using fixed-point arithmetic

#ifndef UBMARK_MLP_H
#define UBMARK_MLP_H


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
);

#endif /* UBMARK_CNN_H */