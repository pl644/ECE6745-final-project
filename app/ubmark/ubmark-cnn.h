//========================================================================
// ubmark-cnn
//========================================================================
// This microbenchmark implements a simple CNN with convolution layer
// using fixed-point arithmetic

#ifndef UBMARK_CNN_H
#define UBMARK_CNN_H

// Define fixed-point representation
// We use 16-bit integers with 8 bits for the fractional part
typedef int fixed_t;

// Conversion factor for fixed-point arithmetic
#define FIX_SCALE 256
#define FIX_SHIFT 8

// Convert float to fixed-point
#define FLOAT_TO_FIX(f) ((fixed_t)((f) * FIX_SCALE))

// Convert fixed-point to float (for testing only)
#define FIX_TO_FLOAT(f) (((float)(f)) / FIX_SCALE)

// Fixed-point multiply
#define FIX_MUL(a, b) (((a) * (b)) >> FIX_SHIFT)

// Fixed-point addition is just normal addition

// Main function to perform convolution operation
void ubmark_cnn_conv(
  fixed_t output[4][16][63][63],
  fixed_t input[4][3][64][64],
  fixed_t weights[16][3][4][4],
  fixed_t bias[16]
);

// Helper functions
void ubmark_cnn_pad_input(
  fixed_t padded_input[4][3][66][66],
  fixed_t input[4][3][64][64]
);

#endif /* UBMARK_CNN_H */