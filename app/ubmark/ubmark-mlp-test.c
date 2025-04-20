#include "ece6745.h"
#include <stdio.h>
#include <stdlib.h>
#include "ubmark-mlp.h"

// Define fixed-point helper macros
#define FIXED_POINT_BITS 7
#define FIXED_POINT_SCALE (1 << FIXED_POINT_BITS)  // 128
#define TO_FIXED(x) ((int)((x) * FIXED_POINT_SCALE + ((x) >= 0 ? 0.5f : -0.5f)))
#define FIXED_ABS(x) ((x) < 0 ? -(x) : (x))


// Integer-only printing function for Q4.7 fixed point
void print_fixed_point(int val) {
    int integer_part = val >> FIXED_POINT_BITS;
    int fraction_part = val & (FIXED_POINT_SCALE - 1);
    
    // Convert fraction to decimal for display (multiplying by 100 to get percentage)
    int decimal_part = (fraction_part * 100) / FIXED_POINT_SCALE;
    
    ece6745_wprintf(L"%d.%02d", integer_part, decimal_part);
}

void test_case_mnist_small() {
    ECE6745_CHECK(L"test_case_mnist_small");
    
    // Small test case with known values
    const int batch_size = 1;
    const int input_size = 4;    // Small input for testing (instead of 784)
    const int hidden1_size = 3;  // Small hidden layer (instead of 256)
    const int hidden2_size = 2;  // Small hidden layer (instead of 256)
    const int output_size = 2;   // Small output (instead of 10)
    
    // Convert float inputs to Q4.7 fixed-point
    int input_fixed[] = {
        TO_FIXED(0.5), TO_FIXED(-0.3), TO_FIXED(0.7), TO_FIXED(0.1)
    };
    
    // Layer 1 weights and biases (transposed for correct multiplication)
    int fc1_weights[] = {
        TO_FIXED(0.1), TO_FIXED(0.2), TO_FIXED(0.3),  // Weights for input 0
        TO_FIXED(-0.1), TO_FIXED(0.1), TO_FIXED(0.2), // Weights for input 1
        TO_FIXED(0.3), TO_FIXED(0.1), TO_FIXED(0.2),  // Weights for input 2
        TO_FIXED(0.2), TO_FIXED(0.3), TO_FIXED(0.1)   // Weights for input 3
    };
    
    int fc1_bias[] = {
        TO_FIXED(0.1), TO_FIXED(0.1), TO_FIXED(0.1)
    };
    
    // Layer 2 weights and biases
    int fc2_weights[] = {
        TO_FIXED(0.1), TO_FIXED(0.2),  // Weights for hidden1[0]
        TO_FIXED(0.3), TO_FIXED(0.1),  // Weights for hidden1[1]
        TO_FIXED(0.2), TO_FIXED(0.3)   // Weights for hidden1[2]
    };
    
    int fc2_bias[] = {
        TO_FIXED(0.1), TO_FIXED(0.2)
    };
    
    // Layer 3 weights and biases
    int fc3_weights[] = {
        TO_FIXED(0.1), TO_FIXED(0.2),  // Weights for hidden2[0]
        TO_FIXED(0.3), TO_FIXED(0.1)   // Weights for hidden2[1]
    };
    
    int fc3_bias[] = {
        TO_FIXED(0.1), TO_FIXED(0.1)
    };
    
    // Output buffer for fixed-point results
    int output_fixed[2] = {0};
    
    // Call the mnist inference function
    mnist_inference(
        input_fixed, 
        fc1_weights, fc1_bias,
        fc2_weights, fc2_bias,
        fc3_weights, fc3_bias,
        output_fixed, batch_size,input_size,hidden1_size,hidden2_size,output_size
    );
    
    // Expected values in fixed-point (manually calculated)
    // Layer 1 output (before ReLU):
    //   [0]: 0.1 + 0.5*0.1 + (-0.3)*(-0.1) + 0.7*0.3 + 0.1*0.2 = 0.41
    //   [1]: 0.1 + 0.5*0.2 + (-0.3)*0.1 + 0.7*0.1 + 0.1*0.3 = 0.27
    //   [2]: 0.1 + 0.5*0.3 + (-0.3)*0.2 + 0.7*0.2 + 0.1*0.1 = 0.34
    // After ReLU: [0.41, 0.27, 0.34]
    
    // Layer 2 output (before ReLU):
    //   [0]: 0.1 + 0.41*0.1 + 0.27*0.3 + 0.34*0.2 = 0.29
    //   [1]: 0.2 + 0.41*0.2 + 0.27*0.1 + 0.34*0.3 = 0.411
    // After ReLU: [0.29, 0.411]
    
    // Layer 3 output (final):
    //   [0]: 0.1 + 0.29*0.1 + 0.411*0.3 = 0.2523
    //   [1]: 0.1 + 0.29*0.2 + 0.411*0.1 = 0.1991
    
    int expected_output[] = {
        TO_FIXED(0.2523), 
        TO_FIXED(0.1991)
    };
    
    // Check results using integer-only display
    ECE6745_CHECK(L"Output[0] should equal 0.2523");
    int diff_0 = FIXED_ABS(output_fixed[0] - expected_output[0]);
    int pass_0 = diff_0 < 3;  // Allow small difference due to fixed-point rounding
    ece6745_wprintf(L"Got: ");
    print_fixed_point(output_fixed[0]);
    ece6745_wprintf(L" (%d), Expected: ", output_fixed[0]);
    print_fixed_point(expected_output[0]);
    ece6745_wprintf(L" (%d), Diff: %d\n", expected_output[0], diff_0);
    ECE6745_CHECK(L"Test output[0]");
    if (pass_0)
        ece6745_wprintf(L"PASSED\n");
    else
        ece6745_wprintf(L"FAILED\n");
    
    ECE6745_CHECK(L"Output[1] should equal 0.1991");
    int diff_1 = FIXED_ABS(output_fixed[1] - expected_output[1]);
    int pass_1 = diff_1 < 3;  // Allow small difference due to fixed-point rounding
    ece6745_wprintf(L"Got: ");
    print_fixed_point(output_fixed[1]);
    ece6745_wprintf(L" (%d), Expected: ", output_fixed[1]);
    print_fixed_point(expected_output[1]);
    ece6745_wprintf(L" (%d), Diff: %d\n", expected_output[1], diff_1);
    ECE6745_CHECK(L"Test output[1]");
    if (pass_1)
        ece6745_wprintf(L"PASSED\n");
    else
        ece6745_wprintf(L"FAILED\n");
}

int main(int argc, char** argv) {
    __n = (argc == 1) ? 0 : ece6745_atoi(argv[1]);
    
    if ((__n <= 0) || (__n == 1)) test_case_mnist_small();
    
    ece6745_wprintf(L"\n\n");
    return ece6745_check_status;
}