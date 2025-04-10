//========================================================================
// Unit tests for ubmark fclayer
//========================================================================

#include "ece6745.h"
#include "ubmark-fclayer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

//------------------------------------------------------------------------
// Helper functions
//------------------------------------------------------------------------

// Helper function to read data from files
// void read_file(const char* filename, float* buffer, int size) {
//     FILE* fp = fopen(filename, "r");
//     if (!fp) {
//         ece6745_wprintf(L"Error: Could not open file %s\n", filename);
//         return;
//     }
    
//     for (int i = 0; i < size; i++) {
//         if (fscanf(fp, "%f", &buffer[i]) != 1) {
//             ece6745_wprintf(L"Error: Could not read all data from file %s\n", filename);
//             break;
//         }
//     }
    
//     fclose(fp);
// }

//------------------------------------------------------------------------
// Test small linear layer with fixed values
//------------------------------------------------------------------------

void test_case_1_small() {
    ECE6745_CHECK(L"test_case_1_small");
    
    // Small test case with known values
    const int batch_size = 2;
    const int input_features = 3;
    const int output_features = 2;
    
    // Input: 2 samples, 3 features each
    float input[] = {
        1.0f, 2.0f, 3.0f,  // Sample 1
        4.0f, 5.0f, 6.0f   // Sample 2
    };
    
    // Weights: 3 input features, 2 output features
    float weights[] = {
        0.1f, 0.2f,  // Weights for input feature 1
        0.3f, 0.4f,  // Weights for input feature 2
        0.5f, 0.6f   // Weights for input feature 3
    };
    
    // Bias: 2 output features
    float bias[] = {0.1f, 0.2f};
    
    // Output buffer
    float output[4] = {0};
    
    // Call the linear layer function - use ubmark_fclayer instead of ubmark_linear
    ubmark_fclayer(
        input,
        weights,
        bias,
        output,
        batch_size,
        input_features,
        output_features
    );
    
    // Expected values
    float expected_0 = 2.3f;  // Sample 1, output 0: (1*0.1 + 2*0.3 + 3*0.5) + 0.1 = 2.3
    float expected_1 = 3.0f;  // Sample 1, output 1: (1*0.2 + 2*0.4 + 3*0.6) + 0.2 = 3.0
    float expected_2 = 5.0f;  // Sample 2, output 0: (4*0.1 + 5*0.3 + 6*0.5) + 0.1 = 5.0
    float expected_3 = 6.6f;  // Sample 2, output 1: (4*0.2 + 5*0.4 + 6*0.6) + 0.2 = 6.6
    
    // Check results
    ECE6745_CHECK(L"Output[0] should equal 2.3");
    int pass_0 = fabsf(output[0] - expected_0) < 1e-5f;
    ece6745_wprintf(L"Got: %f, Expected: %f, Diff: %f\n", output[0], expected_0, fabsf(output[0] - expected_0));
    ECE6745_CHECK(L"Test output[0]");
    if (pass_0)
        ece6745_wprintf(L"PASSED\n");
    else
        ece6745_wprintf(L"FAILED\n");
    
    ECE6745_CHECK(L"Output[1] should equal 3.0");
    int pass_1 = fabsf(output[1] - expected_1) < 1e-5f;
    ece6745_wprintf(L"Got: %f, Expected: %f, Diff: %f\n", output[1], expected_1, fabsf(output[1] - expected_1));
    ECE6745_CHECK(L"Test output[1]");
    if (pass_1)
        ece6745_wprintf(L"PASSED\n");
    else
        ece6745_wprintf(L"FAILED\n");
    
    ECE6745_CHECK(L"Output[2] should equal 5.0");
    int pass_2 = fabsf(output[2] - expected_2) < 1e-5f;
    ece6745_wprintf(L"Got: %f, Expected: %f, Diff: %f\n", output[2], expected_2, fabsf(output[2] - expected_2));
    ECE6745_CHECK(L"Test output[2]");
    if (pass_2)
        ece6745_wprintf(L"PASSED\n");
    else
        ece6745_wprintf(L"FAILED\n");
    
    ECE6745_CHECK(L"Output[3] should equal 6.6");
    int pass_3 = fabsf(output[3] - expected_3) < 1e-5f;
    ece6745_wprintf(L"Got: %f, Expected: %f, Diff: %f\n", output[3], expected_3, fabsf(output[3] - expected_3));
    ECE6745_CHECK(L"Test output[3]");
    if (pass_3)
        ece6745_wprintf(L"PASSED\n");
    else
        ece6745_wprintf(L"FAILED\n");
}

//------------------------------------------------------------------------
// Test with data from PyTorch
//------------------------------------------------------------------------

// void test_case_2_pytorch_data() {
//     ECE6745_CHECK(L"test_case_2_pytorch_data");
    
//     // Parameters for the test (should match those in the Python script)
//     const int batch_size = 1024;
//     const int input_features = 1024;
//     const int output_features = 1024;
    
//     // Allocate memory for data - use size_t for malloc to avoid sign conversion warnings
//     float* input = (float*)malloc((size_t)batch_size * (size_t)input_features * sizeof(float));
//     float* weights = (float*)malloc((size_t)input_features * (size_t)output_features * sizeof(float));
//     float* bias = (float*)malloc((size_t)output_features * sizeof(float));
//     float* output = (float*)malloc((size_t)batch_size * (size_t)output_features * sizeof(float));
//     float* expected = (float*)malloc((size_t)batch_size * (size_t)output_features * sizeof(float));
    
//     if (!input || !weights || !bias || !output || !expected) {
//         ece6745_wprintf(L"Error: Memory allocation failed\n");
//         goto cleanup;
//     }
    
//     // Read input data from files
//     read_file("input0.data", input, batch_size * input_features);
//     read_file("input1.data", weights, input_features * output_features);
//     read_file("input2.data", bias, output_features);
//     read_file("output.data", expected, batch_size * output_features);
    
//     // Call the linear layer function - use ubmark_fclayer instead of ubmark_linear
//     ubmark_fclayer(
//         input,
//         weights,
//         bias,
//         output,
//         batch_size,
//         input_features,
//         output_features
//     );
    
//     // Verify results
//     int mismatch_count = 0;
//     const int max_mismatches_to_report = 10;
    
//     for (int i = 0; i < batch_size * output_features; i++) {
//         if (fabsf(output[i] - expected[i]) >= 1e-4f) {
//             if (mismatch_count < max_mismatches_to_report) {
//                 ece6745_wprintf(L"Mismatch at index %d: got %f, expected %f\n", 
//                                 i, output[i], expected[i]);
//             }
//             mismatch_count++;
//         }
//     }
    
//     ECE6745_CHECK(L"All values should match expected results");
//     ece6745_wprintf(L"Mismatches: %d out of %d\n", mismatch_count, batch_size * output_features);
//     ECE6745_CHECK(L"Test PyTorch output");
//     if (mismatch_count == 0)
//         ece6745_wprintf(L"PASSED\n");
//     else
//         ece6745_wprintf(L"FAILED\n");
    
// cleanup:
//     // Free allocated memory
//     free(input);
//     free(weights);
//     free(bias);
//     free(output);
//     free(expected);
// }

//------------------------------------------------------------------------
// main
//------------------------------------------------------------------------

int main(int argc, char** argv) {
    __n = (argc == 1) ? 0 : ece6745_atoi(argv[1]);
    
    if ((__n <= 0) || (__n == 1)) test_case_1_small();
    // if ((__n <= 0) || (__n == 2)) test_case_2_pytorch_data();
    
    ece6745_wprintf(L"\n\n");
    return ece6745_check_status;
}