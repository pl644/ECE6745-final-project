//========================================================================
// Unit tests for ubmark linear layer
//========================================================================

#include "ece6745.h"
#include "ubmark-fclayer.h"
#include <cmath>
#include <stdio.h>

// Helper function to compare floats with tolerance
bool compare_float(float a, float b, float epsilon = 1e-4) {
    return std::fabs(a - b) < epsilon;
}

// Helper function to read data from files
void read_file(const char* filename, float* buffer, int size) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        ece6745_wprintf(L"Error: Could not open file %s\n", filename);
        return;
    }
    
    for (int i = 0; i < size; i++) {
        if (fscanf(fp, "%f", &buffer[i]) != 1) {
            ece6745_wprintf(L"Error: Could not read all data from file %s\n", filename);
            break;
        }
    }
    
    fclose(fp);
}

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
    
    // Expected output: 2 samples, 2 features each
    // Sample 1: (1*0.1 + 2*0.3 + 3*0.5) + 0.1 = 2.3
    //           (1*0.2 + 2*0.4 + 3*0.6) + 0.2 = 3.0
    // Sample 2: (4*0.1 + 5*0.3 + 6*0.5) + 0.1 = 5.0
    //           (4*0.2 + 5*0.4 + 6*0.6) + 0.2 = 6.6
    float expected[] = {
        2.3f, 3.0f,  // Output for sample 1
        5.0f, 6.6f   // Output for sample 2
    };
    
    // Output buffer
    float output[batch_size * output_features] = {0};
    
    // Call the linear layer function
    ubmark_fclayer(
        input,
        weights,
        bias,
        output,
        batch_size,
        input_features,
        output_features
    );
    
    // Verify each output value
    bool all_correct = true;
    for (int i = 0; i < batch_size * output_features; i++) {
        if (!compare_float(output[i], expected[i])) {
            ece6745_wprintf(L"Output[%d] = %f, Expected = %f\n", 
                           i, output[i], expected[i]);
            all_correct = false;
        }
    }
    
    ECE6745_CHECK_CUSTOM(all_correct, L"All values match expected results");
}

//------------------------------------------------------------------------
// Test with data from PyTorch
//------------------------------------------------------------------------

void test_case_2_pytorch_data() {
    ECE6745_CHECK(L"test_case_2_pytorch_data");
    
    // Parameters for the test (should match those in the Python script)
    const int batch_size = 1024;
    const int input_features = 1024;
    const int output_features = 1024;
    
    // Allocate memory for data
    float* input = new float[batch_size * input_features];
    float* weights = new float[input_features * output_features];
    float* bias = new float[output_features];
    float* output = new float[batch_size * output_features];
    float* expected = new float[batch_size * output_features];
    
    // Read input data from files
    read_file("input0.data", input, batch_size * input_features);
    read_file("input1.data", weights, input_features * output_features);
    read_file("input2.data", bias, output_features);
    read_file("output.data", expected, batch_size * output_features);
    
    // Call the linear layer function
    ubmark_fclayer(
        input,
        weights,
        bias,
        output,
        batch_size,
        input_features,
        output_features
    );
    
    // Verify results
    int mismatch_count = 0;
    const int max_mismatches_to_report = 10;
    
    for (int i = 0; i < batch_size * output_features; i++) {
        if (!compare_float(output[i], expected[i])) {
            if (mismatch_count < max_mismatches_to_report) {
                ece6745_wprintf(L"Mismatch at index %d: got %f, expected %f\n", 
                                i, output[i], expected[i]);
            }
            mismatch_count++;
        }
    }
    
    if (mismatch_count > 0) {
        if (mismatch_count > max_mismatches_to_report) {
            ece6745_wprintf(L"... and %d more mismatches\n", 
                            mismatch_count - max_mismatches_to_report);
        }
        ECE6745_CHECK_CUSTOM(false, L"Output matches expected values");
    } else {
        ECE6745_CHECK_CUSTOM(true, L"All output values match expected values");
    }
    
    // Free allocated memory
    delete[] input;
    delete[] weights;
    delete[] bias;
    delete[] output;
    delete[] expected;
}

//------------------------------------------------------------------------
// main
//------------------------------------------------------------------------

int main(int argc, char** argv) {
    __n = (argc == 1) ? 0 : ece6745_atoi(argv[1]);
    
    if ((__n <= 0) || (__n == 1)) test_case_1_small();
    if ((__n <= 0) || (__n == 2)) test_case_2_pytorch_data();
    
    ece6745_wprintf(L"\n\n");
    return ece6745_check_status;
}