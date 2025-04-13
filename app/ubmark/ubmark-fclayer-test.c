//========================================================================
// Unit tests for ubmark fclayer
//========================================================================

#include "ece6745.h"
#include "ubmark-fclayer.h"
#include <stdio.h>
#include <stdlib.h>

// Define Q4.7 fixed-point format
#define FIXED_POINT_BITS 7
#define FIXED_POINT_SCALE (1 << FIXED_POINT_BITS)  // 128
#define TO_FIXED(x) ((int)((x) * FIXED_POINT_SCALE + 0.5f))
#define FIXED_MULT(a, b) (((a) * (b)) >> FIXED_POINT_BITS)
#define FIXED_ABS(x) ((x) < 0 ? -(x) : (x))

// Function prototype for fixed-point implementation
void ubmark_fclayer_fixed(
    int* input,
    int* weights,
    int* bias,
    int* output,
    int batch_size,
    int input_features,
    int output_features
);

// Integer-only printing function for Q4.7 fixed point
void print_fixed_point(int val) {
    int integer_part = val >> FIXED_POINT_BITS;
    int fraction_part = val & (FIXED_POINT_SCALE - 1);
    
    // Convert fraction to decimal for display (multiplying by 100 to get percentage)
    int decimal_part = (fraction_part * 100) / FIXED_POINT_SCALE;
    
    ece6745_wprintf(L"%d.%02d", integer_part, decimal_part);
}

void test_case_1_small() {
    ECE6745_CHECK(L"test_case_1_small");
    
    // Small test case with known values
    const int batch_size = 2;
    const int input_features = 3;
    const int output_features = 2;
    
    // Convert float inputs to Q4.7 fixed-point
    int input_fixed[] = {
        TO_FIXED(1.0f), TO_FIXED(2.0f), TO_FIXED(3.0f),  // Sample 1
        TO_FIXED(4.0f), TO_FIXED(5.0f), TO_FIXED(6.0f)   // Sample 2
    };
    
    // Convert float weights to Q4.7 fixed-point
    int weights_fixed[] = {
        TO_FIXED(0.1f), TO_FIXED(0.2f),  // Weights for input feature 1
        TO_FIXED(0.3f), TO_FIXED(0.4f),  // Weights for input feature 2
        TO_FIXED(0.5f), TO_FIXED(0.6f)   // Weights for input feature 3
    };
    
    // Convert float bias to Q4.7 fixed-point
    int bias_fixed[] = {TO_FIXED(0.1f), TO_FIXED(0.2f)};
    
    // Output buffer for fixed-point results
    int output_fixed[4] = {0};
    
    // Call the fixed-point linear layer function
    ubmark_fclayer_fixed(
        input_fixed,
        weights_fixed,
        bias_fixed,
        output_fixed,
        batch_size,
        input_features,
        output_features
    );
    
    // Expected values in fixed-point
    int expected_0_fixed = TO_FIXED(2.3f);  // Sample 1, output 0: (1*0.1 + 2*0.3 + 3*0.5) + 0.1 = 2.3
    int expected_1_fixed = TO_FIXED(3.0f);  // Sample 1, output 1: (1*0.2 + 2*0.4 + 3*0.6) + 0.2 = 3.0
    int expected_2_fixed = TO_FIXED(5.0f);  // Sample 2, output 0: (4*0.1 + 5*0.3 + 6*0.5) + 0.1 = 5.0
    int expected_3_fixed = TO_FIXED(6.6f);  // Sample 2, output 1: (4*0.2 + 5*0.4 + 6*0.6) + 0.2 = 6.6
    
    // Check results using integer-only display
    ECE6745_CHECK(L"Output[0] should equal 2.3");
    int diff_0 = FIXED_ABS(output_fixed[0] - expected_0_fixed);
    int pass_0 = diff_0 < 3;  // Allow small difference due to fixed-point rounding
    ece6745_wprintf(L"Got: ");
    print_fixed_point(output_fixed[0]);
    ece6745_wprintf(L" (%d), Expected: ", output_fixed[0]);
    print_fixed_point(expected_0_fixed);
    ece6745_wprintf(L" (%d), Diff: %d\n", expected_0_fixed, diff_0);
    ECE6745_CHECK(L"Test output[0]");
    if (pass_0)
        ece6745_wprintf(L"PASSED\n");
    else
        ece6745_wprintf(L"FAILED\n");
    
    ECE6745_CHECK(L"Output[1] should equal 3.0");
    int diff_1 = FIXED_ABS(output_fixed[1] - expected_1_fixed);
    int pass_1 = diff_1 < 3;  // Allow small difference due to fixed-point rounding
    ece6745_wprintf(L"Got: ");
    print_fixed_point(output_fixed[1]);
    ece6745_wprintf(L" (%d), Expected: ", output_fixed[1]);
    print_fixed_point(expected_1_fixed);
    ece6745_wprintf(L" (%d), Diff: %d\n", expected_1_fixed, diff_1);
    ECE6745_CHECK(L"Test output[1]");
    if (pass_1)
        ece6745_wprintf(L"PASSED\n");
    else
        ece6745_wprintf(L"FAILED\n");
    
    ECE6745_CHECK(L"Output[2] should equal 5.0");
    int diff_2 = FIXED_ABS(output_fixed[2] - expected_2_fixed);
    int pass_2 = diff_2 < 3;  // Allow small difference due to fixed-point rounding
    ece6745_wprintf(L"Got: ");
    print_fixed_point(output_fixed[2]);
    ece6745_wprintf(L" (%d), Expected: ", output_fixed[2]);
    print_fixed_point(expected_2_fixed);
    ece6745_wprintf(L" (%d), Diff: %d\n", expected_2_fixed, diff_2);
    ECE6745_CHECK(L"Test output[2]");
    if (pass_2)
        ece6745_wprintf(L"PASSED\n");
    else
        ece6745_wprintf(L"FAILED\n");
    
    ECE6745_CHECK(L"Output[3] should equal 6.6");
    int diff_3 = FIXED_ABS(output_fixed[3] - expected_3_fixed);
    int pass_3 = diff_3 < 3;  // Allow small difference due to fixed-point rounding
    ece6745_wprintf(L"Got: ");
    print_fixed_point(output_fixed[3]);
    ece6745_wprintf(L" (%d), Expected: ", output_fixed[3]);
    print_fixed_point(expected_3_fixed);
    ece6745_wprintf(L" (%d), Diff: %d\n", expected_3_fixed, diff_3);
    ECE6745_CHECK(L"Test output[3]");
    if (pass_3)
        ece6745_wprintf(L"PASSED\n");
    else
        ece6745_wprintf(L"FAILED\n");
}

int main(int argc, char** argv) {
    __n = (argc == 1) ? 0 : ece6745_atoi(argv[1]);
    
    if ((__n <= 0) || (__n == 1)) test_case_1_small();
    // if ((__n <= 0) || (__n == 2)) test_case_2_pytorch_data();
    
    ece6745_wprintf(L"\n\n");
    return ece6745_check_status;
}