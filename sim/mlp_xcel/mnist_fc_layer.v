module mlp_xcel_FullyConnected #(
    parameter BATCH_SIZE = 4,
    parameter INPUT_CHANNEL = 4,
    parameter OUTPUT_CHANNEL = 2
)(
    input  logic [31:0] input_vector [BATCH_SIZE-1:0][INPUT_CHANNEL-1:0],
    input  logic [31:0] weights [INPUT_CHANNEL-1:0][OUTPUT_CHANNEL-1:0],
    input  logic [31:0] biases [OUTPUT_CHANNEL-1:0],
    output logic [31:0] output_result [BATCH_SIZE-1:0][OUTPUT_CHANNEL-1:0]
);
    // Calculate outputs directly without systolic array structure
    // This simplifies the implementation and avoids signal connectivity issues
    
    genvar b, o, i;
    generate
        for (b = 0; b < BATCH_SIZE; b = b + 1) begin : batch_loop
            for (o = 0; o < OUTPUT_CHANNEL; o = o + 1) begin : output_loop
                // Each output starts with the bias value
                logic [31:0] sum;
                
                // Calculate the sum directly instead of using separate PEs
                always_comb begin
                    sum = biases[o];
                    for (int i = 0; i < INPUT_CHANNEL; i = i + 1) begin
                        sum = sum + (input_vector[b][i] * weights[i][o]);
                    end
                    output_result[b][o] = sum;
                end
            end
        end
    endgenerate
endmodule