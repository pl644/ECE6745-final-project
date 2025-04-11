//=========================================================================
// Fully Connected Layer using Systolic Array
//=========================================================================

`include "mlp_xcel/mnist_fc_layer_PE.v"

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
    // Implement the direct matrix multiplication
    genvar b, o, i;
    generate
        for (b = 0; b < BATCH_SIZE; b = b + 1) begin : batch_loop
            for (o = 0; o < OUTPUT_CHANNEL; o = o + 1) begin : output_loop
                // Start with bias
                logic [31:0] accum;
                assign accum = biases[o];
                
                // Add dot product of input and weights
                logic [31:0] products [INPUT_CHANNEL-1:0];
                logic [31:0] sums [INPUT_CHANNEL:0];
                
                assign sums[0] = accum;
                
                for (i = 0; i < INPUT_CHANNEL; i = i + 1) begin : input_loop
                    // Use PE for each multiply-accumulate operation
                    mlp_xcel_SystolicPE pe (
                        .weight_in(weights[i][o]),
                        .act_in(input_vector[b][i]),
                        .sum_in(sums[i]),
                        .act_out(), // Not used
                        .sum_out(sums[i+1])
                    );
                    
                    // Store individual products for tracing
                    assign products[i] = weights[i][o] * input_vector[b][i];
                end
                
                // Final output
                assign output_result[b][o] = sums[INPUT_CHANNEL];
            end
        end
    endgenerate

    //----------------------------------------------------------------------
    // Line Tracing
    //----------------------------------------------------------------------

    `ifndef SYNTHESIS
    
    // Trace strings and formatting variables
    logic [`VC_TRACE_NBITS-1:0] str;
    integer f, b_idx, o_idx, i_idx;
    
    // Line trace function
    `VC_TRACE_BEGIN
    begin
        // Header for line trace
        vc_trace.append_str(trace_str, "FC|");
        
        // Trace first batch, first few inputs
        vc_trace.append_str(trace_str, "in[0]:");
        for (i_idx = 0; i_idx < INPUT_CHANNEL && i_idx < 2; i_idx = i_idx + 1) begin
            $sformat(str, "%x", input_vector[0][i_idx]);
            vc_trace.append_str(trace_str, str);
            if (i_idx < INPUT_CHANNEL-1 && i_idx < 1) begin
                vc_trace.append_str(trace_str, ",");
            end
        end
        if (INPUT_CHANNEL > 2) begin
            vc_trace.append_str(trace_str, "...");
        end
        
        // Show weights for a sample connection
        vc_trace.append_str(trace_str, "|W[0][0]:");
        $sformat(str, "%x", weights[0][0]);
        vc_trace.append_str(trace_str, str);
        
        // Show bias for first output
        vc_trace.append_str(trace_str, "|B[0]:");
        $sformat(str, "%x", biases[0]);
        vc_trace.append_str(trace_str, str);
        
        // Process pipeline stages
        vc_trace.append_str(trace_str, "|(");
        
        // For first batch, show partial sums in the pipeline
        for (i_idx = 0; i_idx <= INPUT_CHANNEL && i_idx < 4; i_idx = i_idx + 1) begin
            b_idx = 0;  // First batch
            o_idx = 0;  // First output
            
            if (i_idx < INPUT_CHANNEL) begin
                $sformat(str, "%x", sums[i_idx]);
                vc_trace.append_str(trace_str, str);
                vc_trace.append_str(trace_str, "->");
            end else if (i_idx == INPUT_CHANNEL) begin
                $sformat(str, "%x", sums[i_idx]);
                vc_trace.append_str(trace_str, str);
            end
        end
        
        if (INPUT_CHANNEL > 4) begin
            vc_trace.append_str(trace_str, "...");
        }
        
        vc_trace.append_str(trace_str, ")");
        
        // Show output for first batch
        vc_trace.append_str(trace_str, "|out[0]:");
        for (o_idx = 0; o_idx < OUTPUT_CHANNEL && o_idx < 2; o_idx = o_idx + 1) begin
            $sformat(str, "%x", output_result[0][o_idx]);
            vc_trace.append_str(trace_str, str);
            if (o_idx < OUTPUT_CHANNEL-1 && o_idx < 1) begin
                vc_trace.append_str(trace_str, ",");
            end
        end
        if (OUTPUT_CHANNEL > 2) begin
            vc_trace.append_str(trace_str, "...");
        }
        
        vc_trace.append_str(trace_str, "|");
    end
    `VC_TRACE_END
    
    `endif /* SYNTHESIS */

endmodule