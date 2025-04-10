//========================================================================
// Systolic Array Processing Element (PE)
//========================================================================

module mlp_xcel_SystolicPE(
    input  logic [31:0] weight_in,  // Weight input
    input  logic [31:0] act_in,     // Activation input from west
    input  logic [31:0] sum_in,     // Partial sum input from north
    
    output logic [31:0] act_out,    // Activation output to east
    output logic [31:0] sum_out     // Partial sum output to south
);
    // Multiply-accumulate operation
    logic [31:0] product;
    
    // Compute product
    assign product = weight_in * act_in;
    
    // Compute sum
    assign sum_out = sum_in + product;
    
    // Forward activation
    assign act_out = act_in;
    
endmodule