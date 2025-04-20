//========================================================================
// Fully Connected Layer Verilog Implementation
//========================================================================

`ifndef MLP_XCEL_FULLY_CONNECTED_LAYER_V
`define MLP_XCEL_FULLY_CONNECTED_LAYER_V

`include "vc/trace.v"
`include "vc/counters.v"
`include "vc/muxes.v"
`include "vc/regs.v"
`include "vc/arithmetic.v"

//========================================================================
// Processing Element for Systolic Array
//========================================================================

module mlp_xcel_SystolicPE (
  input  logic        clk,
  input  logic        reset,
  input  logic        pe_en,      // Enable signal
  
  input  logic [11:0] weight,     // Weight register value
  input  logic        weight_en,  // Enable weight loading
  
  input  logic [11:0] act_in,     // Activation input from west
  input  logic [11:0] sum_in,     // Partial sum input from north
  
  output logic [11:0] act_out,    // Activation output to east
  output logic [11:0] sum_out     // Partial sum output to south
);
  // Internal registers
  logic [11:0] weight_reg;
  logic [11:0] act_reg;
  logic [11:0] sum_reg;
  logic [23:0] product;
  
  // Weight register
  always_ff @(posedge clk) begin
    if (reset) begin
      weight_reg <= 12'b0;
    end 
    else if (weight_en) begin
      weight_reg <= weight;
    end
  end
  
  // Forward activation register and accumulate
  always_ff @(posedge clk) begin
    if (reset) begin
      act_reg <= 12'b0;
      sum_reg <= 12'b0;
    end 
    else if (pe_en) begin
      // Pass activation to the next PE
      act_reg <= act_in;
      
      // Multiply and accumulate (MAC)
      product = act_reg * weight_reg;
      sum_reg <= sum_in + product[11:0]; // Use only lower 12 bits
    end
  end
  
  // Assign outputs
  assign act_out = act_reg;
  assign sum_out = sum_reg;
  
endmodule

//========================================================================
// Fully Connected Layer Datapath
//========================================================================

module mlp_xcel_FCLayerDpath 
#(
  parameter BATCH_SIZE = 2,
  parameter INPUT_CHANNEL = 3,
  parameter OUTPUT_CHANNEL = 2
)(
  input  logic        clk,
  input  logic        reset,

  // Data signals
  input  logic [31:0] istream_msg,
  output logic [31:0] ostream_msg,

  // Control signals (ctrl -> dpath)
  input  logic        weight_wen,
  input  logic        bias_wen,
  input  logic        input_wen,
  input  logic        pe_en,
  input  logic        clear_accum,
  input  logic        result_wen,

  // Status signals (dpath -> ctrl)
  output logic        msg_type_is_weight,
  output logic        msg_type_is_bias,
  output logic        msg_type_is_input,
  output logic        input_batch_done,
  output logic        all_outputs_done
);

  
  // Message type constants
  localparam MSG_TYPE_WEIGHT = 2'b00;
  localparam MSG_TYPE_BIAS   = 2'b01;
  localparam MSG_TYPE_INPUT  = 2'b10;
  localparam MSG_TYPE_OUTPUT = 2'b11;
  
  // Message parsing
  logic [1:0]  msg_type;
  logic [9:0]  msg_input_idx;
  logic [7:0]  msg_output_idx;
  logic [3:0]  msg_batch_idx;
  logic [11:0] msg_value;
  
  // Storage arrays
  logic [11:0] biases [OUTPUT_CHANNEL-1:0];
  logic [11:0] input_buffer [INPUT_CHANNEL-1:0];
  logic [11:0] output_buffer [OUTPUT_CHANNEL-1:0];
  logic [11:0] weight_buffer [INPUT_CHANNEL-1:0][OUTPUT_CHANNEL-1:0];
  
  // Input count tracking
  logic [7:0]  input_count_regs [BATCH_SIZE-1:0];
  logic [3:0]  batch_counter;
  logic [3:0]  compute_cycles;
  logic [BATCH_SIZE-1:0] batch_ready;
  
  // Parse message fields
  assign msg_type = istream_msg[31:30];
  
  // Status signals based on message type
  assign msg_type_is_weight = (msg_type == MSG_TYPE_WEIGHT);
  assign msg_type_is_bias   = (msg_type == MSG_TYPE_BIAS);
  assign msg_type_is_input  = (msg_type == MSG_TYPE_INPUT);
  
  // Extract message fields
  assign msg_input_idx  = (msg_type_is_weight) ? istream_msg[29:20] : 
                         ((msg_type_is_input)  ? istream_msg[25:16] : 10'b0);
  assign msg_output_idx = (msg_type_is_weight) ? istream_msg[19:12] : 
                         ((msg_type_is_bias)   ? istream_msg[29:22] : 8'b0);
  assign msg_batch_idx  = (msg_type_is_input)  ? istream_msg[29:26] : 4'b0;
  
  // Extract data value
  assign msg_value = istream_msg[11:0];
  
  // Input batch completion logic
  always_ff @(posedge clk) begin
    if (reset) begin
      for (int i = 0; i < BATCH_SIZE; i++) begin
        input_count_regs[i] <= 8'b0;
        batch_ready[i] <= 1'b0;
      end
    end
    else if (input_wen) begin
      // Increment the input count for the current batch
      input_count_regs[msg_batch_idx] <= input_count_regs[msg_batch_idx] + 1;
      
      // Check if this is the last input for the batch
      if (input_count_regs[msg_batch_idx] == INPUT_CHANNEL-1) begin
        batch_ready[msg_batch_idx] <= 1'b1;
      end
    end
    else if (clear_accum) begin
      // Reset the input count and batch ready flag when we're done processing
      input_count_regs[batch_counter] <= 8'b0;
      batch_ready[batch_counter] <= 1'b0;
    end
  end
  
  // Check if a batch has received all inputs
  assign input_batch_done = msg_type_is_input && 
                           ((batch_ready[msg_batch_idx]) || 
                           (input_count_regs[msg_batch_idx] == INPUT_CHANNEL-1));
  
  // Storage for weights, biases and input buffer
  always_ff @(posedge clk) begin
    if (reset) begin
      for (int i = 0; i < OUTPUT_CHANNEL; i++) begin
        biases[i] <= 12'b0;
      end
      
      for (int i = 0; i < INPUT_CHANNEL; i++) begin
        for (int j = 0; j < OUTPUT_CHANNEL; j++) begin
          weight_buffer[i][j] <= 12'b0;
        end
      end
    end
    else if (bias_wen && (msg_output_idx < OUTPUT_CHANNEL)) begin
      biases[msg_output_idx] <= msg_value;
    end
    else if (weight_wen && (msg_input_idx < INPUT_CHANNEL) && (msg_output_idx < OUTPUT_CHANNEL)) begin
      weight_buffer[msg_input_idx][msg_output_idx] <= msg_value;
    end
    
    if (input_wen && (msg_input_idx < INPUT_CHANNEL)) begin
      input_buffer[msg_input_idx] <= msg_value;
    end
  end
  
  // Count compute cycles
  always_ff @(posedge clk) begin
    if (reset || clear_accum) begin
      compute_cycles <= 4'b0;
    end
    else if (pe_en) begin
      compute_cycles <= compute_cycles + 1;
    end
  end
  
  // Batch counter
  always_ff @(posedge clk) begin
    if (reset) begin
      batch_counter <= 4'b0;
    end
    else if (clear_accum && all_outputs_done) begin
      batch_counter <= (batch_counter == BATCH_SIZE-1) ? 4'b0 : batch_counter + 1;
    end
  end
  
  //----------------------------------------------------------------------
  // Systolic Array Implementation
  //----------------------------------------------------------------------
  
  // PE signals
  logic [11:0] pe_act_in  [INPUT_CHANNEL][OUTPUT_CHANNEL];
  logic [11:0] pe_act_out [INPUT_CHANNEL][OUTPUT_CHANNEL];
  logic [11:0] pe_sum_in  [INPUT_CHANNEL][OUTPUT_CHANNEL];
  logic [11:0] pe_sum_out [INPUT_CHANNEL][OUTPUT_CHANNEL];
  logic        pe_wen     [INPUT_CHANNEL][OUTPUT_CHANNEL];
  logic [11:0] pe_weight  [INPUT_CHANNEL][OUTPUT_CHANNEL];
  
  // Initialize inputs and interconnect signals
  genvar i, j;
  generate
    for (i = 0; i < INPUT_CHANNEL; i++) begin : row_inputs
      for (j = 0; j < OUTPUT_CHANNEL; j++) begin : col_inputs
        // First column gets input from input_buffer
        if (j == 0) begin
          assign pe_act_in[i][j] = (pe_en && compute_cycles == 0) ? input_buffer[i] : 12'b0;
        end else begin
          // Other columns get input from previous column's output
          assign pe_act_in[i][j] = pe_act_out[i][j-1];
        end
        
        // First row gets zero sum_in
        if (i == 0) begin
          assign pe_sum_in[i][j] = 12'b0;
        end else begin
          // Other rows get input from row above
          assign pe_sum_in[i][j] = pe_sum_out[i-1][j];
        end
        
        // Weight loading logic
        assign pe_wen[i][j] = weight_wen && (msg_input_idx == i) && (msg_output_idx == j);
        assign pe_weight[i][j] = msg_value;
      end
    end
  endgenerate
  
  // Systolic Array of PEs
  generate
    for (i = 0; i < INPUT_CHANNEL; i++) begin : pe_rows
      for (j = 0; j < OUTPUT_CHANNEL; j++) begin : pe_cols
        // Instantiate processing element
        mlp_xcel_SystolicPE pe (
          .clk       (clk),
          .reset     (reset),
          .pe_en     (pe_en),
          .weight    (pe_weight[i][j]),
          .weight_en (pe_wen[i][j]),
          .act_in    (pe_act_in[i][j]),
          .sum_in    (pe_sum_in[i][j]),
          .act_out   (pe_act_out[i][j]),
          .sum_out   (pe_sum_out[i][j])
        );
      end
    end
  endgenerate
  
  // Accumulated outputs from systolic array
  logic [11:0] pe_accumulated [OUTPUT_CHANNEL-1:0];
  
  // Initialize accumulated outputs
  always_ff @(posedge clk) begin
    if (reset || clear_accum) begin
      for (int j = 0; j < OUTPUT_CHANNEL; j++) begin
        pe_accumulated[j] <= 12'b0;
      end
    end
    else begin
      // Accumulate outputs from the bottom row of PEs at each cycle
      for (int j = 0; j < OUTPUT_CHANNEL; j++) begin
        pe_accumulated[j] <= pe_accumulated[j] + pe_sum_out[INPUT_CHANNEL-1][j];
      end
    end
  end
  
  // Capture outputs from bottom row
  always_ff @(posedge clk) begin
    if (reset) begin
      for (int j = 0; j < OUTPUT_CHANNEL; j++) begin
        output_buffer[j] <= 12'b0;
      end
    end
    else if (compute_cycles >= INPUT_CHANNEL + 3) begin // Allow extra cycles for propagation
      for (int j = 0; j < OUTPUT_CHANNEL; j++) begin
        output_buffer[j] <= pe_accumulated[j] + biases[j];
      end
    end
  end
  
  // Output message construction
  logic [1:0]  out_type;
  logic [3:0]  out_batch;
  logic [7:0]  out_channel;
  logic [17:0] out_value;
  logic [17:0] out_value_array [OUTPUT_CHANNEL-1:0];
  logic [7:0] output_counter;
  
  assign out_type = MSG_TYPE_OUTPUT;
  
  // Output value array update
  always_ff @(posedge clk) begin
    if (reset) begin
      for (int j = 0; j < OUTPUT_CHANNEL; j++) begin
        out_value_array[j] <= 18'b0;
      end
    end
    else if (compute_cycles >= INPUT_CHANNEL) begin
      for (int j = 0; j < OUTPUT_CHANNEL; j++) begin
        // Sign extend the result for 18-bit output
        if (output_buffer[j][11]) // Negative number
          out_value_array[j] <= {6'b111111, output_buffer[j]};
        else
          out_value_array[j] <= {6'b000000, output_buffer[j]};
      end
    end
  end

  // Output registers
  always_ff @(posedge clk) begin
    if (reset) begin
      out_batch <= 4'b0;
      out_channel <= 8'b0;
      out_value <= 18'b0;
    end
    else if (result_wen) begin
      out_batch <= batch_counter;
      out_channel <= output_counter;
      out_value <= out_value_array[output_counter];
    end
  end
  
  // Output message
  assign ostream_msg = {out_type, out_batch, out_channel, out_value};
  
  always_ff @(posedge clk) begin
    if (reset || (clear_accum && all_outputs_done)) begin
      output_counter <= 8'b0;
    end
    else if (clear_accum) begin
      output_counter <= output_counter + 1;
    end
  end
  
  // Signal when all outputs for the current batch are done
  assign all_outputs_done = (output_counter == OUTPUT_CHANNEL-1);

endmodule

//========================================================================
// Fully Connected Layer Control Unit
//========================================================================

module mlp_xcel_FCLayerCtrl (
  input  logic clk,
  input  logic reset,

  // Dataflow signals
  input  logic istream_val,
  output logic istream_rdy,
  
  output logic ostream_val,
  input  logic ostream_rdy,

  // Control signals (ctrl -> dpath)
  output logic weight_wen,
  output logic bias_wen,
  output logic input_wen,
  output logic pe_en,         
  output logic clear_accum,
  output logic result_wen,

  // Status signals (dpath -> ctrl)
  input  logic msg_type_is_weight,
  input  logic msg_type_is_bias,
  input  logic msg_type_is_input,
  input  logic input_batch_done,
  input  logic all_outputs_done
);
  // FSM states
  localparam STATE_IDLE = 3'd0;
  localparam STATE_CONFIG = 3'd1;
  localparam STATE_WAIT_INPUTS = 3'd2;
  localparam STATE_COMPUTE = 3'd3;
  localparam STATE_SEND_OUTPUT = 3'd4;
  localparam STATE_PREPARE_NEXT = 3'd5;  // New state
  
  // State register
  logic [2:0] state_reg;
  logic [2:0] state_next;
  
  // Output registers
  logic pending_output;
  
  // Compute cycle counter
  logic [3:0] compute_cycles;
  logic compute_done;
  
  // New preparation counter
  logic [1:0] prep_counter;
  
  // State transition logic
  always_ff @(posedge clk) begin
    if (reset) begin
      state_reg <= STATE_IDLE;
      compute_cycles <= 4'b0;
      pending_output <= 1'b0;
      prep_counter <= 2'b0;
    end
    else begin
      state_reg <= state_next;
      
      // Handle compute cycle counter
      if (state_reg == STATE_COMPUTE) begin
        compute_cycles <= compute_cycles + 1;
      end else begin
        compute_cycles <= 4'b0;
      end
      
      // Handle preparation counter
      if (state_reg == STATE_PREPARE_NEXT) begin
        prep_counter <= prep_counter + 1;
      end else begin
        prep_counter <= 2'b0;
      end
      
      // Track when we have an output ready to send
      if (result_wen) begin
        pending_output <= 1'b1;
      end else if (state_reg == STATE_SEND_OUTPUT && ostream_rdy) begin
        pending_output <= 1'b0;
      end
    end
  end
  
  // Compute is done after processing enough cycles for systolic array
  assign compute_done = (compute_cycles >= 4'd8);
  
  // State transition logic
  always_comb begin
    state_next = state_reg; // Default: stay in current state
    
    case (state_reg)
      STATE_IDLE: begin
        if (istream_val) begin
          state_next = STATE_CONFIG;
        end
      end
      
      STATE_CONFIG: begin
        if (msg_type_is_input && input_batch_done) begin
          state_next = STATE_COMPUTE;
        end else begin
          state_next = STATE_WAIT_INPUTS;
        end
      end
      
      STATE_WAIT_INPUTS: begin
        if (istream_val && msg_type_is_input && input_batch_done) begin
          state_next = STATE_COMPUTE;
        end else if (istream_val) begin
          state_next = STATE_WAIT_INPUTS;
        end
      end
      
      STATE_COMPUTE: begin
        if (compute_done) begin
          state_next = STATE_SEND_OUTPUT;
        end
      end
      
      STATE_SEND_OUTPUT: begin
        if (ostream_rdy) begin
          if (all_outputs_done) begin
            state_next = STATE_IDLE;
          end else begin
            // Go to prepare next state instead of back to compute
            state_next = STATE_PREPARE_NEXT;
          end
        end
      end
      
      STATE_PREPARE_NEXT: begin
        // After enough cycles, move back to send output
        if (prep_counter >= 2'b01) begin
          state_next = STATE_SEND_OUTPUT;
        end
      end
      
      default: state_next = STATE_IDLE;
    endcase
  end
  
  // Control signal generation
  always_comb begin
    // Default values
    istream_rdy = 1'b0;
    ostream_val = 1'b0;
    weight_wen = 1'b0;
    bias_wen = 1'b0;
    input_wen = 1'b0;
    pe_en = 1'b0;
    clear_accum = 1'b0;
    result_wen = 1'b0;
    
    case (state_reg)
      STATE_IDLE, STATE_CONFIG, STATE_WAIT_INPUTS: begin
        istream_rdy = 1'b1;
        
        if (istream_val) begin
          weight_wen = msg_type_is_weight;
          bias_wen = msg_type_is_bias;
          input_wen = msg_type_is_input;
        end
      end
      
      STATE_COMPUTE: begin
        pe_en = 1'b1;
        
        if (compute_done) begin
          result_wen = 1'b1;
        end
      end
      
      STATE_SEND_OUTPUT: begin
        ostream_val = pending_output;
        
        if (ostream_rdy && pending_output) begin
          clear_accum = 1'b1;
        end
      end
      
      STATE_PREPARE_NEXT: begin
        // On the second cycle, set result_wen to prepare next output
        if (prep_counter == 2'b01) begin
          result_wen = 1'b1;
        end
      end
      
      default: begin
        // Default case to handle all other possible states
        istream_rdy = 1'b0;
        ostream_val = 1'b0;
        weight_wen = 1'b0;
        bias_wen = 1'b0;
        input_wen = 1'b0;
        pe_en = 1'b0;
        clear_accum = 1'b0;
        result_wen = 1'b0;
      end
    endcase
  end

endmodule

//========================================================================
// Fully Connected Layer Top-Level Module
//========================================================================

module mlp_xcel_FullyConnected 
(
  input  logic        clk,
  input  logic        reset,

  input  logic        istream_val,
  output logic        istream_rdy,
  input  logic [31:0] istream_msg,

  output logic        ostream_val,
  input  logic        ostream_rdy,
  output logic [31:0] ostream_msg
);
  // Control signals
  logic weight_wen;
  logic bias_wen;
  logic input_wen;
  logic pe_en;
  logic clear_accum;
  logic result_wen;

  // Status signals
  logic msg_type_is_weight;
  logic msg_type_is_bias;
  logic msg_type_is_input;
  logic input_batch_done;
  logic all_outputs_done;

  // Instantiate datapath
  mlp_xcel_FCLayerDpath dpath (
    .clk              (clk),
    .reset            (reset),
    .istream_msg      (istream_msg),
    .ostream_msg      (ostream_msg),
    .weight_wen       (weight_wen),
    .bias_wen         (bias_wen),
    .input_wen        (input_wen),
    .pe_en            (pe_en),
    .clear_accum      (clear_accum),
    .result_wen       (result_wen),
    .msg_type_is_weight (msg_type_is_weight),
    .msg_type_is_bias   (msg_type_is_bias),
    .msg_type_is_input  (msg_type_is_input),
    .input_batch_done   (input_batch_done),
    .all_outputs_done   (all_outputs_done)
  );

  // Instantiate control unit
  mlp_xcel_FCLayerCtrl ctrl (
    .clk              (clk),
    .reset            (reset),
    .istream_val      (istream_val),
    .istream_rdy      (istream_rdy),
    .ostream_val      (ostream_val),
    .ostream_rdy      (ostream_rdy),
    .weight_wen       (weight_wen),
    .bias_wen         (bias_wen),
    .input_wen        (input_wen),
    .pe_en            (pe_en),
    .clear_accum      (clear_accum),
    .result_wen       (result_wen),
    .msg_type_is_weight (msg_type_is_weight),
    .msg_type_is_bias   (msg_type_is_bias),
    .msg_type_is_input  (msg_type_is_input),
    .input_batch_done   (input_batch_done),
    .all_outputs_done   (all_outputs_done)
  );

  //----------------------------------------------------------------------
  // Line Tracing
  //----------------------------------------------------------------------

`ifndef SYNTHESIS

logic [`VC_TRACE_NBITS-1:0] str;
`VC_TRACE_BEGIN
begin
  $sformat( str, "%x", istream_msg );
  vc_trace.append_val_rdy_str( trace_str, istream_val, istream_rdy, str );

  vc_trace.append_str( trace_str, "(" );

  // Adding useful state information to the trace
  case ( ctrl.state_reg )
    ctrl.STATE_IDLE:        vc_trace.append_str( trace_str, "I" );
    ctrl.STATE_CONFIG:      vc_trace.append_str( trace_str, "C" );
    ctrl.STATE_WAIT_INPUTS: vc_trace.append_str( trace_str, "W" );
    ctrl.STATE_COMPUTE:     vc_trace.append_str( trace_str, "P" );
    ctrl.STATE_SEND_OUTPUT: vc_trace.append_str( trace_str, "O" );
    ctrl.STATE_PREPARE_NEXT: vc_trace.append_str( trace_str, "N" );
    default:                vc_trace.append_str( trace_str, "?" );
  endcase
  
  // Show systolic array calculation information
  $sformat( str, " b:%d o:%d cc:%d out_value:%d output_counter:%d result_wen:%d out_value_array:%d %d", 
            dpath.batch_counter,
            dpath.output_counter,
            dpath.compute_cycles,
            dpath.out_value,
            dpath.output_counter,
            dpath.result_wen,
            dpath.out_value_array[0],
            dpath.out_value_array[1]
          );
  vc_trace.append_str( trace_str, str );

  vc_trace.append_str( trace_str, ")" );

  $sformat( str, "%x", ostream_msg );
  vc_trace.append_val_rdy_str( trace_str, ostream_val, ostream_rdy, str );
end
`VC_TRACE_END

  `endif /* SYNTHESIS */

endmodule

`endif /* MLP_XCEL_FULLY_CONNECTED_LAYER_V */
