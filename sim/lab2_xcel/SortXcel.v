//=========================================================================
// Sorting Accelerator Implementation (Merge Sort)
//=========================================================================
// Sort array in memory containing positive integers.
// Accelerator register interface:
//
//  xr0 : go/done
//  xr1 : base address of array
//  xr2 : number of elements in array
//
// This implementation reads all elements into internal storage first,
// then sorts them, and finally writes back to memory.
//
// We use Claude 3.7 Sonnet as an assistant to draft our architecture. 

`ifndef LAB2_SORT_SORT_XCEL_V
`define LAB2_SORT_SORT_XCEL_V

`include "vc/trace.v"

`include "vc/mem-msgs.v"
`include "vc/xcel-msgs.v"
`include "vc/queues.v"

//=========================================================================
// Sorting Accelerator Implementation
//=========================================================================

module lab2_xcel_SortXcel
(
  input  logic         clk,
  input  logic         reset,

  input  xcel_req_t    xcel_reqstream_msg,
  input  logic         xcel_reqstream_val,
  output logic         xcel_reqstream_rdy,

  output xcel_resp_t   xcel_respstream_msg,
  output logic         xcel_respstream_val,
  input  logic         xcel_respstream_rdy,

  output mem_req_4B_t  mem_reqstream_msg,
  output logic         mem_reqstream_val,
  input  logic         mem_reqstream_rdy,

  input  mem_resp_4B_t mem_respstream_msg,
  input  logic         mem_respstream_val,
  output logic         mem_respstream_rdy
);

  // 4-state sim fix: force outputs to be zero if invalid
  xcel_resp_t  xcel_respstream_msg_raw;
  mem_req_4B_t mem_reqstream_msg_raw;

  assign xcel_respstream_msg = xcel_respstream_msg_raw & {33{xcel_respstream_val}};
  assign mem_reqstream_msg   = mem_reqstream_msg_raw & {78{mem_reqstream_val}};

  // Accelerator ports and queues
  logic      xcelreq_deq_val;
  logic      xcelreq_deq_rdy;
  xcel_req_t xcelreq_deq_msg;

  vc_Queue#(`VC_QUEUE_PIPE,$bits(xcel_req_t),1) xcelreq_q
  (
    .clk     (clk),
    .reset   (reset),
    .num_free_entries(),

    .enq_val (xcel_reqstream_val),
    .enq_rdy (xcel_reqstream_rdy),
    .enq_msg (xcel_reqstream_msg),

    .deq_val (xcelreq_deq_val),
    .deq_rdy (xcelreq_deq_rdy),
    .deq_msg (xcelreq_deq_msg)
  );

  // Memory ports and queues
  logic         memresp_deq_val;
  logic         memresp_deq_rdy;
  mem_resp_4B_t memresp_deq_msg;

  vc_Queue#(`VC_QUEUE_PIPE,$bits(mem_resp_4B_t),1) memresp_q
  (
    .clk     (clk),
    .reset   (reset),
    .num_free_entries(),

    .enq_val (mem_respstream_val),
    .enq_rdy (mem_respstream_rdy),
    .enq_msg (mem_respstream_msg),

    .deq_val (memresp_deq_val),
    .deq_rdy (memresp_deq_rdy),
    .deq_msg (memresp_deq_msg)
  );

  // Main configuration registers
  logic [31:0] size, size_next;             // Array size
  logic [31:0] base_addr, base_addr_next;   // Base address of array
  
  // Internal buffer for storing array elements (max 64 elements)
  logic [31:0] buffer [0:127];   
  logic [31:0] temp [0:127];     
  
  logic buffer_wen;
  logic [6:0] buffer_waddr;
  logic [31:0] buffer_wdata;
  
  logic temp_wen;
  logic [6:0] temp_waddr;
  logic [31:0] temp_wdata;
  logic [31:0] read_idx, read_idx_next;   
  logic [31:0] write_idx, write_idx_next;   
  logic [31:0] merge_width, merge_width_next; 
  logic [31:0] left_idx, left_idx_next;      
  logic [31:0] right_idx, right_idx_next;     
  logic [31:0] merge_out_idx, merge_out_idx_next; 
  logic [31:0] merge_start_idx, merge_start_idx_next; 
  logic [31:0] left_end, left_end_next;     
  logic [31:0] right_end, right_end_next;  
  logic [31:0] copy_idx, copy_idx_next;
  logic [31:0] next_start_idx; // Temporary variable for calculations

  always_ff @(posedge clk) begin
    if (buffer_wen) 
      buffer[buffer_waddr] <= buffer_wdata;
      
    if (temp_wen) 
      temp[temp_waddr] <= temp_wdata;
  end

  always_ff @(posedge clk) begin
    if (reset) begin
      size <= 0;
      base_addr <= 0;
      read_idx <= 0;
      write_idx <= 0;
      merge_width <= 0;
      left_idx <= 0;
      right_idx <= 0;
      merge_out_idx <= 0;
      merge_start_idx <= 0;
      left_end <= 0;
      right_end <= 0;
      copy_idx <= 0;
    end
    else begin
      size <= size_next;
      base_addr <= base_addr_next;
      read_idx <= read_idx_next;
      write_idx <= write_idx_next;
      merge_width <= merge_width_next;
      left_idx <= left_idx_next;
      right_idx <= right_idx_next;
      merge_out_idx <= merge_out_idx_next;
      merge_start_idx <= merge_start_idx_next;
      left_end <= left_end_next;
      right_end <= right_end_next;
      copy_idx <= copy_idx_next;
    end
  end

  //======================================================================
  // State Definition
  //======================================================================

  localparam STATE_XCFG       = 5'd0; 
  localparam STATE_INIT       = 5'd1; 
  
  localparam STATE_READ_REQ   = 5'd2;  
  localparam STATE_READ_RESP  = 5'd3;  
  
  localparam STATE_SORT_INIT  = 5'd4;  
  localparam STATE_MERGE_INIT = 5'd5;  
  localparam STATE_MERGE      = 5'd6;  
  localparam STATE_MERGE_FIN  = 5'd7;  
  
  localparam STATE_COPY_INIT  = 5'd8;  
  localparam STATE_COPY       = 5'd9;  
  
  localparam STATE_WIDTH_DONE = 5'd10; 
  
  localparam STATE_WRITE_INIT = 5'd11;
  localparam STATE_WRITE_REQ  = 5'd12;
  localparam STATE_WRITE_RESP = 5'd13; 
  
  localparam STATE_DONE       = 5'd14;

  logic [4:0] state_reg;
  logic [4:0] state_reg_next;
  logic go;

  // State transition logic
  always_ff @(posedge clk) begin
    if (reset)
      state_reg <= STATE_XCFG;
    else
      state_reg <= state_reg_next;
  end

  //======================================================================
  // Combined state outputs and next state logic (only one always_comb)
  //======================================================================

  always_comb begin
    // Default values for all outputs
    xcelreq_deq_rdy     = 0;
    xcel_respstream_val = 0;
    mem_reqstream_val   = 0;
    memresp_deq_rdy     = 0;
    go                  = 0;
    
    buffer_wen          = 0;
    buffer_waddr        = 0;
    buffer_wdata        = 0;
    
    temp_wen            = 0;
    temp_waddr          = 0;
    temp_wdata          = 0;
    
    size_next           = size;
    base_addr_next      = base_addr;
    read_idx_next       = read_idx;
    write_idx_next      = write_idx;
    merge_width_next    = merge_width;
    left_idx_next       = left_idx;
    right_idx_next      = right_idx;
    merge_out_idx_next  = merge_out_idx;
    merge_start_idx_next = merge_start_idx;
    left_end_next       = left_end;
    right_end_next      = right_end;
    copy_idx_next       = copy_idx;
    next_start_idx      = 0;

    xcel_respstream_msg_raw = '0;
    mem_reqstream_msg_raw   = '0;
    
    // Default: stay in current state
    state_reg_next = state_reg;

    //--------------------------------------------------------------------
    // STATE: XCFG
    //--------------------------------------------------------------------
    if (state_reg == STATE_XCFG) begin
      xcelreq_deq_rdy     = xcel_respstream_rdy;
      xcel_respstream_val = xcelreq_deq_val;

      if (xcelreq_deq_val) begin
        if (xcelreq_deq_msg.type_ == `VC_XCEL_REQ_MSG_TYPE_READ) begin
          xcel_respstream_msg_raw.type_ = `VC_XCEL_RESP_MSG_TYPE_READ;
          xcel_respstream_msg_raw.data  = 1;
        end
        else begin
          if (xcelreq_deq_msg.addr == 0) begin
            go = 1;
            if (xcel_respstream_rdy) 
              state_reg_next = STATE_INIT;
          end
          else if (xcelreq_deq_msg.addr == 1)
            base_addr_next = xcelreq_deq_msg.data;
          else if (xcelreq_deq_msg.addr == 2)
            size_next = xcelreq_deq_msg.data;

          xcel_respstream_msg_raw.type_ = `VC_XCEL_RESP_MSG_TYPE_WRITE;
          xcel_respstream_msg_raw.data  = 0;
        end
      end
    end
    
    //--------------------------------------------------------------------
    // STATE: INIT 
    //--------------------------------------------------------------------
    else if (state_reg == STATE_INIT) begin
      read_idx_next = 0;
      write_idx_next = 0;
      state_reg_next = STATE_READ_REQ;
    end
    
    //--------------------------------------------------------------------
    // STATE: READ_REQ 
    //--------------------------------------------------------------------
    else if (state_reg == STATE_READ_REQ) begin
      if (read_idx < size) begin
        mem_reqstream_val = 1;
        
        mem_reqstream_msg_raw.type_  = `VC_MEM_REQ_MSG_TYPE_READ;
        mem_reqstream_msg_raw.opaque = 0;
        mem_reqstream_msg_raw.addr   = base_addr + (read_idx << 2);
        mem_reqstream_msg_raw.len    = 0;
        mem_reqstream_msg_raw.data   = 0;
        
        if (mem_reqstream_rdy)
          state_reg_next = STATE_READ_RESP;
      end
      else begin
        state_reg_next = STATE_SORT_INIT;
      end
    end
    
    //--------------------------------------------------------------------
    // STATE: READ_RESP 
    //--------------------------------------------------------------------
    else if (state_reg == STATE_READ_RESP) begin
      memresp_deq_rdy = 1;
      if (memresp_deq_val) begin
        buffer_wen = 1;
        buffer_waddr = read_idx[6:0]; // Ensure we mask to 6 bits
        buffer_wdata = memresp_deq_msg.data;
        read_idx_next = read_idx + 1;
        state_reg_next = STATE_READ_REQ;
      end
    end
    
    //--------------------------------------------------------------------
    // STATE: SORT_INIT 
    //--------------------------------------------------------------------
    else if (state_reg == STATE_SORT_INIT) begin
      merge_width_next = 1;
      merge_start_idx_next = 0;
      state_reg_next = STATE_MERGE_INIT;
    end
    
    //--------------------------------------------------------------------
    // STATE: MERGE_INIT
    //--------------------------------------------------------------------
    else if (state_reg == STATE_MERGE_INIT) begin
      left_idx_next = merge_start_idx;
      
      // Ensure left_end doesn't exceed array size
      if (merge_start_idx + merge_width <= size)
        left_end_next = merge_start_idx + merge_width;
      else
        left_end_next = size;
      
      right_idx_next = left_end_next;
      
      // Ensure right_end doesn't exceed array size
      if (right_idx_next + merge_width <= size)
        right_end_next = right_idx_next + merge_width;
      else
        right_end_next = size;
      
      merge_out_idx_next = merge_start_idx;
      
      // Check if current merge_start_idx is within array bounds
      if (merge_start_idx < size)
        state_reg_next = STATE_MERGE;
      else
        state_reg_next = STATE_WIDTH_DONE;
    end
    
    //--------------------------------------------------------------------
    // STATE: MERGE 
    //--------------------------------------------------------------------
    else if (state_reg == STATE_MERGE) begin
      if (left_idx < left_end && right_idx < right_end) begin
        if (buffer[left_idx[6:0]] <= buffer[right_idx[6:0]]) begin
          temp_wen = 1;
          temp_waddr = merge_out_idx[6:0];
          temp_wdata = buffer[left_idx[6:0]];
          left_idx_next = left_idx + 1;
        end else begin
          temp_wen = 1;
          temp_waddr = merge_out_idx[6:0];
          temp_wdata = buffer[right_idx[6:0]];
          right_idx_next = right_idx + 1;
        end
        merge_out_idx_next = merge_out_idx + 1;
      end
      else if (left_idx < left_end) begin
        temp_wen = 1;
        temp_waddr = merge_out_idx[6:0];
        temp_wdata = buffer[left_idx[6:0]];
        left_idx_next = left_idx + 1;
        merge_out_idx_next = merge_out_idx + 1;
      end
      else if (right_idx < right_end) begin
        temp_wen = 1;
        temp_waddr = merge_out_idx[6:0];
        temp_wdata = buffer[right_idx[6:0]];
        right_idx_next = right_idx + 1;
        merge_out_idx_next = merge_out_idx + 1;
      end
      else begin
        state_reg_next = STATE_MERGE_FIN;
      end
    end
    
    //--------------------------------------------------------------------
    // STATE: MERGE_FIN 
    //--------------------------------------------------------------------
    else if (state_reg == STATE_MERGE_FIN) begin
      // Calculate next start index using full 32-bit arithmetic
      next_start_idx = merge_start_idx + (merge_width << 1);
      merge_start_idx_next = next_start_idx;
      
      // Compare with full 32-bit value to avoid overflow
      if (next_start_idx < size)
        state_reg_next = STATE_MERGE_INIT;
      else
        state_reg_next = STATE_COPY_INIT;
    end
    
    //--------------------------------------------------------------------
    // STATE: COPY_INIT
    //--------------------------------------------------------------------
    else if (state_reg == STATE_COPY_INIT) begin
      copy_idx_next = 0;
      state_reg_next = STATE_COPY;
    end
    
    //--------------------------------------------------------------------
    // STATE: COPY
    //--------------------------------------------------------------------
    else if (state_reg == STATE_COPY) begin
      if (copy_idx < size) begin
        buffer_wen = 1;
        buffer_waddr = copy_idx[6:0]; // Ensure we mask to 6 bits
        buffer_wdata = temp[copy_idx[6:0]];
        copy_idx_next = copy_idx + 1;
        
        // Check if this will be the last element
        if (copy_idx == 7'd127 || copy_idx + 1 >= size)
          state_reg_next = STATE_WIDTH_DONE;
      end
      else begin
        state_reg_next = STATE_WIDTH_DONE;
      end
    end
    
    //--------------------------------------------------------------------
    // STATE: WIDTH_DONE 
    //--------------------------------------------------------------------
    else if (state_reg == STATE_WIDTH_DONE) begin
      // Double the merge width for next pass
      merge_width_next = merge_width << 1;
      // Reset merge start index to begin next pass
      merge_start_idx_next = 0;
      
      // Check if merge width exceeds array size
      if (merge_width_next >= size)
        state_reg_next = STATE_WRITE_INIT;
      else
        state_reg_next = STATE_MERGE_INIT;
    end
    
    //--------------------------------------------------------------------
    // STATE: WRITE_INIT
    //--------------------------------------------------------------------
    else if (state_reg == STATE_WRITE_INIT) begin
      write_idx_next = 0;
      state_reg_next = STATE_WRITE_REQ;
    end
    
    //--------------------------------------------------------------------
    // STATE: WRITE_REQ
    //--------------------------------------------------------------------
    else if (state_reg == STATE_WRITE_REQ) begin
      if (write_idx < size) begin
        mem_reqstream_val = 1;
        
        mem_reqstream_msg_raw.type_  = `VC_MEM_REQ_MSG_TYPE_WRITE;
        mem_reqstream_msg_raw.opaque = 0;
        mem_reqstream_msg_raw.addr   = base_addr + (write_idx << 2);
        mem_reqstream_msg_raw.len    = 0;
        mem_reqstream_msg_raw.data   = buffer[write_idx[6:0]];
        
        if (mem_reqstream_rdy)
          state_reg_next = STATE_WRITE_RESP;
      end
      else begin
        state_reg_next = STATE_DONE;
      end
    end
    
    //--------------------------------------------------------------------
    // STATE: WRITE_RESP
    //--------------------------------------------------------------------
    else if (state_reg == STATE_WRITE_RESP) begin
      memresp_deq_rdy = 1;
      if (memresp_deq_val) begin
        write_idx_next = write_idx + 1;
        state_reg_next = STATE_WRITE_REQ;
      end
    end
    
    //--------------------------------------------------------------------
    // STATE: DONE
    //--------------------------------------------------------------------
    else if (state_reg == STATE_DONE) begin
      read_idx_next = 0;
      write_idx_next = 0;
      merge_width_next = 0;
      state_reg_next = STATE_XCFG;
    end
  end

  //----------------------------------------------------------------------
  // Line Tracing
  //----------------------------------------------------------------------

  `ifndef SYNTHESIS

  vc_XcelReqMsgTrace xcel_reqstream_msg_trace
  (
    .clk (clk),
    .reset (reset),
    .val   (xcel_reqstream_val),
    .rdy   (xcel_reqstream_rdy),
    .msg   (xcel_reqstream_msg)
  );

  vc_XcelRespMsgTrace xcel_respstream_msg_trace
  (
    .clk (clk),
    .reset (reset),
    .val   (xcel_respstream_val),
    .rdy   (xcel_respstream_rdy),
    .msg   (xcel_respstream_msg)
  );

  logic [`VC_TRACE_NBITS-1:0] str;
  `VC_TRACE_BEGIN
  begin
    xcel_reqstream_msg_trace.line_trace( trace_str );

    vc_trace.append_str( trace_str, "(" );

    // Line trace for debugging
    vc_trace.append_str( trace_str, " " );
    case (state_reg)
      STATE_XCFG:       vc_trace.append_str( trace_str, "XCFG       " );
      STATE_INIT:       vc_trace.append_str( trace_str, "INIT       " );
      STATE_READ_REQ:   vc_trace.append_str( trace_str, "READ_REQ   " );
      STATE_READ_RESP:  vc_trace.append_str( trace_str, "READ_RESP  " );
      STATE_SORT_INIT:  vc_trace.append_str( trace_str, "SORT_INIT  " );
      STATE_MERGE_INIT: vc_trace.append_str( trace_str, "MERGE_INIT " );
      STATE_MERGE:      vc_trace.append_str( trace_str, "MERGE      " );
      STATE_MERGE_FIN:  vc_trace.append_str( trace_str, "MERGE_FIN  " );
      STATE_COPY_INIT:  vc_trace.append_str( trace_str, "COPY_INIT  " );
      STATE_COPY:       vc_trace.append_str( trace_str, "COPY       " );
      STATE_WIDTH_DONE: vc_trace.append_str( trace_str, "WIDTH_DONE " );
      STATE_WRITE_INIT: vc_trace.append_str( trace_str, "WRITE_INIT " );
      STATE_WRITE_REQ:  vc_trace.append_str( trace_str, "WRITE_REQ  " );
      STATE_WRITE_RESP: vc_trace.append_str( trace_str, "WRITE_RESP " );
      STATE_DONE:       vc_trace.append_str( trace_str, "DONE       " );
      default:          vc_trace.append_str( trace_str, "?          " );
    endcase

    // Print key indices and values for debugging
    $sformat( str, "w:%d r/w:%d/%d li:%d ri:%d ms:%d cpy:%d", 
              merge_width, read_idx, write_idx, left_idx, right_idx, 
              merge_start_idx, copy_idx);
    vc_trace.append_str( trace_str, str );

    vc_trace.append_str( trace_str, ")" );

    xcel_respstream_msg_trace.line_trace( trace_str );
  end
  `VC_TRACE_END

  `endif /* SYNTHESIS */

endmodule

`endif /* LAB2_XCEL_SORT_XCEL_V */