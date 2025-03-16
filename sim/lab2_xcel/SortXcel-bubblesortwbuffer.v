//=========================================================================
// Sorting Accelerator Implementation (Bubble Sort)
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
  logic [31:0] buffer [0:63];
  
  // Buffer write control signal
  logic buffer_wen;
  logic [5:0] buffer_waddr;
  logic [31:0] buffer_wdata;
  
  // Sorting variables
  logic [31:0] read_idx, read_idx_next;    // Index for reading from memory
  logic [31:0] write_idx, write_idx_next;  // Index for writing back to memory
  logic [31:0] i, i_next;                  // Outer loop index
  logic [31:0] j, j_next;                  // Inner loop index
  logic [31:0] temp, temp_next;            // Temporary value for swapping
  logic swap_occurred, swap_occurred_next;  // Flag to detect if swaps occurred

  // Buffer sequential write logic
  always_ff @(posedge clk) begin
    if (buffer_wen && buffer_waddr < 64)
      buffer[buffer_waddr] <= buffer_wdata;
  end

  // Register updates
  always_ff @(posedge clk) begin
    if (reset) begin
      size <= 0;
      base_addr <= 0;
      read_idx <= 0;
      write_idx <= 0;
      i <= 0;
      j <= 0;
      temp <= 0;
      swap_occurred <= 0;
    end
    else begin
      size <= size_next;
      base_addr <= base_addr_next;
      read_idx <= read_idx_next;
      write_idx <= write_idx_next;
      i <= i_next;
      j <= j_next;
      temp <= temp_next;
      swap_occurred <= swap_occurred_next;
    end
  end

  //======================================================================
  // State Definition
  //======================================================================

  localparam STATE_XCFG       = 4'd0;  // Configuration handling
  localparam STATE_INIT       = 4'd1;  // Initialization
  
  // Data reading states
  localparam STATE_READ_REQ   = 4'd2;  // Request data from memory
  localparam STATE_READ_RESP  = 4'd3;  // Process read response
  
  // Bubble sort states
  localparam STATE_SORT_INIT  = 4'd4;  // Initialize sorting
  localparam STATE_OUTER      = 4'd5;  // Outer loop of bubble sort
  localparam STATE_INNER      = 4'd6;  // Inner loop of bubble sort
  localparam STATE_COMPARE    = 4'd7;  // Compare and swap if needed
  
  // Data writing states
  localparam STATE_WRITE_INIT = 4'd8;  // Initialize write operation
  localparam STATE_WRITE_REQ  = 4'd9;  // Request write to memory
  localparam STATE_WRITE_RESP = 4'd10; // Process write response
  
  localparam STATE_DONE       = 4'd11; // All operations complete

  logic [3:0] state_reg;
  logic go;

  // State transition logic
  always_ff @(posedge clk) begin
    if (reset)
      state_reg <= STATE_XCFG;
    else begin
      case (state_reg)
        STATE_XCFG:
          if (go & xcel_respstream_rdy)
            state_reg <= STATE_INIT;
          
        STATE_INIT:
          state_reg <= STATE_READ_REQ;
          
        STATE_READ_REQ:
          if (read_idx < size && mem_reqstream_rdy)
            state_reg <= STATE_READ_RESP;
          else if (read_idx >= size)
            state_reg <= STATE_SORT_INIT;
            
        STATE_READ_RESP:
          if (memresp_deq_val)
            state_reg <= STATE_READ_REQ;
        
        STATE_SORT_INIT:
          state_reg <= STATE_OUTER;
          
        STATE_OUTER:
          if (i < size - 1)
            state_reg <= STATE_INNER;
          else
            state_reg <= STATE_WRITE_INIT;
            
        STATE_INNER:
          if (j < size - i - 1)
            state_reg <= STATE_COMPARE;
          else if (!swap_occurred) // Early termination if no swaps occurred this pass
            state_reg <= STATE_WRITE_INIT;
          else
            state_reg <= STATE_OUTER;
            
        STATE_COMPARE: begin
          state_reg <= STATE_INNER;
        end
            
        STATE_WRITE_INIT:
          state_reg <= STATE_WRITE_REQ;
            
        STATE_WRITE_REQ:
          if (write_idx < size && mem_reqstream_rdy)
            state_reg <= STATE_WRITE_RESP;
          else if (write_idx >= size)
            state_reg <= STATE_DONE;
            
        STATE_WRITE_RESP:
          if (memresp_deq_val)
            state_reg <= STATE_WRITE_REQ;
            
        STATE_DONE:
          state_reg <= STATE_XCFG;
          
        default:
          state_reg <= STATE_XCFG;
      endcase
    end
  end

  //======================================================================
  // State Outputs
  //======================================================================

  always_comb begin
    // Default values
    xcelreq_deq_rdy     = 0;
    xcel_respstream_val = 0;
    mem_reqstream_val   = 0;
    memresp_deq_rdy     = 0;
    go                  = 0;
    
    // Default buffer control signals
    buffer_wen          = 0;
    buffer_waddr        = 0;
    buffer_wdata        = 0;
    
    // Keep register values by default
    size_next           = size;
    base_addr_next      = base_addr;
    read_idx_next       = read_idx;
    write_idx_next      = write_idx;
    i_next              = i;
    j_next              = j;
    temp_next           = temp;
    swap_occurred_next  = swap_occurred;

    xcel_respstream_msg_raw = '0;
    mem_reqstream_msg_raw   = '0;

    //--------------------------------------------------------------------
    // STATE: XCFG - Configuration handling
    //--------------------------------------------------------------------
    if (state_reg == STATE_XCFG) begin
      xcelreq_deq_rdy     = xcel_respstream_rdy;
      xcel_respstream_val = xcelreq_deq_val;

      if (xcelreq_deq_val) begin
        if (xcelreq_deq_msg.type_ == `VC_XCEL_REQ_MSG_TYPE_READ) begin
          xcel_respstream_msg_raw.type_ = `VC_XCEL_RESP_MSG_TYPE_READ;
          xcel_respstream_msg_raw.data  = 1; // Return 1 when done
        end
        else begin
          if (xcelreq_deq_msg.addr == 0) begin
            go = 1;
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
    // STATE: INIT - Initialization
    //--------------------------------------------------------------------
    else if (state_reg == STATE_INIT) begin
      read_idx_next = 0;
      write_idx_next = 0;
      i_next = 0;
      j_next = 0;
    end
    
    //--------------------------------------------------------------------
    // STATE: READ_REQ - Request data from memory
    //--------------------------------------------------------------------
    else if (state_reg == STATE_READ_REQ) begin
      if (read_idx < size) begin
        mem_reqstream_val = 1;
        
        mem_reqstream_msg_raw.type_  = `VC_MEM_REQ_MSG_TYPE_READ;
        mem_reqstream_msg_raw.opaque = 0;
        mem_reqstream_msg_raw.addr   = base_addr + (read_idx << 2);
        mem_reqstream_msg_raw.len    = 0;
        mem_reqstream_msg_raw.data   = 0;
      end
    end
    
    //--------------------------------------------------------------------
    // STATE: READ_RESP - Process read response
    //--------------------------------------------------------------------
    else if (state_reg == STATE_READ_RESP) begin
      memresp_deq_rdy = 1;
      if (memresp_deq_val) begin
        // Store received data in buffer
        buffer_wen = 1;
        buffer_waddr = read_idx[5:0]; // Only use lower 6 bits (buffer size 64)
        buffer_wdata = memresp_deq_msg.data;
        read_idx_next = read_idx + 1;
      end
    end
    
    //--------------------------------------------------------------------
    // STATE: SORT_INIT - Initialize sorting
    //--------------------------------------------------------------------
    else if (state_reg == STATE_SORT_INIT) begin
      // Initialize bubble sort indices
      i_next = 0;
      j_next = 0;
      swap_occurred_next = 0;
    end
    
    //--------------------------------------------------------------------
    // STATE: OUTER - Outer loop of bubble sort
    //--------------------------------------------------------------------
    else if (state_reg == STATE_OUTER) begin
      if (i < size - 1) begin
        j_next = 0;
        swap_occurred_next = 0; // Reset swap flag for new pass
      end
    end
    
    //--------------------------------------------------------------------
    // STATE: INNER - Inner loop of bubble sort
    //--------------------------------------------------------------------
    else if (state_reg == STATE_INNER) begin
      // Complete the second half of the swap if temp has a value (meaning we performed a swap)
      if (temp != 0) begin
        buffer_wen = 1;
        buffer_waddr = j; // Current j was incremented from previous j+1
        buffer_wdata = temp;
        temp_next = 0; // Reset temp
      end
      
      if (j < size - i - 1) begin
        // Continue with inner loop
      end
      else begin
        i_next = i + 1;
      end
    end
    
    //--------------------------------------------------------------------
    // STATE: COMPARE - Compare and swap if needed
    //--------------------------------------------------------------------
    else if (state_reg == STATE_COMPARE) begin
      // Compare buffer[j] and buffer[j+1]
      if (buffer[j+1] < buffer[j]) begin
        // Swap elements
        temp_next = buffer[j];
        
        // First part of swap in buffer - write buffer[j+1] to buffer[j]
        buffer_wen = 1;
        buffer_waddr = j;
        buffer_wdata = buffer[j+1];
        
        // Set swap flag
        swap_occurred_next = 1;
      end
      j_next = j + 1;
    end
    
    //--------------------------------------------------------------------
    // STATE: WRITE_INIT - Initialize write operation
    //--------------------------------------------------------------------
    else if (state_reg == STATE_WRITE_INIT) begin
      write_idx_next = 0;
    end
    
    //--------------------------------------------------------------------
    // STATE: WRITE_REQ - Request write to memory
    //--------------------------------------------------------------------
    else if (state_reg == STATE_WRITE_REQ) begin
      if (write_idx < size) begin
        mem_reqstream_val = 1;
        
        mem_reqstream_msg_raw.type_  = `VC_MEM_REQ_MSG_TYPE_WRITE;
        mem_reqstream_msg_raw.opaque = 0;
        mem_reqstream_msg_raw.addr   = base_addr + (write_idx << 2);
        mem_reqstream_msg_raw.len    = 0;
        mem_reqstream_msg_raw.data   = buffer[write_idx[5:0]]; // Only use lower 6 bits
      end
    end
    
    //--------------------------------------------------------------------
    // STATE: WRITE_RESP - Process write response
    //--------------------------------------------------------------------
    else if (state_reg == STATE_WRITE_RESP) begin
      memresp_deq_rdy = 1;
      if (memresp_deq_val) begin
        write_idx_next = write_idx + 1;
      end
    end
    
    //--------------------------------------------------------------------
    // STATE: DONE - All operations complete
    //--------------------------------------------------------------------
    else if (state_reg == STATE_DONE) begin
      // Reset state for next sort
      read_idx_next = 0;
      write_idx_next = 0;
      i_next = 0;
      j_next = 0;
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
      STATE_OUTER:      vc_trace.append_str( trace_str, "OUTER      " );
      STATE_INNER:      vc_trace.append_str( trace_str, "INNER      " );
      STATE_COMPARE:    vc_trace.append_str( trace_str, "COMPARE    " );
      STATE_WRITE_INIT: vc_trace.append_str( trace_str, "WRITE_INIT " );
      STATE_WRITE_REQ:  vc_trace.append_str( trace_str, "WRITE_REQ  " );
      STATE_WRITE_RESP: vc_trace.append_str( trace_str, "WRITE_RESP " );
      STATE_DONE:       vc_trace.append_str( trace_str, "DONE       " );
      default:          vc_trace.append_str( trace_str, "?          " );
    endcase

    // Print key indices and values for debugging
    $sformat( str, "r/w:%d/%d i:%d j:%d", 
              read_idx, write_idx, i, j);
    vc_trace.append_str( trace_str, str );

    vc_trace.append_str( trace_str, ")" );

    xcel_respstream_msg_trace.line_trace( trace_str );
  end
  `VC_TRACE_END

  `endif /* SYNTHESIS */

endmodule

`endif /* LAB2_XCEL_SORT_XCEL_V */