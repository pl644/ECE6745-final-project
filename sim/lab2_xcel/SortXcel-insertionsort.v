//=========================================================================
// Sorting Accelerator Implementation
//=========================================================================
// Sort array in memory containing positive integers.
// Accelerator register interface:
//
//  xr0 : go/done
//  xr1 : base address of array
//  xr2 : number of elements in array
//
// Accelerator protocol involves the following steps:
//  1. Write the base address of array via xr1
//  2. Write the number of elements in array via xr2
//  3. Tell accelerator to go by writing xr0
//  4. Wait for accelerator to finish by reading xr0, result will be 1
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

  // Extra state registers

  logic [31:0] i, i_next;
  logic [31:0] j, j_next;
  logic [31:0] size, size_next;
  logic [31:0] base_addr, base_addr_next;
  logic [31:0] temp_val, temp_val_next;
  logic [31:0] array_val_i, array_val_i_next;
  logic [31:0] array_val_j, array_val_j_next;
  
  // Insertion sort specific states
  logic [31:0] key, key_next;
  logic [31:0] pos, pos_next;

  always_ff @(posedge clk) begin
    if (reset) begin
      i          <= 0;
      j          <= 0;
      size       <= 0;
      base_addr  <= 0;
      temp_val   <= 0;
      array_val_i <= 0;
      array_val_j <= 0;
      key        <= 0;
      pos        <= 0;
    end
    else begin
      i          <= i_next;
      j          <= j_next;
      size       <= size_next;
      base_addr  <= base_addr_next;
      temp_val   <= temp_val_next;
      array_val_i <= array_val_i_next;
      array_val_j <= array_val_j_next;
      key        <= key_next;
      pos        <= pos_next;
    end
  end

  //======================================================================
  // State Update
  //======================================================================

  localparam STATE_XCFG        = 4'd0;  // Wait for configuration
  localparam STATE_OUTER_LOOP  = 4'd1;  // Outer loop of insertion sort
  localparam STATE_READ_KEY    = 4'd2;  // Read current key element
  localparam STATE_WAIT_KEY    = 4'd3;  // Wait for key read response
  localparam STATE_INNER_INIT  = 4'd4;  // Initialize inner loop
  localparam STATE_INNER_LOOP  = 4'd5;  // Inner loop of insertion sort
  localparam STATE_READ_CMP    = 4'd6;  // Read element for comparison
  localparam STATE_WAIT_CMP    = 4'd7;  // Wait for comparison read response
  localparam STATE_COMPARE     = 4'd8;  // Compare key with current element
  localparam STATE_SHIFT       = 4'd9;  // Shift element right
  localparam STATE_WAIT_SHIFT  = 4'd10; // Wait for shift write response
  localparam STATE_INSERT_KEY  = 4'd11; // Insert key at final position
  localparam STATE_WAIT_INSERT = 4'd12; // Wait for key insertion response
  localparam STATE_DONE        = 4'd13; // Done sorting

  logic [3:0] state_reg;
  logic go;

  always_ff @(posedge clk) begin
    if (reset)
      state_reg <= STATE_XCFG;
    else begin
      state_reg <= state_reg;

      case (state_reg)
        STATE_XCFG:
          if (go & xcel_respstream_rdy)
            state_reg <= STATE_OUTER_LOOP;

        STATE_OUTER_LOOP:
          if (i < size)
            state_reg <= STATE_READ_KEY;
          else
            state_reg <= STATE_DONE;

        STATE_READ_KEY:
          if (mem_reqstream_rdy)
            state_reg <= STATE_WAIT_KEY;

        STATE_WAIT_KEY:
          if (memresp_deq_val)
            state_reg <= STATE_INNER_INIT;

        STATE_INNER_INIT:
          state_reg <= STATE_INNER_LOOP;

        STATE_INNER_LOOP:
          if (j >= 0 && j < size) // Safety check for j
            state_reg <= STATE_READ_CMP;
          else
            state_reg <= STATE_INSERT_KEY;

        STATE_READ_CMP:
          if (mem_reqstream_rdy)
            state_reg <= STATE_WAIT_CMP;

        STATE_WAIT_CMP:
          if (memresp_deq_val)
            state_reg <= STATE_COMPARE;

        STATE_COMPARE:
          if (array_val_j > key)
            state_reg <= STATE_SHIFT;
          else
            state_reg <= STATE_INSERT_KEY;

        STATE_SHIFT:
          if (mem_reqstream_rdy)
            state_reg <= STATE_WAIT_SHIFT;

        STATE_WAIT_SHIFT:
          if (memresp_deq_val)
            state_reg <= STATE_INNER_LOOP;

        STATE_INSERT_KEY:
          if (mem_reqstream_rdy)
            state_reg <= STATE_WAIT_INSERT;

        STATE_WAIT_INSERT:
          if (memresp_deq_val)
            state_reg <= STATE_OUTER_LOOP;

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
    xcelreq_deq_rdy     = 0;
    xcel_respstream_val = 0;
    mem_reqstream_val   = 0;
    memresp_deq_rdy     = 0;
    go                  = 0;

    base_addr_next      = base_addr;
    size_next           = size;
    i_next              = i;
    j_next              = j;
    temp_val_next       = temp_val;
    array_val_i_next    = array_val_i;
    array_val_j_next    = array_val_j;
    key_next            = key;
    pos_next            = pos;

    xcel_respstream_msg_raw = '0;
    mem_reqstream_msg_raw   = '0;

    //--------------------------------------------------------------------
    // STATE: XCFG
    //--------------------------------------------------------------------
    // In this state we handle the accelerator configuration protocol

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
            go        = 1;
            i_next    = 1; // Start from second element for insertion sort
            j_next    = 0;
            pos_next  = 0;
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
    // STATE: OUTER_LOOP
    //--------------------------------------------------------------------
    // Outer loop of insertion sort

    else if (state_reg == STATE_OUTER_LOOP) begin
      // If we're done, reset indices
      if (i >= size) begin
        i_next = 0;
        j_next = 0;
      end
    end

    //--------------------------------------------------------------------
    // STATE: READ_KEY
    //--------------------------------------------------------------------
    // Read current key element (the element to be inserted)

    else if (state_reg == STATE_READ_KEY) begin
      mem_reqstream_val = 1;
      
      mem_reqstream_msg_raw.type_  = `VC_MEM_REQ_MSG_TYPE_READ;
      mem_reqstream_msg_raw.opaque = 0;
      mem_reqstream_msg_raw.addr   = base_addr + (i << 2);
      mem_reqstream_msg_raw.len    = 0;
      mem_reqstream_msg_raw.data   = 0;
    end

    //--------------------------------------------------------------------
    // STATE: WAIT_KEY
    //--------------------------------------------------------------------
    // Wait for key read response

    else if (state_reg == STATE_WAIT_KEY) begin
      memresp_deq_rdy = 1;
      if (memresp_deq_val) begin
        key_next = memresp_deq_msg.data;
      end
    end

    //--------------------------------------------------------------------
    // STATE: INNER_INIT
    //--------------------------------------------------------------------
    // Initialize inner loop

    else if (state_reg == STATE_INNER_INIT) begin
      j_next = i - 1;
      pos_next = i; // Position where key will eventually be inserted
    end

    //--------------------------------------------------------------------
    // STATE: INNER_LOOP
    //--------------------------------------------------------------------
    // Inner loop of insertion sort

    else if (state_reg == STATE_INNER_LOOP) begin
      if (j < 0) begin
        // If j is negative, we've moved past the beginning of the array
        // so it's time to insert the key
        pos_next = 0;
      end
    end

    //--------------------------------------------------------------------
    // STATE: READ_CMP
    //--------------------------------------------------------------------
    // Read element for comparison

    else if (state_reg == STATE_READ_CMP) begin
      mem_reqstream_val = 1;
      
      mem_reqstream_msg_raw.type_  = `VC_MEM_REQ_MSG_TYPE_READ;
      mem_reqstream_msg_raw.opaque = 0;
      mem_reqstream_msg_raw.addr   = base_addr + (j << 2);
      mem_reqstream_msg_raw.len    = 0;
      mem_reqstream_msg_raw.data   = 0;
    end

    //--------------------------------------------------------------------
    // STATE: WAIT_CMP
    //--------------------------------------------------------------------
    // Wait for comparison read response

    else if (state_reg == STATE_WAIT_CMP) begin
      memresp_deq_rdy = 1;
      if (memresp_deq_val) begin
        array_val_j_next = memresp_deq_msg.data;
      end
    end

    //--------------------------------------------------------------------
    // STATE: COMPARE
    //--------------------------------------------------------------------
    // Compare key with current element

    else if (state_reg == STATE_COMPARE) begin
      if (array_val_j > key) begin
        // Need to shift this element to the right
        pos_next = j; // Update position for key insertion
      end
    end

    //--------------------------------------------------------------------
    // STATE: SHIFT
    //--------------------------------------------------------------------
    // Shift element right

    else if (state_reg == STATE_SHIFT) begin
      mem_reqstream_val = 1;
      
      mem_reqstream_msg_raw.type_  = `VC_MEM_REQ_MSG_TYPE_WRITE;
      mem_reqstream_msg_raw.opaque = 0;
      mem_reqstream_msg_raw.addr   = base_addr + ((j + 1) << 2);
      mem_reqstream_msg_raw.len    = 0;
      mem_reqstream_msg_raw.data   = array_val_j;
    end

    //--------------------------------------------------------------------
    // STATE: WAIT_SHIFT
    //--------------------------------------------------------------------
    // Wait for shift write response

    else if (state_reg == STATE_WAIT_SHIFT) begin
      memresp_deq_rdy = 1;
      if (memresp_deq_val) begin
        j_next = j - 1; // Move to next element
      end
    end

    //--------------------------------------------------------------------
    // STATE: INSERT_KEY
    //--------------------------------------------------------------------
    // Insert key at final position

    else if (state_reg == STATE_INSERT_KEY) begin
      mem_reqstream_val = 1;
      
      mem_reqstream_msg_raw.type_  = `VC_MEM_REQ_MSG_TYPE_WRITE;
      mem_reqstream_msg_raw.opaque = 0;
      mem_reqstream_msg_raw.addr   = base_addr + (pos << 2);
      mem_reqstream_msg_raw.len    = 0;
      mem_reqstream_msg_raw.data   = key;
    end

    //--------------------------------------------------------------------
    // STATE: WAIT_INSERT
    //--------------------------------------------------------------------
    // Wait for key insertion response

    else if (state_reg == STATE_WAIT_INSERT) begin
      memresp_deq_rdy = 1;
      if (memresp_deq_val) begin
        i_next = i + 1; // Move to next element in outer loop
      end
    end

    //--------------------------------------------------------------------
    // STATE: DONE
    //--------------------------------------------------------------------
    // Done with sorting

    else if (state_reg == STATE_DONE) begin
      // No specific action needed here, will transition to XCFG automatically
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
      STATE_XCFG:        vc_trace.append_str( trace_str, "XCFG    " );
      STATE_OUTER_LOOP:  vc_trace.append_str( trace_str, "OUTER   " );
      STATE_READ_KEY:    vc_trace.append_str( trace_str, "RD_KEY  " );
      STATE_WAIT_KEY:    vc_trace.append_str( trace_str, "W_KEY   " );
      STATE_INNER_INIT:  vc_trace.append_str( trace_str, "IN_INIT " );
      STATE_INNER_LOOP:  vc_trace.append_str( trace_str, "INNER   " );
      STATE_READ_CMP:    vc_trace.append_str( trace_str, "RD_CMP  " );
      STATE_WAIT_CMP:    vc_trace.append_str( trace_str, "W_CMP   " );
      STATE_COMPARE:     vc_trace.append_str( trace_str, "COMPARE " );
      STATE_SHIFT:       vc_trace.append_str( trace_str, "SHIFT   " );
      STATE_WAIT_SHIFT:  vc_trace.append_str( trace_str, "W_SHIFT " );
      STATE_INSERT_KEY:  vc_trace.append_str( trace_str, "INSERT  " );
      STATE_WAIT_INSERT: vc_trace.append_str( trace_str, "W_INSERT" );
      STATE_DONE:        vc_trace.append_str( trace_str, "DONE    " );
      default:           vc_trace.append_str( trace_str, "?       " );
    endcase

    // Print indices and values for debugging
    $sformat( str, "i:%2d j:%2d pos:%2d ", i, j, pos );
    vc_trace.append_str( trace_str, str );

    $sformat( str, "key:%x a[j]:%x", key, array_val_j );
    vc_trace.append_str( trace_str, str );

    vc_trace.append_str( trace_str, ")" );

    xcel_respstream_msg_trace.line_trace( trace_str );
  end
  `VC_TRACE_END

  `endif /* SYNTHESIS */

endmodule

`endif /* LAB2_XCEL_SORT_XCEL_V */