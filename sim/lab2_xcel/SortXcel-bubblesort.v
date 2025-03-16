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

  always_ff @(posedge clk) begin
    if (reset) begin
      i          <= 0;
      j          <= 0;
      size       <= 0;
      base_addr  <= 0;
      temp_val   <= 0;
      array_val_i <= 0;
      array_val_j <= 0;
    end
    else begin
      i          <= i_next;
      j          <= j_next;
      size       <= size_next;
      base_addr  <= base_addr_next;
      temp_val   <= temp_val_next;
      array_val_i <= array_val_i_next;
      array_val_j <= array_val_j_next;
    end
  end

  //======================================================================
  // State Update
  //======================================================================

  localparam STATE_XCFG       = 4'd0;  // Wait for configuration
  localparam STATE_OUTER      = 4'd1;  // Outer loop of bubble sort
  localparam STATE_INNER      = 4'd2;  // Inner loop of bubble sort
  localparam STATE_RD_I       = 4'd3;  // Read array[j]
  localparam STATE_WAIT_RD_I  = 4'd4;  // Wait for read array[j] response
  localparam STATE_RD_J       = 4'd5;  // Read array[j+1]
  localparam STATE_WAIT_RD_J  = 4'd6;  // Wait for read array[j+1] response
  localparam STATE_CMP        = 4'd7;  // Compare array[j] and array[j+1]
  localparam STATE_WR_I       = 4'd8;  // Write to array[j]
  localparam STATE_WAIT_WR_I  = 4'd9;  // Wait for write to array[j] response
  localparam STATE_WR_J       = 4'd10; // Write to array[j+1]
  localparam STATE_WAIT_WR_J  = 4'd11; // Wait for write to array[j+1] response
  localparam STATE_DONE       = 4'd12; // Done sorting

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
            state_reg <= STATE_OUTER;

        STATE_OUTER:
          if (i < size - 1)
            state_reg <= STATE_INNER;
          else
            state_reg <= STATE_DONE;

        STATE_INNER:
          if (j < size - i - 1)
            state_reg <= STATE_RD_I;
          else
            state_reg <= STATE_OUTER;

        STATE_RD_I:
          if (mem_reqstream_rdy)
            state_reg <= STATE_WAIT_RD_I;

        STATE_WAIT_RD_I:
          if (memresp_deq_val)
            state_reg <= STATE_RD_J;

        STATE_RD_J:
          if (mem_reqstream_rdy)
            state_reg <= STATE_WAIT_RD_J;

        STATE_WAIT_RD_J:
          if (memresp_deq_val)
            state_reg <= STATE_CMP;

        STATE_CMP:
          if (array_val_j < array_val_i)
            state_reg <= STATE_WR_I;
          else
            state_reg <= STATE_INNER;

        STATE_WR_I:
          if (mem_reqstream_rdy)
            state_reg <= STATE_WAIT_WR_I;

        STATE_WAIT_WR_I:
          if (memresp_deq_val)
            state_reg <= STATE_WR_J;

        STATE_WR_J:
          if (mem_reqstream_rdy)
            state_reg <= STATE_WAIT_WR_J;

        STATE_WAIT_WR_J:
          if (memresp_deq_val)
            state_reg <= STATE_INNER;

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
            i_next    = 0;
            j_next    = 0;
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
    // STATE: OUTER
    //--------------------------------------------------------------------
    // Outer loop of bubble sort

    else if (state_reg == STATE_OUTER) begin
      if (i < size - 1) begin
        j_next = 0;
      end
      else begin
        i_next = 0;
        j_next = 0;
      end
    end

    //--------------------------------------------------------------------
    // STATE: INNER
    //--------------------------------------------------------------------
    // Inner loop of bubble sort

    else if (state_reg == STATE_INNER) begin
      if (j < size - i - 1) begin
        // Continue with inner loop
      end
      else begin
        i_next = i + 1;
      end
    end

    //--------------------------------------------------------------------
    // STATE: RD_I
    //--------------------------------------------------------------------
    // Read array[j]

    else if (state_reg == STATE_RD_I) begin
      mem_reqstream_val = 1;
      
      mem_reqstream_msg_raw.type_  = `VC_MEM_REQ_MSG_TYPE_READ;
      mem_reqstream_msg_raw.opaque = 0;
      mem_reqstream_msg_raw.addr   = base_addr + (j << 2);
      mem_reqstream_msg_raw.len    = 0;
      mem_reqstream_msg_raw.data   = 0;
    end

    //--------------------------------------------------------------------
    // STATE: WAIT_RD_I
    //--------------------------------------------------------------------
    // Wait for memory response for array[j]

    else if (state_reg == STATE_WAIT_RD_I) begin
      memresp_deq_rdy = 1;
      if (memresp_deq_val) begin
        array_val_i_next = memresp_deq_msg.data;
      end
    end

    //--------------------------------------------------------------------
    // STATE: RD_J
    //--------------------------------------------------------------------
    // Read array[j+1]

    else if (state_reg == STATE_RD_J) begin
      mem_reqstream_val = 1;
      
      mem_reqstream_msg_raw.type_  = `VC_MEM_REQ_MSG_TYPE_READ;
      mem_reqstream_msg_raw.opaque = 0;
      mem_reqstream_msg_raw.addr   = base_addr + ((j + 1) << 2);
      mem_reqstream_msg_raw.len    = 0;
      mem_reqstream_msg_raw.data   = 0;
    end

    //--------------------------------------------------------------------
    // STATE: WAIT_RD_J
    //--------------------------------------------------------------------
    // Wait for memory response for array[j+1]

    else if (state_reg == STATE_WAIT_RD_J) begin
      memresp_deq_rdy = 1;
      if (memresp_deq_val) begin
        array_val_j_next = memresp_deq_msg.data;
      end
    end

    //--------------------------------------------------------------------
    // STATE: CMP
    //--------------------------------------------------------------------
    // Compare array[j] and array[j+1]

    else if (state_reg == STATE_CMP) begin
      // If array[j+1] < array[j], swap them
      if (array_val_j < array_val_i) begin
        temp_val_next = array_val_i;
      end
      else begin
        j_next = j + 1;
      end
    end

    //--------------------------------------------------------------------
    // STATE: WR_I
    //--------------------------------------------------------------------
    // Write array[j] = array[j+1]

    else if (state_reg == STATE_WR_I) begin
      mem_reqstream_val = 1;
      
      mem_reqstream_msg_raw.type_  = `VC_MEM_REQ_MSG_TYPE_WRITE;
      mem_reqstream_msg_raw.opaque = 0;
      mem_reqstream_msg_raw.addr   = base_addr + (j << 2);
      mem_reqstream_msg_raw.len    = 0;
      mem_reqstream_msg_raw.data   = array_val_j;
    end

    //--------------------------------------------------------------------
    // STATE: WAIT_WR_I
    //--------------------------------------------------------------------
    // Wait for memory response for writing to array[j]

    else if (state_reg == STATE_WAIT_WR_I) begin
      memresp_deq_rdy = 1;
    end

    //--------------------------------------------------------------------
    // STATE: WR_J
    //--------------------------------------------------------------------
    // Write array[j+1] = temp (original array[j])

    else if (state_reg == STATE_WR_J) begin
      mem_reqstream_val = 1;
      
      mem_reqstream_msg_raw.type_  = `VC_MEM_REQ_MSG_TYPE_WRITE;
      mem_reqstream_msg_raw.opaque = 0;
      mem_reqstream_msg_raw.addr   = base_addr + ((j + 1) << 2);
      mem_reqstream_msg_raw.len    = 0;
      mem_reqstream_msg_raw.data   = temp_val;
    end

    //--------------------------------------------------------------------
    // STATE: WAIT_WR_J
    //--------------------------------------------------------------------
    // Wait for memory response for writing to array[j+1]

    else if (state_reg == STATE_WAIT_WR_J) begin
      memresp_deq_rdy = 1;
      if (memresp_deq_val) begin
        j_next = j + 1;
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
      STATE_XCFG:      vc_trace.append_str( trace_str, "X    " );
      STATE_OUTER:     vc_trace.append_str( trace_str, "OUT  " );
      STATE_INNER:     vc_trace.append_str( trace_str, "IN   " );
      STATE_RD_I:      vc_trace.append_str( trace_str, "RD_I " );
      STATE_WAIT_RD_I: vc_trace.append_str( trace_str, "W_RDI" );
      STATE_RD_J:      vc_trace.append_str( trace_str, "RD_J " );
      STATE_WAIT_RD_J: vc_trace.append_str( trace_str, "W_RDJ" );
      STATE_CMP:       vc_trace.append_str( trace_str, "CMP  " );
      STATE_WR_I:      vc_trace.append_str( trace_str, "WR_I " );
      STATE_WAIT_WR_I: vc_trace.append_str( trace_str, "W_WRI" );
      STATE_WR_J:      vc_trace.append_str( trace_str, "WR_J " );
      STATE_WAIT_WR_J: vc_trace.append_str( trace_str, "W_WRJ" );
      STATE_DONE:      vc_trace.append_str( trace_str, "DONE " );
      default:         vc_trace.append_str( trace_str, "?    " );
    endcase

    // Print indices and values for debugging
    $sformat( str, "i:%2d j:%2d ", i, j );
    vc_trace.append_str( trace_str, str );

    $sformat( str, "a[j]:%x a[j+1]:%x", array_val_i, array_val_j );
    vc_trace.append_str( trace_str, str );

    vc_trace.append_str( trace_str, ")" );

    xcel_respstream_msg_trace.line_trace( trace_str );
  end
  `VC_TRACE_END

  `endif /* SYNTHESIS */

endmodule

`endif /* LAB2_XCEL_SORT_XCEL_V */
