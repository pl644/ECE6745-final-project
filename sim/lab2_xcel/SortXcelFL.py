#=========================================================================
# Sort Xcel Unit FL Model
#=========================================================================
# Sort array in memory containing positive integers.
# Accelerator register interface:
#
#  xr0 : go/done
#  xr1 : base address of array
#  xr2 : number of elements in array
#
# Accelerator protocol involves the following steps:
#  1. Write the base address of array via xr1
#  2. Write the number of elements in array via xr2
#  3. Tell accelerator to go by writing xr0
#  4. Wait for accelerator to finish by reading xr0, result will be 1
#

from pymtl3 import *
from pymtl3.stdlib.mem.ifcs  import MemRequesterIfc
from pymtl3.stdlib.mem       import MemMsgType, mk_mem_msg
from pymtl3.stdlib.mem       import MemRequesterAdapterFL
from pymtl3.stdlib.xcel.ifcs import XcelResponderIfc
from pymtl3.stdlib.xcel      import XcelMsgType, mk_xcel_msg
from pymtl3.stdlib.stream    import OStreamBlockingAdapterFL
from pymtl3.stdlib.stream    import IStreamBlockingAdapterFL

class SortXcelFL( Component ):

  def construct( s ):

    MemReqMsg,  MemRespMsg  = mk_mem_msg( 8, 32, 32 )
    XcelReqMsg, XcelRespMsg = mk_xcel_msg( 5, 32 )

    # Interface

    s.xcel = XcelResponderIfc( XcelReqMsg, XcelRespMsg )
    s.mem  = MemRequesterIfc( MemReqMsg, MemRespMsg )

    # Proc <-> Xcel Adapters

    s.xcelreq_q  = IStreamBlockingAdapterFL( XcelReqMsg  )
    s.xcelresp_q = OStreamBlockingAdapterFL( XcelRespMsg )

    connect( s.xcelreq_q.istream,  s.xcel.reqstream  )
    connect( s.xcelresp_q.ostream, s.xcel.respstream )

    # Xcel <-> Memory Adapters

    s.mem_adapter = MemRequesterAdapterFL( MemReqMsg, MemRespMsg )

    connect( s.mem, s.mem_adapter.requester )

    # Storage

    s.base_addr  = 0
    s.array_size = 0

    @update_once
    def up_sort_xcel():

      # We loop handling accelerator requests. We are only expecting
      # writes to xr0-2, so any other requests are an error. We exit the
      # loop when we see the write to xr0.

      go = False
      while not go:

        xcelreq_msg = s.xcelreq_q.deq()

        if xcelreq_msg.type_ == XcelMsgType.WRITE:
          assert xcelreq_msg.addr in [0,1,2], \
            "Only reg writes to 0,1,2 allowed during setup!"

          # Use xcel register address to configure accelerator

          if   xcelreq_msg.addr == 0: go = True
          if   xcelreq_msg.addr == 1: s.base_addr  = xcelreq_msg.data
          elif xcelreq_msg.addr == 2: s.array_size = xcelreq_msg.data

          # Send xcel response message

          s.xcelresp_q.enq( XcelRespMsg( XcelMsgType.WRITE, 0 ) )

      array = []
      for i in range(s.array_size):
        array.append( s.mem_adapter.read( s.base_addr + i*4, 4 ) )

      array = sorted(array)

      for i in range(s.array_size):
        s.mem_adapter.write( s.base_addr + i*4, 4, array[i] )

      # Now wait for read of xr0

      xcelreq_msg = s.xcelreq_q.deq()

      # Only expecting read from xr0, so any other request is an xcel
      # protocol error.

      assert xcelreq_msg.type_ == XcelMsgType.READ, \
        "Only reg reads allowed during done phase!"

      assert xcelreq_msg.addr == 0, \
        "Only reg read to 0 allowed during done phase!"

      # Send xcel response message indicating xcel is done

      s.xcelresp_q.enq( XcelRespMsg( XcelMsgType.READ, 1 ) )

  # Line tracing

  def line_trace( s ):
    return f"{s.xcel.reqstream}(){s.xcel.respstream}"

