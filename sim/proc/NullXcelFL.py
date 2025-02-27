#=========================================================================
# NullXcelFL
#=========================================================================
# This is an empty accelerator model. It includes a single 32-bit
# register named xr0 for testing purposes. It includes a memory
# interface, but this memory interface is not used.

from pymtl3 import *
from pymtl3.stdlib.mem.ifcs  import MemRequesterIfc
from pymtl3.stdlib.mem       import MemMsgType, mk_mem_msg
from pymtl3.stdlib.xcel      import XcelMsgType, mk_xcel_msg
from pymtl3.stdlib.xcel.ifcs import XcelResponderIfc
from pymtl3.stdlib.primitive import RegEn
from pymtl3.stdlib.stream    import StreamNormalQueue

class NullXcelFL(Component):

  def construct( s ):

    MemReqMsg,  MemRespMsg  = mk_mem_msg( 8, 32, 32 )
    XcelReqMsg, XcelRespMsg = mk_xcel_msg( 5, 32 )

    # Interface

    s.xcel = XcelResponderIfc( XcelReqMsg, XcelRespMsg )
    s.mem  = MemRequesterIfc( MemReqMsg, MemRespMsg )

    # Queues

    s.xcelreq_q = StreamNormalQueue( XcelReqMsg, 2 )
    s.xcelreq_q.istream //= s.xcel.reqstream

    # Single accelerator register

    s.xr0 = RegEn( 32 )

    # Direct connections for xcelreq/xcelresp

    s.xr0.in_                   //= s.xcelreq_q.ostream.msg.data
    s.xcel.respstream.msg.type_ //= s.xcelreq_q.ostream.msg.type_

    # Directly connect response val/rdy to queue

    s.xcel.respstream.val //= s.xcelreq_q.ostream.val
    s.xcel.respstream.rdy //= s.xcelreq_q.ostream.rdy

    # Even though memreq/memresp interface is not hooked up, we still
    # need to set the output ports correctly.

    s.mem.reqstream.val  //= 0
    s.mem.reqstream.msg  //= MemReqMsg( MemMsgType.READ, 0, 0 )
    s.mem.respstream.rdy //= 0

    # Combinational block

    @update
    def up_null_xcel():

      # Mux to force xcelresp data to zero on a write
      # Enable xr0 only upon write requests and both val/rdy on resp side

      if s.xcelreq_q.ostream.msg.type_ == XcelMsgType.WRITE:
        s.xr0.en @= s.xcel.respstream.val & s.xcel.respstream.rdy
        s.xcel.respstream.msg.data @= 0
      else:
        s.xr0.en @= 0
        s.xcel.respstream.msg.data @= s.xr0.out

  def line_trace( s ):
    return f"{s.xcel.reqstream}(){s.xcel.respstream}"

