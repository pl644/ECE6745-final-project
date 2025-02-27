#=========================================================================
# NullXcel PyMTL Wrapper
#=========================================================================

from pymtl3 import *
from pymtl3.passes.backends.verilog import *
from pymtl3.stdlib.mem.ifcs  import MemRequesterIfc
from pymtl3.stdlib.mem       import MemMsgType, mk_mem_msg
from pymtl3.stdlib.xcel      import XcelMsgType, mk_xcel_msg
from pymtl3.stdlib.xcel.ifcs import XcelResponderIfc

class NullXcel( VerilogPlaceholder, Component ):
  def construct( s ):

    MemReqMsg,  MemRespMsg  = mk_mem_msg( 8, 32, 32 )
    XcelReqMsg, XcelRespMsg = mk_xcel_msg( 5, 32 )

    s.xcel = XcelResponderIfc( XcelReqMsg, XcelRespMsg )
    s.mem  = MemRequesterIfc( MemReqMsg, MemRespMsg )

