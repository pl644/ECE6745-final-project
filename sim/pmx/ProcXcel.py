#=========================================================================
# ProcXcel
#=========================================================================
# No caches, just processor + accelerator

from pymtl3 import *
from pymtl3.stdlib.stream.ifcs import IStreamIfc, OStreamIfc
from pymtl3.stdlib.mem.ifcs    import MemRequesterIfc
from pymtl3.stdlib.mem         import mk_mem_msg

class ProcXcel( Component ):

  def construct( s, ProcType, XcelType ):

    CacheReqMsg, CacheRespMsg = mk_mem_msg( 8, 32, 32 )

    # interface to outside ProcMemXcel

    s.proc = ProcType()
    s.xcel = XcelType()

    # Processor core id is hard coded to zero

    s.proc.core_id //= 0

    # Connect processor to xcel

    s.xcel.xcel //= s.proc.xcel

    # Pass proc/mngr interface up to top-level

    s.mngr2proc = IStreamIfc( Bits32 )
    s.proc2mngr = OStreamIfc( Bits32 )

    s.mngr2proc //= s.proc.mngr2proc
    s.proc2mngr //= s.proc.proc2mngr

    # Pass memory interfaces up to top-level

    s.imem = MemRequesterIfc( CacheReqMsg, CacheRespMsg )
    s.dmem = MemRequesterIfc( CacheReqMsg, CacheRespMsg )
    s.xmem = MemRequesterIfc( CacheReqMsg, CacheRespMsg )

    s.imem //= s.proc.imem
    s.dmem //= s.proc.dmem
    s.xmem //= s.xcel.mem

    # Pass other signals up to top-level

    s.stats_en    = OutPort()
    s.commit_inst = OutPort()

    s.stats_en    //= s.proc.stats_en
    s.commit_inst //= s.proc.commit_inst

  def line_trace( s ):
    return s.proc.line_trace() + "|" + s.xcel.line_trace()

