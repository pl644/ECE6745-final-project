#=========================================================================
# IntMulNstageStep PyMTL3 Wrapper
#=========================================================================

from pymtl3 import *
from pymtl3.passes.backends.verilog import *

class IntMulNstageStep( VerilogPlaceholder, Component ):
  def construct( s ):

    s.in_val     = InPort  ()
    s.in_a       = InPort  (32)
    s.in_b       = InPort  (32)
    s.in_result  = InPort  (32)

    s.out_val    = OutPort ()
    s.out_a      = OutPort (32)
    s.out_b      = OutPort (32)
    s.out_result = OutPort (32)

