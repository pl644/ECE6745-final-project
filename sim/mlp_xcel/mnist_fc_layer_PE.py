#=========================================================================
# mnist_fc_layer_PE PyMTL3 Wrapper
#=========================================================================

from pymtl3 import *
from pymtl3.passes.backends.verilog import *

class SystolicPE(VerilogPlaceholder, Component):
    def construct(s):
        # Data signals
        s.weight_in = InPort(32)
        s.act_in    = InPort(32)
        s.sum_in    = InPort(32)
        
        s.act_out   = OutPort(32)
        s.sum_out   = OutPort(32)