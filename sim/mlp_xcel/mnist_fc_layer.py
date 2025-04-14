#=========================================================================
# MLP FC layer PyMTL3 Wrapper
#=========================================================================

from pymtl3 import *
from pymtl3.stdlib.stream.ifcs import IStreamIfc, OStreamIfc
from pymtl3.passes.backends.verilog import *

class FullyConnected(VerilogPlaceholder ,Component):
  def construct(s):
    # Interface
    s.istream = IStreamIfc(Bits32)  # Input stream
    s.ostream = OStreamIfc(Bits32)  # Output stream

    s.set_metadata( VerilogTranslationPass.explicit_module_name,
                    'FullyConnected' )