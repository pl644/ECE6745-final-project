#=========================================================================
# IntMulNstage PyMTL3 Wrapper
#=========================================================================

from pymtl3 import *
from pymtl3.passes.backends.verilog import *
from pymtl3.stdlib.stream.ifcs import IStreamIfc, OStreamIfc

class IntMulNstage( VerilogPlaceholder, Component ):
  def construct( s, nstages=4 ):
    s.istream = IStreamIfc( Bits64 )
    s.ostream = OStreamIfc( Bits32 )

    s.set_metadata( VerilogTranslationPass.explicit_module_name,
                    f"IntMulNstage__{nstages}stage" )

