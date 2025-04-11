from pymtl3 import *
from pymtl3.passes.backends.verilog import *

# Define bit widths
Bits32 = mk_bits(32)

class FullyConnected(Component):
  def construct(s, batch_size=4, input_channel = 4, output_channel=2):
    # Input and output ports
    s.input_vector = [[InPort(Bits32) for _ in range(input_channel)]for _ in range(batch_size)]
    s.output_result = [[OutPort(Bits32) for _ in range(output_channel)]for _ in range(batch_size)]
    
    # Make weights and biases configurable via input ports
    s.weights = [[InPort(Bits32) for _ in range(output_channel)] for _ in range(input_channel)]
    s.biases = [InPort(Bits32) for _ in range(output_channel)]