from pymtl3 import *
from pymtl3.passes.backends.verilog import *

# Define bit widths
Bits32 = mk_bits(32)

class FullyConnected_FL(Component):
  def construct(s, batch_size=4, input_channel = 4, output_channel=2):
    # Input and output ports
    s.input_vector = [[InPort(Bits32) for _ in range(input_channel)]for _ in range(batch_size)]
    s.output = [[OutPort(Bits32) for _ in range(output_channel)]for _ in range(batch_size)]
    
    # Make weights and biases configurable via input ports
    s.weights = [[InPort(Bits32) for _ in range(output_channel)] for _ in range(input_channel)]
    s.biases = [InPort(Bits32) for _ in range(output_channel)]
    
    # Define the update logic
    @update
    def compute():
       for i in range(batch_size):
        for j in range(output_channel):
          # Full dot product computation
          accum = s.biases[j]
          for k in range(input_channel):
            # Fixed-point multiplication
             accum += (s.input_vector[i][k] * s.weights[k][j])
          s.output[i][j] @= accum
          # ReLU activation: max(0, x)
          # if with_bias < 0:
          #   s.output[i] @= Bits32(0)
          # else:
          #   s.output[i] @= with_bias
    
    # Add line trace method required by PyMTL3 test framework
    # def line_trace(s):
    #   return f"{s.input_vector}(){s.output}"