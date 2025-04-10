from pymtl3 import *
from pymtl3.passes.backends.verilog import *

# Define bit widths
Bits32 = mk_bits(32)

class FullyConnected_FL(Component):
  def construct(s, input_size=4, output_size=2):
    # Input and output ports
    s.input_vector = [InPort(Bits32) for _ in range(input_size)]
    s.output = [OutPort(Bits32) for _ in range(output_size)]
    
    # Make weights and biases configurable via input ports
    s.weights = [[InPort(Bits32) for _ in range(input_size)] for _ in range(output_size)]
    s.biases = [InPort(Bits32) for _ in range(output_size)]
    
    # Define the update logic
    @update
    def compute():
      for i in range(output_size):
        # Full dot product computation
        accum = Bits32(0)
        for j in range(input_size):
          # Fixed-point multiplication
          prod = (s.input_vector[j] * s.weights[i][j]) >> 8
          accum += prod
        
        # Add bias and apply ReLU activation
        with_bias = accum + s.biases[i]
        # ReLU activation: max(0, x)
        if with_bias < 0:
          s.output[i] @= Bits32(0)
        else:
          s.output[i] @= with_bias
    
    # Add line trace method required by PyMTL3 test framework
    def line_trace(s):
      return f"{s.input_vector}(){s.output}"