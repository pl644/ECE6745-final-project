from pymtl3 import *
from pymtl3.stdlib.test_utils import run_test_vector_sim
from mlp_xcel.mnist_fc_layer_fl import FullyConnected_FL
import random
#-------------------------------------------------------------------------
# test for new datapath module TrailingZeroCounter
#-------------------------------------------------------------------------

def test_fully_connected_fl(cmdline_opts):
  # Create and test with batch_size=2, input_channel=3, output_channel=2
  # This makes it easier to verify matrix multiplication results
  dut = FullyConnected_FL(batch_size=2, input_channel=3, output_channel=2)
  
  # Initialize test data manually since we need to set 2D array inputs
  dut.apply(DefaultPassGroup())
  dut.sim_reset()
  
  # Test case 1: Set input values
  # Input vector: 2×3 matrix
  # [[1, 2, 3],
  #  [4, 5, 6]]
  dut.input_vector[0][0] @= 1
  dut.input_vector[0][1] @= 2
  dut.input_vector[0][2] @= 3
  dut.input_vector[1][0] @= 4
  dut.input_vector[1][1] @= 5
  dut.input_vector[1][2] @= 6
  
  # Weights: 3×2 matrix
  # [[1, 2],
  #  [3, 4],
  #  [5, 6]]
  dut.weights[0][0] @= 1
  dut.weights[0][1] @= 2
  dut.weights[1][0] @= 3
  dut.weights[1][1] @= 4
  dut.weights[2][0] @= 5
  dut.weights[2][1] @= 6
  
  # Biases: [10, 20]
  dut.biases[0] @= 10
  dut.biases[1] @= 20
  
  # Simulate one cycle
  dut.sim_tick()
  
  # Expected output: input_vector @ weights + biases
  # [[1, 2, 3],     [[1, 2],      [[10, 20]]
  #  [4, 5, 6]]  ×   [3, 4],   +
  #                  [5, 6]]
  #
  # = [[1×1 + 2×3 + 3×5 + 10, 1×2 + 2×4 + 3×6 + 20],
  #    [4×1 + 5×3 + 6×5 + 10, 4×2 + 5×4 + 6×6 + 20]]
  #
  # = [[32, 44],
  #    [74, 98]]
  
  # Check outputs
  assert dut.output[0][0] == 32
  assert dut.output[0][1] == 48
  assert dut.output[1][0] == 59
  assert dut.output[1][1] == 84
  




if __name__ == "_main_":
    test_fully_connected_fl()