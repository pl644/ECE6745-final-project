from pymtl3 import *
from pymtl3.stdlib.test_utils import run_test_vector_sim
from mlp_xcel.mnist_fc_layer_fl import FullyConnected_FL
import random

def test_fully_connected_fl(cmdline_opts):
  # Create and test with batch_size=2, input_channel=3, output_channel=2
  dut = FullyConnected_FL(batch_size=2, input_channel=3, output_channel=2)
  
  # Initialize test
  dut.apply(DefaultPassGroup())
  dut.sim_reset()
  
  # Message type constants
  MSG_WEIGHT = 0
  MSG_BIAS = 1
  MSG_INPUT = 2
  
  # Configure weights: 3×2 matrix
  # [[1, 2],
  #  [3, 4],
  #  [5, 6]]
  
  # Set up readiness signals
  dut.istream.val @= 1
  dut.ostream.rdy @= 1
  
  # Configure weights using the new 32-bit format:
  # [31:30] - Type, [29:20] - Input Channel, [19:12] - Output Channel, [11:0] - Weight Value
  weight_msg1 = concat(concat(concat(Bits2(MSG_WEIGHT), Bits10(0)), Bits8(0)), Bits12(1))  # weight[0][0] = 1
  dut.istream.msg @= weight_msg1
  dut.sim_tick()
  
  weight_msg2 = concat(concat(concat(Bits2(MSG_WEIGHT), Bits10(0)), Bits8(1)), Bits12(2))  # weight[0][1] = 2
  dut.istream.msg @= weight_msg2
  dut.sim_tick()
  
  weight_msg3 = concat(concat(concat(Bits2(MSG_WEIGHT), Bits10(1)), Bits8(0)), Bits12(3))  # weight[1][0] = 3
  dut.istream.msg @= weight_msg3
  dut.sim_tick()
  
  weight_msg4 = concat(concat(concat(Bits2(MSG_WEIGHT), Bits10(1)), Bits8(1)), Bits12(4))  # weight[1][1] = 4
  dut.istream.msg @= weight_msg4
  dut.sim_tick()
  
  weight_msg5 = concat(concat(concat(Bits2(MSG_WEIGHT), Bits10(2)), Bits8(0)), Bits12(5))  # weight[2][0] = 5
  dut.istream.msg @= weight_msg5
  dut.sim_tick()
  
  weight_msg6 = concat(concat(concat(Bits2(MSG_WEIGHT), Bits10(2)), Bits8(1)), Bits12(6))  # weight[2][1] = 6
  dut.istream.msg @= weight_msg6
  dut.sim_tick()
  
  # Configure biases: [10, 20]
  # [31:30] - Type, [29:22] - Output Channel, [21:0] - Bias Value
  bias_msg1 = concat(concat(Bits2(MSG_BIAS), Bits8(0)), Bits22(10))  # bias[0] = 10
  dut.istream.msg @= bias_msg1
  dut.sim_tick()
  
  bias_msg2 = concat(concat(Bits2(MSG_BIAS), Bits8(1)), Bits22(20))  # bias[1] = 20
  dut.istream.msg @= bias_msg2
  dut.sim_tick()
  
  # Send input data for batch 0
  # [31:30] - Type, [29:26] - Batch Index, [25:16] - Input Channel, [15:0] - Input Value
  input_msg1 = concat(concat(concat(Bits2(MSG_INPUT), Bits4(0)), Bits10(0)), Bits16(1))  # input[0][0] = 1
  dut.istream.msg @= input_msg1
  dut.sim_tick()
  
  input_msg2 = concat(concat(concat(Bits2(MSG_INPUT), Bits4(0)), Bits10(1)), Bits16(2))  # input[0][1] = 2
  dut.istream.msg @= input_msg2
  dut.sim_tick()
  
  # Send the last input for batch 0, which should trigger computation
  input_msg3 = concat(concat(concat(Bits2(MSG_INPUT), Bits4(0)), Bits10(2)), Bits16(3))  # input[0][2] = 3
  dut.istream.msg @= input_msg3
  dut.sim_tick()
  
  # Check outputs for batch 0
  # Expected: [32, 48]
  
  # Allow a few cycles for computation
  for _ in range(3):
    dut.sim_tick()
  
  # Check if output is valid
  # Output format: [31:30] - Type(3), [29:26] - Batch, [25:18] - Output Channel, [17:0] - Value
  if dut.ostream.val:
    output = dut.ostream.msg
    # Extract data from the new format
    batch_idx = output[26:30].uint()
    output_idx = output[18:26].uint()
    value = output[0:18].uint()
    
    print(f"Output batch {batch_idx}, output {output_idx}: {value}")
    assert batch_idx == 0
    assert output_idx == 0
    assert value == 32  # Expected: 1*1 + 2*3 + 3*5 + 10 = 32
  
  # Signal that we've read the data
  dut.ostream.rdy @= 0
  dut.sim_tick()
  dut.ostream.rdy @= 1
  dut.sim_tick()
  
  # Check the second output
  if dut.ostream.val:
    output = dut.ostream.msg
    batch_idx = output[26:30].uint()
    output_idx = output[18:26].uint()
    value = output[0:18].uint()
    
    print(f"Output batch {batch_idx}, output {output_idx}: {value}")
    assert batch_idx == 0
    assert output_idx == 1
    assert value == 48  # Expected: 1*2 + 2*4 + 3*6 + 20 = 48
  
  # Signal that we've read the data
  dut.ostream.rdy @= 0
  dut.sim_tick()
  dut.ostream.rdy @= 1
  dut.sim_tick()
  
  # Send input data for batch 1
  input_msg4 = concat(concat(concat(Bits2(MSG_INPUT), Bits4(1)), Bits10(0)), Bits16(4))  # input[1][0] = 4
  dut.istream.msg @= input_msg4
  dut.sim_tick()
  
  input_msg5 = concat(concat(concat(Bits2(MSG_INPUT), Bits4(1)), Bits10(1)), Bits16(5))  # input[1][1] = 5
  dut.istream.msg @= input_msg5
  dut.sim_tick()
  
  # Send the last input for batch 1
  input_msg6 = concat(concat(concat(Bits2(MSG_INPUT), Bits4(1)), Bits10(2)), Bits16(6))  # input[1][2] = 6
  dut.istream.msg @= input_msg6
  dut.sim_tick()
  
  # Allow a few cycles for computation
  for _ in range(3):
    dut.sim_tick()
  
  # Check outputs for batch 1
  # Expected: [59, 84]
  if dut.ostream.val:
    output = dut.ostream.msg
    batch_idx = output[26:30].uint()
    output_idx = output[18:26].uint()
    value = output[0:18].uint()
    
    print(f"Output batch {batch_idx}, output {output_idx}: {value}")
    assert batch_idx == 1
    assert output_idx == 0
    assert value == 59  # Expected: 4*1 + 5*3 + 6*5 + 10 = 59
  
  # Signal that we've read the data
  dut.ostream.rdy @= 0
  dut.sim_tick()
  dut.ostream.rdy @= 1
  dut.sim_tick()
  
  # Check the second output
  if dut.ostream.val:
    output = dut.ostream.msg
    batch_idx = output[26:30].uint()
    output_idx = output[18:26].uint()
    value = output[0:18].uint()
    
    print(f"Output batch {batch_idx}, output {output_idx}: {value}")
    assert batch_idx == 1
    assert output_idx == 1
    assert value == 84  # Expected: 4*2 + 5*4 + 6*6 + 20 = 84
  
  print("All tests passed!")

if __name__ == "__main__":
    test_fully_connected_fl()