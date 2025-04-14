#=========================================================================
# MLP_FC_layer_FL_test
#=========================================================================

import pytest
from pymtl3 import *
from pymtl3.stdlib.test_utils import mk_test_case_table, run_sim
from pymtl3.stdlib.stream import StreamSourceFL, StreamSinkFL
from mlp_xcel.mnist_fc_layer_fl import FullyConnected_FL

#-------------------------------------------------------------------------
# TestHarness
#-------------------------------------------------------------------------

class TestHarness(Component):
    def construct(s, dut):
        # Instantiate models
        s.src = StreamSourceFL(Bits32)
        s.sink = StreamSinkFL(Bits32)
        s.dut = dut

        # Connect
        s.src.ostream //= s.dut.istream
        s.dut.ostream //= s.sink.istream

    def done(s):
        return s.src.done() and s.sink.done()

    def line_trace(s):
        return s.src.line_trace() + " > " + s.dut.line_trace() + " > " + s.sink.line_trace()

#-------------------------------------------------------------------------
# Message creation helpers
#-------------------------------------------------------------------------

# Message type constants
MSG_WEIGHT = 0
MSG_BIAS = 1
MSG_INPUT = 2
MSG_OUTPUT = 3  # Output message type

# Helper function to create weight message
def mk_weight_msg(in_channel, out_channel, value):
    return concat(concat(concat(Bits2(MSG_WEIGHT), Bits10(in_channel)), Bits8(out_channel)), Bits12(value))

# Helper function to create bias message
def mk_bias_msg(out_channel, value):
    return concat(concat(Bits2(MSG_BIAS), Bits8(out_channel)), Bits22(value))

# Helper function to create input message
def mk_input_msg(batch_idx, in_channel, value):
    return concat(concat(concat(Bits2(MSG_INPUT), Bits4(batch_idx)), Bits10(in_channel)), Bits16(value))

# Helper function to create expected output message
def mk_output_msg(batch_idx, out_channel, value):
    return concat(concat(concat(Bits2(MSG_OUTPUT), Bits4(batch_idx)), Bits8(out_channel)), Bits18(value))

def mk_imsg( a, b ):
  return concat( Bits32( a, trunc_int=True ), Bits32( b, trunc_int=True ) )

# Make output message, truncate ints to ensure they fit in 32 bits.

def mk_omsg( a ):
  return Bits32( a, trunc_int=True )
#-------------------------------------------------------------------------
# Test Case: small_fc
#-------------------------------------------------------------------------

small_fc_msgs = [
    # Configure weights (3×2 matrix): [[1, 2], [3, 4], [5, 6]]
    mk_weight_msg(0, 0, 1),
    mk_weight_msg(0, 1, 2),
    mk_weight_msg(1, 0, 3),
    mk_weight_msg(1, 1, 4),
    mk_weight_msg(2, 0, 5),
    mk_weight_msg(2, 1, 6),
    
    # Configure biases: [10, 20]
    mk_bias_msg(0, 10),
    mk_bias_msg(1, 20),
    
    # Send input data for batch 0: [1, 2, 3]
    mk_input_msg(0, 0, 1),
    mk_input_msg(0, 1, 2),
    mk_input_msg(0, 2, 3),
    
    # Expected output for batch 0: [32, 48]
    mk_output_msg(0, 0, 32),
    mk_output_msg(0, 1, 48),
    
    # Send input data for batch 1: [4, 5, 6]
    mk_input_msg(1, 0, 4),
    mk_input_msg(1, 1, 5),
    mk_input_msg(1, 2, 6),
    
    # Expected output for batch 1: [59, 84]
    mk_output_msg(1, 0, 59),
    mk_output_msg(1, 1, 84),
]

#-------------------------------------------------------------------------
# Test Case Table
#-------------------------------------------------------------------------

test_case_table = mk_test_case_table([
    (             "msgs              dut_params                                src_delay sink_delay"),
    [ "small_fc",  small_fc_msgs,   {"batch_size": 2, "input_channel": 3, "output_channel": 2}, 0,        0         ],
    # Add more test cases here with different parameters/delays
])

#-------------------------------------------------------------------------
# Run Tests
#-------------------------------------------------------------------------

@pytest.mark.parametrize(**test_case_table)
def test_fully_connected_fl(test_params):
    # Create the DUT
    dut = FullyConnected_FL(**test_params.dut_params)
    
    # Create test harness
    th = TestHarness(dut)
    
    # Setup source and sink streams
    input_msgs = []
    output_msgs = []
    # print(f"Test case: {test_params.msgs}")
    
    # Split messages into inputs and expected outputs
    for i, msg in enumerate(test_params.msgs):
        # msg_type = msg[30:].uint()
        # print(f"Message: {msg}, First 2 bits: {msg_type}")
        if msg[30:].uint() == MSG_OUTPUT:  # If it's an output message
            # print(f"Output message: {msg}")        
            output_msgs.append(msg)
        else:  # If it's an input message (weight, bias, or input)
            # print(f"Input message: {msg}")
            input_msgs.append(msg)
    # print(f"Test case: {input_msgs}")
    # Configure the source and sink
    th.set_param("top.src.construct",
        msgs=input_msgs,
        initial_delay=test_params.src_delay+3,
        interval_delay=test_params.src_delay)
    print(output_msgs)
    
    th.set_param("top.sink.construct",
        msgs=output_msgs,
        initial_delay=test_params.sink_delay+3,
        interval_delay=test_params.sink_delay)
    
    # Run simulation
    run_sim(th)

if __name__ == "__main__":
    # Run directly (not using pytest)
    test_fully_connected_fl(test_case_table["small_fc"])