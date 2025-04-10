#=========================================================================
# mnist_fc_layer_PE_test
#=========================================================================

from pymtl3 import *
from pymtl3.stdlib.test_utils import run_test_vector_sim
from mlp_xcel.mnist_fc_layer_PE import SystolicPE

#-------------------------------------------------------------------------
# test_basic
#-------------------------------------------------------------------------

def test_basic(cmdline_opts):
    # Create the DUT
    dut = SystolicPE()
    
    # Define test vectors
    test_vectors = [
        # weight_in    act_in      sum_in      act_out*    sum_out*
        [ 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000],
        [ 0x00000003, 0x00000004, 0x0000000A, 0x00000004, 0x0000001E],  # 3*4+10=22
        [ 0x00000002, 0x00000005, 0x0000000B, 0x00000005, 0x00000019],  # 2*5+11=21
        [ 0x00000005, 0x00000003, 0x00000010, 0x00000003, 0x00000025],  # 5*3+16=31
        [ 0x00000000, 0x00000007, 0x00000020, 0x00000007, 0x00000020],  # 0*7+32=32
    ]
    
    # Run the simulation with test vectors
    run_test_vector_sim(dut, [
        ('weight_in    act_in      sum_in      act_out*    sum_out*'),
        *test_vectors
    ], cmdline_opts)

#-------------------------------------------------------------------------
# TestHarness
#-------------------------------------------------------------------------

class TestHarness(Component):
    def construct(s):
        # Input/Output ports
        s.input_a = InPort(32)
        s.input_b = InPort(32)
        s.output  = OutPort(32)
        
        # Create a simple PE
        s.pe = SystolicPE()
        
        # Connect the PE
        s.pe.weight_in //= s.input_a
        s.pe.act_in    //= s.input_b
        s.pe.sum_in    //= 0
        s.pe.sum_out   //= s.output
        
    def line_trace(s):
        return f"{s.input_a}:{s.input_b}→{s.output}"

#-------------------------------------------------------------------------
# test_multiply_accumulate
#-------------------------------------------------------------------------

def test_multiply_accumulate(cmdline_opts):
    # Create the test harness
    th = TestHarness()
    
    # Define test vectors
    test_vectors = [
        # input_a  input_b  output*
        [ 0,       0,       0      ],
        [ 2,       3,       6      ],  # 2*3+0=6
        [ 4,       5,       20     ],  # 4*5+0=20
        [ 3,       4,       12     ],  # 3*4+0=12
        [ 10,      13,      130    ],  # 10*13+0=130
        [ 8,       7,       56     ],  # 8*7+0=56
    ]
    
    # Run the simulation with test vectors
    run_test_vector_sim(th, [
        ('input_a  input_b  output*'),
        *test_vectors
    ], cmdline_opts)