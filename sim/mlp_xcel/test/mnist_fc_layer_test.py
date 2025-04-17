#=========================================================================
# SortXcel_test
#=========================================================================

import pytest
from pymtl3 import *
from pymtl3.stdlib.test_utils import run_sim
from mlp_xcel.mnist_fc_layer import FullyConnected
from mlp_xcel.test.mnist_fc_layer_fl_test import TestHarness, test_case_table

@pytest.mark.parametrize( **test_case_table )
def test( test_params, cmdline_opts ):
  dut = FullyConnected()
  th = TestHarness( dut )

  input_msgs = []
  output_msgs = []

  for msg in test_params.msgs:
    if msg[30:].uint() == 3:  # MSG_OUTPUT
      output_msgs.append(msg)
    else:
      input_msgs.append(msg)

  th.set_param("top.src.construct",
    msgs=input_msgs,
    initial_delay=test_params.src_delay+3,
    interval_delay=test_params.src_delay )

  th.set_param("top.sink.construct",
    msgs=output_msgs,
    initial_delay=test_params.sink_delay+3,
    interval_delay=test_params.sink_delay )

  run_sim( th, cmdline_opts, duts=['dut'] )


