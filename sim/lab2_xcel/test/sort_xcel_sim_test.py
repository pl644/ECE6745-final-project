#=========================================================================
# sort_xcel_sim_test
#=========================================================================
# Make sure that sort-xcel-sim works.

import pytest
import os

from subprocess import check_call, CalledProcessError
from itertools  import product

impls  = [ "fl","rtl" ]

inputs = [ "random", "sorted-fwd", "sorted-rev" ]

test_cases = []

for impl in impls:
  for inp in inputs:
    test_cases.append([ impl, inp ])

@pytest.mark.parametrize( "impl,input_", test_cases )
def test( impl, input_, cmdline_opts ):

  # Get path to simulator script

  test_dir = os.path.dirname( os.path.abspath( __file__ ) )
  sim_dir  = os.path.dirname( test_dir )
  sim      = sim_dir + os.path.sep + 'sort-xcel-sim'

  # Command

  cmd = [ sim, "--impl", impl, "--input", input_ ]

  # Display simulator command line

  print("")
  print("Simulator command line:", ' '.join(cmd))

  # Run the simulator

  try:
    check_call(cmd)
  except CalledProcessError as e:
    raise Exception( "Error running simulator!" )

