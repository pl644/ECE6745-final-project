#=========================================================================
# IntMulScycle_perf_test
#=========================================================================
# These are performance regressions to make sure the performance of this
# design is reasonable.

from pymtl3.stdlib.test_utils import run_sim

from lab1_imul.test.IntMulFL_test import TestHarness, mk_imsg, mk_omsg

from lab1_imul.IntMulScycle import IntMulScycle

#-------------------------------------------------------------------------
# run_perf_check
#-------------------------------------------------------------------------
# Takes two input values and the min and max latency. Determines the
# correct result, runs a simulation, and confirms the final cycle count
# is within the min and max latency.
#
# Note that in all of our tests, the very first message is sent to the
# multiplier on cycle 4 so there are three extra cycles at the beginning
# of the test. There are always an extra 5 cycles after the last message
# is received before the test is done. Plus the cycle count seems to add
# one more cycle probably based on a final increment of the cycle
# counter. So the total cycle count will be the latency of the multiplier
# plus 9.

def run_perf_check( cmdline_opts, a, b, min_latency, max_latency ):

  result = mk_omsg( a * b )

  th = TestHarness( IntMulScycle() )

  th.set_param( "top.src.construct",  msgs=[ mk_imsg(a,b) ] )
  th.set_param( "top.sink.construct", msgs=[ result ] )

  run_sim( th, cmdline_opts, duts=['imul'] )

  latency = th.sim_cycle_count() - 9

  print("min target latency = ",min_latency)
  print("max target latency = ",max_latency)
  print("    actual latency = ",latency)

  assert min_latency <= latency and latency <= max_latency

#-------------------------------------------------------------------------
# test_perf0
#-------------------------------------------------------------------------

def test_perf0( cmdline_opts ):
  run_perf_check( cmdline_opts, 0xffffffff, 0xffffffff, 2, 2 )

#-------------------------------------------------------------------------
# test_perf1
#-------------------------------------------------------------------------

def test_perf1( cmdline_opts ):
  run_perf_check( cmdline_opts, 0x33333333, 0x33333333, 2, 2 )

#-------------------------------------------------------------------------
# test_perf2
#-------------------------------------------------------------------------

def test_perf2( cmdline_opts ):
  run_perf_check( cmdline_opts, 0x11111111, 0x11111111, 2, 2 )

#-------------------------------------------------------------------------
# test_perf3
#-------------------------------------------------------------------------

def test_perf3( cmdline_opts ):
  run_perf_check( cmdline_opts, 0, 0, 2, 2 )

