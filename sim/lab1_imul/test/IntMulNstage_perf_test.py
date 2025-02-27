#=========================================================================
# IntMulNstage_perf_test
#=========================================================================
# These are performance regressions to make sure the performance of this
# design is reasonable.

from pymtl3.stdlib.test_utils import run_sim

from lab1_imul.test.IntMulFL_test import TestHarness, mk_imsg, mk_omsg

from lab1_imul.IntMulNstage import IntMulNstage

#-------------------------------------------------------------------------
# run_perf_check
#-------------------------------------------------------------------------
# Takes two input values and the min and max latency. Determines the
# correct result, runs a simulation, and confirms the final cycle count
# matches the expected latency.
#
# Note that in all of our tests, the very first message is sent to the
# multiplier on cycle 4 so there are three extra cycles at the beginning
# of the test. There are always an extra 5 cycles after the last message
# is received before the test is done. Plus the cycle count seems to add
# one more cycle probably based on a final increment of the cycle
# counter. So the total cycle count will be the latency of the multiplier
# plus 9.

def run_perf_check( cmdline_opts, nstages, target_latency ):

  result = mk_omsg( 0xf * 0xf )

  th = TestHarness( IntMulNstage(nstages=nstages) )

  th.set_param( "top.src.construct",  msgs=[ mk_imsg(0xf,0xf) ] )
  th.set_param( "top.sink.construct", msgs=[ result ] )

  run_sim( th, cmdline_opts, duts=['imul'] )

  latency = th.sim_cycle_count() - 9

  print("target latency = ",target_latency)
  print("actual latency = ",latency)

  assert latency == target_latency

#-------------------------------------------------------------------------
# test_perf_1stage
#-------------------------------------------------------------------------
# Expected latency is 2 not 1 because this includes the stage before
# the pipeline.

def test_perf_1stage( cmdline_opts ):
  run_perf_check( cmdline_opts, 1, 2 )

#-------------------------------------------------------------------------
# test_perf_2stage
#-------------------------------------------------------------------------
# Expected latency is 3 not 2 because this includes the stage before
# the pipeline.

def test_perf_2stage( cmdline_opts ):
  run_perf_check( cmdline_opts, 2, 3 )

#-------------------------------------------------------------------------
# test_perf_4stage
#-------------------------------------------------------------------------
# Expected latency is 5 not 4 because this includes the stage before
# the pipeline.

def test_perf_4stage( cmdline_opts ):
  run_perf_check( cmdline_opts, 4, 5 )

#-------------------------------------------------------------------------
# test_perf_8stage
#-------------------------------------------------------------------------
# Expected latency is 9 not 8 because this includes the stage before
# the pipeline.

def test_perf_8stage( cmdline_opts ):
  run_perf_check( cmdline_opts, 8, 9 )

#-------------------------------------------------------------------------
# run_perf_throughput
#-------------------------------------------------------------------------
# Takes a stream of eight input values and makes sure the given pipelined
# multiplier can sustain one transaction per cycle.
#
# Note that in all of our tests, the very first message is sent to the
# multiplier on cycle 4. So for an N stage multiplier the first result
# should come out on cycle 4+N. There There are always an extra 5 cycles
# after the last message is received before the test is done. Plus the
# cycle count seems to add one more cycle probably based on a final
# increment of the cycle counter.
#
# This means the steady state throughput for 8 transactions is the
# (cycle_count - (9+N)) / 8
#

def run_perf_throughput( cmdline_opts, nstages ):

  imsgs = []
  omsgs = []

  for i in range(8):
    imsgs.extend([mk_imsg(i,i)])
    omsgs.extend([mk_omsg(i*i)])

  th = TestHarness( IntMulNstage(nstages=nstages) )

  th.set_param( "top.src.construct",  msgs=imsgs )
  th.set_param( "top.sink.construct", msgs=omsgs )

  run_sim( th, cmdline_opts, duts=['imul'] )

  latency    = th.sim_cycle_count() - 9
  throughput = (th.sim_cycle_count() - (9+nstages)) / 8.0

  print("actual latency    = ",latency)
  print("actual throughput = ",throughput)

  assert throughput == 1.0

#-------------------------------------------------------------------------
# test_perf_throughput
#-------------------------------------------------------------------------

def test_perf_throughput_1stage( cmdline_opts ):
  run_perf_throughput( cmdline_opts, 1 )

def test_perf_throughput_2stage( cmdline_opts ):
  run_perf_throughput( cmdline_opts, 2 )

def test_perf_throughput_4stage( cmdline_opts ):
  run_perf_throughput( cmdline_opts, 4 )

def test_perf_throughput_8stage( cmdline_opts ):
  run_perf_throughput( cmdline_opts, 8 )

