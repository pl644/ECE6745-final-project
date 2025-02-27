#=========================================================================
# Proc_perf_test.py
#=========================================================================
# These tests will check how many cycles the alternative processor will
# take. There are multiple types of tests that check how squashes, stalls,
# and bypasses are handled. The correctness of the processor outputs are
# not checked, only the number of cycles taken to complete.

import pytest

from pymtl3 import *
from pymtl3.stdlib.test_utils import run_sim

from proc.test.harness import TestHarness

from proc.tinyrv2_encoding import assemble
from proc.Proc  import Proc

#-------------------------------------------------------------------------
# Setup
#-------------------------------------------------------------------------

def run_perf_test( cmdline_opts, asm_prog, min_latency, max_latency ):

  # Instantiate model

  model = TestHarness( Proc )
  model.elaborate()

  mem_image = assemble( asm_prog )
  model.load( mem_image )

  run_sim( model, cmdline_opts, duts=['proc'] )

  # There are 10 additional cycles needed for testing overhead.
  # 3 cycles are for setting up the simulation, 1 cycle is for
  # csrw to check the value in the register, 5 cycles are for
  # the remaining cycles in the simulation, and 1 cycle is for
  # ending the simulation. All these extra cycles are removed
  # when checking the number of cycles taken by each test.
  #  1r .        >          |                       |    |    |    |[ ]                         >                  [  ]|[ ]                         >.                 [  ] >
  #  2r .        >          |                       |    |    |    |[ ]                         >                  [  ]|[ ]                         >.                 [  ] >
  #  3: .        >          |                       |    |    |    |[ ]rd:00:00000200:0:        >                  [  ]|[ ]                         >.                 [  ] >
  #  4: .        >  00000200|                       |    |    |    |[ ]rd:00:00000204:0:        >rd:00:0:0:00100213[ *]|[ ]                         >.                 [  ] >
  #  5: .        >  00000204|addi   x04, x00, 0x001 |    |    |    |[ ]rd:00:00000208:0:        >rd:00:0:0:7c001073[ *]|[ ]                         >.                 [  ] >
  #  6: .        >  00000208|csrw   proc2mngr, x00  |addi|    |    |[ ]rd:00:0000020c:0:        >rd:00:0:0:fc0020f3[ *]|[ ]                         >.                 [  ] >
  #  7: .        >  #       |#                      |csrw|addi|    |[ ]                         >#                 [ *]|[ ]                         >.                 [  ] >
  #  8: .        >  #       |#                      |    |csrw|addi|[ ]                         >#                 [ *]|[ ]                         >.                 [  ] >
  #  9: .        >  #       |#                      |    |    |csrw|[ ]                         >#                 [ *]|[ ]                         >.                 [  ] > 00000000
  # 10: .        >  #       |#                      |    |    |    |[ ]                         >#                 [ *]|[ ]                         >.                 [  ] >
  # 11: .        >  #       |#                      |    |    |    |[ ]                         >#                 [ *]|[ ]                         >.                 [  ] >
  # 12: .        >  #       |#                      |    |    |    |[ ]                         >#                 [ *]|[ ]                         >.                 [  ] >
  # 13: .        >  #       |#                      |    |    |    |[ ]                         >#                 [ *]|[ ]                         >.                 [  ] >
  # 14: .        >  #       |#                      |    |    |    |[ ]                         >#                 [ *]|[ ]                         >.

  latency = model.sim_cycle_count() - 10

  print("min target latency = ",min_latency)
  print("max target latency = ",max_latency)
  print("    actual latency = ",latency)

  assert min_latency <= latency and latency <= max_latency

#-------------------------------------------------------------------------
# RS1 X
#-------------------------------------------------------------------------

def test_perf0( cmdline_opts ):

  prog="""
                    # 1 2 3 4 5 6 7 8 9
    addi x1, x0, 1  # F D X M W
    addi x2, x1, 1  #   F D X M W

    # need final CSRW to stop the test
    csrw proc2mngr, x0 > 0
  """

  run_perf_test( cmdline_opts, prog, 6, 6 )

#-------------------------------------------------------------------------
# RS2 X
#-------------------------------------------------------------------------

def test_perf1( cmdline_opts ):

  prog="""
                     # 1 2 3 4 5 6 7 8 9
    addi x1, x0,  1  # F D X M W
    add  x2, x1, x0  #   F D X M W

    # need final CSRW to stop the test
    csrw proc2mngr, x0 > 0
  """

  run_perf_test( cmdline_opts, prog, 6, 6 )

#-------------------------------------------------------------------------
# RS1 RS2 X
#-------------------------------------------------------------------------

def test_perf2( cmdline_opts ):

  prog="""
                     # 1 2 3 4 5 6 7 8 9
    addi x1, x0,  1  # F D X M W
    add  x2, x1, x1  #   F D X M W

    # need final CSRW to stop the test
    csrw proc2mngr, x0 > 0
  """

  run_perf_test( cmdline_opts, prog, 6, 6 )

#-------------------------------------------------------------------------
# RS1 M
#-------------------------------------------------------------------------

def test_perf3( cmdline_opts ):

  prog="""
                    # 1 2 3 4 5 6 7 8 9
    addi x1, x0, 1  # F D X M W
    nop             #   F D X M W
    addi x2, x1, 1  #     F D X M W

    # need final CSRW to stop the test
    csrw proc2mngr, x0 > 0
  """

  run_perf_test( cmdline_opts, prog, 7, 7 )

#-------------------------------------------------------------------------
# RS2 M
#-------------------------------------------------------------------------

def test_perf4( cmdline_opts ):

  prog="""
                     # 1 2 3 4 5 6 7 8 9
    addi x1, x0,  1  # F D X M W
    nop              #   F D X M W
    add  x2, x1, x0  #     F D X M W

    # need final CSRW to stop the test
    csrw proc2mngr, x0 > 0
  """

  run_perf_test( cmdline_opts, prog, 7, 7 )

#-------------------------------------------------------------------------
# RS1 RS2 M
#-------------------------------------------------------------------------

def test_perf5( cmdline_opts ):

  prog="""
                     # 1 2 3 4 5 6 7 8 9
    addi x1, x0,  1  # F D X M W
    nop              #   F D X M W
    add  x2, x1, x1  #     F D X M W

    # need final CSRW to stop the test
    csrw proc2mngr, x0 > 0
  """

  run_perf_test( cmdline_opts, prog, 7, 7 )

#-------------------------------------------------------------------------
# RS1 W
#-------------------------------------------------------------------------

def test_perf6( cmdline_opts ):

  prog="""
                    # 1 2 3 4 5 6 7 8 9
    addi x1, x0, 1  # F D X M W
    nop             #   F D X M W
    nop             #     F D X M W
    addi x2, x1, 1  #       F D X M W

    # need final CSRW to stop the test
    csrw proc2mngr, x0 > 0
  """

  run_perf_test( cmdline_opts, prog, 8, 8 )

#-------------------------------------------------------------------------
# RS2 W
#-------------------------------------------------------------------------

def test_perf7( cmdline_opts ):

  prog="""
                     # 1 2 3 4 5 6 7 8 9
    addi x1, x0,  1  # F D X M W
    nop              #   F D X M W
    nop              #     F D X M W
    add  x2, x1, x0  #       F D X M W

    # need final CSRW to stop the test
    csrw proc2mngr, x0 > 0
  """

  run_perf_test( cmdline_opts, prog, 8, 8 )

#-------------------------------------------------------------------------
# RS1 RS2 W
#-------------------------------------------------------------------------

def test_perf8( cmdline_opts ):

  prog="""
                     # 1 2 3 4 5 6 7 8 9
    addi x1, x0,  1  # F D X M W
    nop              #   F D X M W
    nop              #     F D X M W
    add  x2, x1, x1  #       F D X M W

    # need final CSRW to stop the test
    csrw proc2mngr, x0 > 0
  """

  run_perf_test( cmdline_opts, prog, 8, 8 )

#-------------------------------------------------------------------------
# SW Test
#-------------------------------------------------------------------------

def test_perf9( cmdline_opts ):

  prog="""                          #                   1 1 1 1 1 1
                                    # 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5
    csrr x1, mngr2proc < 0x00002000 # F D X M W
    csrr x2, mngr2proc < 0xdeadbeef #   F D X M W
    sw   x2, 0(x1)                  #     F D X M W
    addi x3, x0, 1                  #       F D X M W

    # need final CSRW to stop the test
    csrw proc2mngr, x0 > 0
  """

  run_perf_test( cmdline_opts, prog, 8, 8 )

#-------------------------------------------------------------------------
# LW Test
#-------------------------------------------------------------------------

def test_perf10( cmdline_opts ):

  prog="""                          #                   1 1 1 1 1 1
                                    # 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5
    addi x3, x0, 1                  # F D X M W
    lw   x3, 0(x1)                  #   F D X M W
    addi x4, x3, 0                  #     F D D X M W

    # need final CSRW to stop the test
    csrw proc2mngr, x0 > 0
  """

  run_perf_test( cmdline_opts, prog, 8, 8 )

#-------------------------------------------------------------------------
# Branch Test
#-------------------------------------------------------------------------

def test_perf11( cmdline_opts ):

  prog="""
                            # 1 2 3 4 5 6 7 8 9
    addi  x1, x0, 1         # F D X M W
    bgeu  x1, x2, label_a   #   F D X M W
    addi  x3, x3, 1         #     F D - - -
    addi  x3, x3, 4         #       F - - - -
    nop                     #
    nop                     #
    nop                     #
    nop                     #
    nop                     #

  label_a:
    addi  x3, x3, 2         #         F D X M W

    # need final CSRW to stop the test
    csrw proc2mngr, x0 > 0
  """

  run_perf_test( cmdline_opts, prog, 9, 9 )

#-------------------------------------------------------------------------
# JAL Test
#-------------------------------------------------------------------------

def test_perf12( cmdline_opts ):

  prog="""
                            # 1 2 3 4 5 6 7 8 9
    addi  x1, x0, 1         # F D X M W
    jal   x1, label_a       #   F D X M W
    addi  x3, x3, 1         #     F - - - -
    nop                     #
    nop                     #
    nop                     #
    nop                     #
    nop                     #

  label_a:
    addi  x3, x3, 2         #       F D X M W

    # need final CSRW to stop the test
    csrw proc2mngr, x0 > 0
  """

  run_perf_test( cmdline_opts, prog, 8, 8 )

#-------------------------------------------------------------------------
# JALR Test
#-------------------------------------------------------------------------

def test_perf13( cmdline_opts ):

  prog="""                    #                   1
                              # 1 2 3 4 5 6 7 8 9 0
    lui x1,      %hi[label_a] # F D X M W
    addi x1, x1, %lo[label_a] #   F D X M W
    jalr x31, x1, 0           #     F D X M W
    addi x3, x3, 1            #       F D - - -
    addi x4, x3, 1            #         F - - - -
    nop                       #
    nop                       #
    nop                       #
    nop                       #
    nop                       #

  label_a:
    addi  x3, x3, 2           #           F D X M W

    # need final CSRW to stop the test
    csrw proc2mngr, x0 > 0
  """

  run_perf_test( cmdline_opts, prog, 10, 10 )

#-------------------------------------------------------------------------
# MUL Test
#-------------------------------------------------------------------------
# Modified for single cycle multiplier -cbatten

def test_perf14( cmdline_opts ):

  # Pipeline diagram shows longest cycle count
  prog="""
                                    # 1 2 3 4 5 6 7 8
    csrr x1, mngr2proc < 0xffffffff # F D X M W
    csrr x2, mngr2proc < 0xffffffff #   F D X M W
    mul  x3, x1, x2                 #     F D X M W
    addi x4, x3, 1                  #       F D X M W
    # need final CSRW to stop the test
    csrw proc2mngr, x0 > 0
  """
  # amount of cycles depend on imul implementation
  run_perf_test( cmdline_opts, prog, 8, 8 )

