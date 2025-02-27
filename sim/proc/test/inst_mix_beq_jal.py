#=========================================================================
# mix jumps and branches
#=========================================================================

from pymtl3 import *
from proc.test.inst_utils import *

#-------------------------------------------------------------------------
# gen_beq_jal_test
#-------------------------------------------------------------------------
# Check that a branch followed by a jump does not execute the jump.

def gen_beq_jal_test():
  return """
    beq  x0, x0, success
    jal  x0, fail

    nop
    nop
    nop
    nop

  fail:
    csrw proc2mngr, x0 # will send wrong value to sink

  success:
    addi x2, x0, 1
    csrw proc2mngr, x2 > 1

  """

#-------------------------------------------------------------------------
# gen_beq_nop_jal_test
#-------------------------------------------------------------------------
# Check that a branch followed by a nop/jump does not execute the jump.

def gen_beq_nop_jal_test():
  return """
    beq  x0, x0, success
    nop
    jal  x0, fail

    nop
    nop
    nop
    nop

  fail:
    csrw proc2mngr, x0 # will send wrong value to sink

  success:
    addi x2, x0, 1
    csrw proc2mngr, x2 > 1

  """

#-------------------------------------------------------------------------
# gen_beq_jalr_test
#-------------------------------------------------------------------------
# Check that a branch followed by a jump does not execute the jump.

def gen_beq_jalr_test():
  return """
    lui  x1,     %hi[fail]
    addi x1, x1, %lo[fail]

    beq  x0, x0, success
    jalr x0, x1, 0

    nop
    nop
    nop
    nop

  fail:
    csrw proc2mngr, x0 # will send wrong value to sink

  success:
    addi x2, x0, 1
    csrw proc2mngr, x2 > 1

  """

#-------------------------------------------------------------------------
# gen_beq_nop_jalr_test
#-------------------------------------------------------------------------
# Check that a branch followed by a nop/jump does not execute the jump.

def gen_beq_nop_jalr_test():
  return """
    lui  x1,     %hi[fail]
    addi x1, x1, %lo[fail]

    beq  x0, x0, success
    nop
    jalr x0, x1, 0

    nop
    nop
    nop
    nop

  fail:
    csrw proc2mngr, x0 # will send wrong value to sink

  success:
    addi x2, x0, 1
    csrw proc2mngr, x2 > 1

  """

#-------------------------------------------------------------------------
# gen_jalr_jal_test
#-------------------------------------------------------------------------
# Check that a jalr followed by a jump does not execute the jump.

def gen_jalr_jal_test():
  return """
    lui  x1,     %hi[success]
    addi x1, x1, %lo[success]

    jalr x0, x1, 0
    jal  x0, fail

    nop
    nop
    nop
    nop

  fail:
    csrw proc2mngr, x0 # will send wrong value to sink

  success:
    addi x2, x0, 1
    csrw proc2mngr, x2 > 1

  """

#-------------------------------------------------------------------------
# gen_jalr_nop_jal_test
#-------------------------------------------------------------------------
# Check that a jalr followed by a nop/jump does not execute the jump.

def gen_jalr_nop_jal_test():
  return """
    lui  x1,     %hi[success]
    addi x1, x1, %lo[success]

    jalr x0, x1, 0
    nop
    jal  x0, fail

    nop
    nop
    nop
    nop

  fail:
    csrw proc2mngr, x0 # will send wrong value to sink

  success:
    addi x2, x0, 1
    csrw proc2mngr, x2 > 1

  """

#-------------------------------------------------------------------------
# gen_many_beq_jal_test
#-------------------------------------------------------------------------
# Check that a branch followed by a jump does not execute the jump.

def gen_many_beq_jal_test():
  return """

    addi x1, x0, 10        # initialize loop counter to 10
    addi x2, x0, 0         # initialize accumulator to 1

  loop:
    beq  x0, x0, success   # skip over the fail target
    jal  x0, fail          # should never be executed

  fail:
    csrw proc2mngr, x0     # will send wrong value to sink

  success:
    addi x2, x2, 1         # increment accumulator
    addi x1, x1, -1        # decrement loop counter
    bne  x1, x0, loop      # loop 10 times

    csrw proc2mngr, x2 > 10
  """
