#=========================================================================
# simple_test.py
#=========================================================================
# This is primarily just for playing around with little assembly code
# sequences.

from proc.test.harness import asm_test, run_test
from proc.ProcFL       import ProcFL
from proc.Proc         import Proc

def test( cmdline_opts ):

  prog="""
    csrr x1, mngr2proc < 5
    csrr x2, mngr2proc < 4
    nop
    nop
    nop
    nop
    nop
    nop
    nop
    nop
    add x3, x1, x2
    nop
    nop
    nop
    nop
    nop
    nop
    nop
    nop
    csrw proc2mngr, x3 > 9
    nop
    nop
    nop
    nop
    nop
    nop
    nop
    nop
  """

  run_test( ProcFL, prog, cmdline_opts=cmdline_opts )

