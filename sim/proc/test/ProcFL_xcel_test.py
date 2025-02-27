#=========================================================================
# ProcFL_xcel_test.py
#=========================================================================

import pytest

from pymtl3            import *
from proc.test.harness import asm_test, run_test
from proc.ProcFL       import ProcFL

from proc.test import inst_xcel

#-------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------

@pytest.mark.usefixtures("cmdline_opts")
class Tests:

  @classmethod
  def setup_class( cls ):
    cls.ProcType = ProcFL

  #-------------------------------------------------------------------------
  # csr
  #-------------------------------------------------------------------------

  @pytest.mark.parametrize( "name,test", [
    asm_test( inst_xcel.gen_basic_test  ),
    asm_test( inst_xcel.gen_bypass_test ),
    asm_test( inst_xcel.gen_random_test ),
  ])
  def test_csr( s, name, test ):
    run_test( s.ProcType, test, cmdline_opts=s.__class__.cmdline_opts )

  def test_csr_delays( s ):
    run_test( s.ProcType, inst_xcel.gen_random_test, delays=True,
              cmdline_opts=s.__class__.cmdline_opts )

