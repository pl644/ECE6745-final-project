#=========================================================================
# ProcFL_jump_test.py
#=========================================================================
# We group all our test cases into a class so that we can easily reuse
# these test cases in our RTL tests. We can simply inherit from this test
# class, overload the setup_class method, and set the ProcType
# appropriately.

import pytest

from pymtl3 import *
from proc.test.harness import asm_test, run_test
from proc.ProcFL import ProcFL

from proc.test import inst_jal
from proc.test import inst_jalr

#-------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------

@pytest.mark.usefixtures("cmdline_opts")
class Tests:

  @classmethod
  def setup_class( cls ):
    cls.ProcType = ProcFL

  #-----------------------------------------------------------------------
  # jal
  #-----------------------------------------------------------------------

  @pytest.mark.parametrize( "name,test", [
    asm_test( inst_jal.gen_basic_test        ) ,

    # ''' LAB TASK '''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Add more rows to the test case table to test more complicated
    # scenarios.
    # ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''\/

    asm_test( inst_jal.gen_link_dep_test     ) ,
    asm_test( inst_jal.gen_jump_test         ) ,
    asm_test( inst_jal.gen_back_to_back_test ) ,
    asm_test( inst_jal.gen_value_test_0      ) ,
    asm_test( inst_jal.gen_value_test_1      ) ,
    asm_test( inst_jal.gen_value_test_2      ) ,
    asm_test( inst_jal.gen_value_test_3      ) ,
    asm_test( inst_jal.gen_jal_stall_test    ) ,

    #'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''/\
  ])

  def test_jal( s, name, test ):
    run_test( s.ProcType, test, cmdline_opts=s.__class__.cmdline_opts )

  # ''' LAB TASK '''''''''''''''''''''''''''''''''''''''''''''''''''''''''
  # random stall and delay
  # ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''\/

  def test_jal_delays( s ):
    run_test( s.ProcType, inst_jal.gen_jump_test, delays=True,
              cmdline_opts=s.__class__.cmdline_opts )

  #'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''/\

  #-----------------------------------------------------------------------
  # jalr
  #-----------------------------------------------------------------------

  @pytest.mark.parametrize( "name,test", [
    asm_test( inst_jalr.gen_basic_test    ),

    # ''' LAB TASK '''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Add more rows to the test case table to test more complicated
    # scenarios.
    # ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''\/

    asm_test( inst_jalr.gen_link_dep_test    ),
    asm_test( inst_jalr.gen_jump_test        ),
    asm_test( inst_jalr.gen_lsb_test         ),
    asm_test( inst_jalr.gen_value_test_0     ),
    asm_test( inst_jalr.gen_value_test_1     ),
    asm_test( inst_jalr.gen_value_test_2     ),
    asm_test( inst_jalr.gen_value_test_3     ),

    #'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''/\
  ])

  def test_jalr( s, name, test ):
    run_test( s.ProcType, test, cmdline_opts=s.__class__.cmdline_opts )

  # ''' LAB TASK '''''''''''''''''''''''''''''''''''''''''''''''''''''''''
  # random stall and delay
  # ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''\/

  def test_jalr_delays( s ):
    run_test( s.ProcType, inst_jalr.gen_jump_test, delays=True,
              cmdline_opts=s.__class__.cmdline_opts )

  #'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''/\

