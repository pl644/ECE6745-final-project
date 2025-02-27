#=========================================================================
# ProcFL_mix_test.py
#=========================================================================
# We group all our test cases into a class so that we can easily reuse
# these test cases in our RTL tests. We can simply inherit from this test
# class, overload the setup_class method, and set the ProcType
# appropriately.

import pytest

from pymtl3 import *
from proc.test.harness import asm_test, run_test
from proc.ProcFL import ProcFL

from proc.test import inst_mix_beq_jal
from proc.test import inst_mix_mul_mem
from proc.test import inst_mix

#-------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------

@pytest.mark.usefixtures("cmdline_opts")
class Tests:

  @classmethod
  def setup_class( cls ):
    cls.ProcType = ProcFL

  #-----------------------------------------------------------------------
  # mix_jal_beq
  #-----------------------------------------------------------------------

  @pytest.mark.parametrize( "name,test", [
    asm_test( inst_mix_beq_jal.gen_beq_jal_test      ),
    asm_test( inst_mix_beq_jal.gen_beq_nop_jal_test  ),
    asm_test( inst_mix_beq_jal.gen_beq_jalr_test     ),
    asm_test( inst_mix_beq_jal.gen_beq_nop_jalr_test ),
    asm_test( inst_mix_beq_jal.gen_jalr_jal_test     ),
    asm_test( inst_mix_beq_jal.gen_jalr_nop_jal_test ),
    asm_test( inst_mix_beq_jal.gen_many_beq_jal_test ),
  ])
  def test_jal_beq( s, name, test ):
    run_test( s.ProcType, test, cmdline_opts=s.__class__.cmdline_opts )

  def test_jal_beq_delays( s ):
    run_test( s.ProcType, inst_mix_beq_jal.gen_many_beq_jal_test, delays=True,
              cmdline_opts=s.__class__.cmdline_opts )

  #-----------------------------------------------------------------------
  # mix_mul_mem
  #-----------------------------------------------------------------------

  @pytest.mark.parametrize( "name,test", [
    asm_test( inst_mix_mul_mem.gen_basic_test     ),
    asm_test( inst_mix_mul_mem.gen_more_test      ),
  ])
  def test_mul_mem( s, name, test ):
    run_test( s.ProcType, test, cmdline_opts=s.__class__.cmdline_opts )

  def test_mul_mem_delays( s ):
    run_test( s.ProcType, inst_mix_mul_mem.gen_more_test, delays=True,
              cmdline_opts=s.__class__.cmdline_opts )

  #-----------------------------------------------------------------------
  # mix
  #-----------------------------------------------------------------------

  @pytest.mark.parametrize( "name,test", [
    asm_test( inst_mix.gen_mix_test     ),
  ])
  def test_mix( s, name, test ):
    run_test( s.ProcType, test, cmdline_opts=s.__class__.cmdline_opts )

  def test_mix_delays( s ):
    run_test( s.ProcType, inst_mix.gen_mix_test, delays=True,
              cmdline_opts=s.__class__.cmdline_opts )

