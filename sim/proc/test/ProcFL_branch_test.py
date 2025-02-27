#=========================================================================
# ProcFL_branch_test.py
#=========================================================================
# We group all our test cases into a class so that we can easily reuse
# these test cases in our RTL tests. We can simply inherit from this test
# class, overload the setup_class method, and set the ProcType
# appropriately.

import pytest

from pymtl3 import *
from proc.test.harness import asm_test, run_test
from proc.ProcFL import ProcFL

from proc.test import inst_beq
from proc.test import inst_bne
from proc.test import inst_bge
from proc.test import inst_bgeu
from proc.test import inst_blt
from proc.test import inst_bltu

#-------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------

@pytest.mark.usefixtures("cmdline_opts")
class Tests:

  @classmethod
  def setup_class( cls ):
    cls.ProcType = ProcFL

  #-----------------------------------------------------------------------
  # beq
  #-----------------------------------------------------------------------

  @pytest.mark.parametrize( "name,test", [
    asm_test( inst_beq.gen_basic_test ) ,

    # ''' LAB TASK '''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Add more rows to the test case table to test more complicated
    # scenarios.
    # ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''\/

    asm_test( inst_beq.gen_src0_dep_taken_test    ),
    asm_test( inst_beq.gen_src0_dep_nottaken_test ),
    asm_test( inst_beq.gen_src1_dep_taken_test    ),
    asm_test( inst_beq.gen_src1_dep_nottaken_test ),
    asm_test( inst_beq.gen_srcs_dep_taken_test    ),
    asm_test( inst_beq.gen_srcs_dep_nottaken_test ),
    asm_test( inst_beq.gen_src0_eq_src1_test      ),
    asm_test( inst_beq.gen_value_test             ),
    asm_test( inst_beq.gen_random_test            ),
    asm_test( inst_beq.gen_back_to_back_test      ),

    #'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''/\
  ])
  def test_beq( s, name, test ):
    run_test( s.ProcType, test, cmdline_opts=s.__class__.cmdline_opts )

  # ''' LAB TASK '''''''''''''''''''''''''''''''''''''''''''''''''''''''''
  # random stall and delay
  # ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''\/

  def test_beq_delays( s ):
    run_test( s.ProcType, inst_beq.gen_random_test, delays=True,
              cmdline_opts=s.__class__.cmdline_opts )

  #'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''/\

  #-----------------------------------------------------------------------
  # bne
  #-----------------------------------------------------------------------

  @pytest.mark.parametrize( "name,test", [
    asm_test( inst_bne.gen_basic_test             ),
    asm_test( inst_bne.gen_src0_dep_taken_test    ),
    asm_test( inst_bne.gen_src0_dep_nottaken_test ),
    asm_test( inst_bne.gen_src1_dep_taken_test    ),
    asm_test( inst_bne.gen_src1_dep_nottaken_test ),
    asm_test( inst_bne.gen_srcs_dep_taken_test    ),
    asm_test( inst_bne.gen_srcs_dep_nottaken_test ),
    asm_test( inst_bne.gen_src0_eq_src1_test      ),
    asm_test( inst_bne.gen_value_test             ),
    asm_test( inst_bne.gen_random_test            ),
    asm_test( inst_bne.gen_back_to_back_test      ),
  ])
  def test_bne( s, name, test ):
    run_test( s.ProcType, test, cmdline_opts=s.__class__.cmdline_opts )

  # ''' LAB TASK '''''''''''''''''''''''''''''''''''''''''''''''''''''''''
  # random stall and delay
  # ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''\/

  def test_bne_delays( s ):
    run_test( s.ProcType, inst_bne.gen_random_test, delays=True,
              cmdline_opts=s.__class__.cmdline_opts )

  #'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''/\
  #-----------------------------------------------------------------------
  # bge
  #-----------------------------------------------------------------------

  @pytest.mark.parametrize( "name,test", [
    asm_test( inst_bge.gen_basic_test             ),

    # ''' LAB TASK '''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Add more rows to the test case table to test more complicated
    # scenarios.
    # ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''\/

    asm_test( inst_bge.gen_src0_dep_taken_test    ),
    asm_test( inst_bge.gen_src0_dep_nottaken_test ),
    asm_test( inst_bge.gen_src1_dep_taken_test    ),
    asm_test( inst_bge.gen_src1_dep_nottaken_test ),
    asm_test( inst_bge.gen_srcs_dep_taken_test    ),
    asm_test( inst_bge.gen_srcs_dep_nottaken_test ),
    asm_test( inst_bge.gen_src0_eq_src1_test      ),
    asm_test( inst_bge.gen_value_test             ),
    asm_test( inst_bge.gen_random_test            ),
    asm_test( inst_bge.gen_back_to_back_test      ),

    #'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''/\
  ])
  def test_bge( s, name, test ):
    run_test( s.ProcType, test, cmdline_opts=s.__class__.cmdline_opts )

  # ''' LAB TASK '''''''''''''''''''''''''''''''''''''''''''''''''''''''''
  # random stall and delay
  # ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''\/

  def test_bge_delays( s ):
    run_test( s.ProcType, inst_bge.gen_random_test, delays=True,
              cmdline_opts=s.__class__.cmdline_opts )

  #'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''/\

  #-----------------------------------------------------------------------
  # bgeu
  #-----------------------------------------------------------------------

  @pytest.mark.parametrize( "name,test", [
    asm_test( inst_bgeu.gen_basic_test             ),

    # ''' LAB TASK '''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Add more rows to the test case table to test more complicated
    # scenarios.
    # ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''\/

    asm_test( inst_bgeu.gen_src0_dep_taken_test    ),
    asm_test( inst_bgeu.gen_src0_dep_nottaken_test ),
    asm_test( inst_bgeu.gen_src1_dep_taken_test    ),
    asm_test( inst_bgeu.gen_src1_dep_nottaken_test ),
    asm_test( inst_bgeu.gen_srcs_dep_taken_test    ),
    asm_test( inst_bgeu.gen_srcs_dep_nottaken_test ),
    asm_test( inst_bgeu.gen_src0_eq_src1_test      ),
    asm_test( inst_bgeu.gen_value_test             ),
    asm_test( inst_bgeu.gen_random_test            ),
    asm_test( inst_bgeu.gen_back_to_back_test      ),

    #'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''/\
  ])
  def test_bgeu( s, name, test ):
    run_test( s.ProcType, test, cmdline_opts=s.__class__.cmdline_opts )

  # ''' LAB TASK '''''''''''''''''''''''''''''''''''''''''''''''''''''''''
  # random stall and delay
  # ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''\/

  def test_bgeu_delays( s ):
    run_test( s.ProcType, inst_bgeu.gen_random_test, delays=True,
              cmdline_opts=s.__class__.cmdline_opts )

  #'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''/\

  #-----------------------------------------------------------------------
  # blt
  #-----------------------------------------------------------------------

  @pytest.mark.parametrize( "name,test", [
    asm_test( inst_blt.gen_basic_test             ),

    # ''' LAB TASK '''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Add more rows to the test case table to test more complicated
    # scenarios.
    # ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''\/

    asm_test( inst_blt.gen_src0_dep_taken_test    ),
    asm_test( inst_blt.gen_src0_dep_nottaken_test ),
    asm_test( inst_blt.gen_src1_dep_taken_test    ),
    asm_test( inst_blt.gen_src1_dep_nottaken_test ),
    asm_test( inst_blt.gen_srcs_dep_taken_test    ),
    asm_test( inst_blt.gen_srcs_dep_nottaken_test ),
    asm_test( inst_blt.gen_src0_eq_src1_test      ),
    asm_test( inst_blt.gen_value_test             ),
    asm_test( inst_blt.gen_random_test            ),
    asm_test( inst_blt.gen_back_to_back_test      ),

    #'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''/\
  ])
  def test_blt( s, name, test ):
    run_test( s.ProcType, test, cmdline_opts=s.__class__.cmdline_opts )

  # ''' LAB TASK '''''''''''''''''''''''''''''''''''''''''''''''''''''''''
  # random stall and delay
  # ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''\/

  def test_blt_delays( s ):
    run_test( s.ProcType, inst_blt.gen_random_test, delays=True,
              cmdline_opts=s.__class__.cmdline_opts )

  #'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''/\

  #-----------------------------------------------------------------------
  # bltu
  #-----------------------------------------------------------------------

  @pytest.mark.parametrize( "name,test", [
    asm_test( inst_bltu.gen_basic_test             ),

    # ''' LAB TASK '''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Add more rows to the test case table to test more complicated
    # scenarios.
    # ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''\/

    asm_test( inst_bltu.gen_src0_dep_taken_test    ),
    asm_test( inst_bltu.gen_src0_dep_nottaken_test ),
    asm_test( inst_bltu.gen_src1_dep_taken_test    ),
    asm_test( inst_bltu.gen_src1_dep_nottaken_test ),
    asm_test( inst_bltu.gen_srcs_dep_taken_test    ),
    asm_test( inst_bltu.gen_srcs_dep_nottaken_test ),
    asm_test( inst_bltu.gen_src0_eq_src1_test      ),
    asm_test( inst_bltu.gen_value_test             ),
    asm_test( inst_bltu.gen_random_test            ),
    asm_test( inst_bltu.gen_back_to_back_test      ),

    #'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''/\
  ])
  def test_bltu( s, name, test ):
    run_test( s.ProcType, test, cmdline_opts=s.__class__.cmdline_opts )

  # ''' LAB TASK '''''''''''''''''''''''''''''''''''''''''''''''''''''''''
  # random stall and delay
  # ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''\/

  def test_bltu_delays( s ):
    run_test( s.ProcType, inst_bltu.gen_random_test, delays=True,
              cmdline_opts=s.__class__.cmdline_opts )

  #'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''/\

