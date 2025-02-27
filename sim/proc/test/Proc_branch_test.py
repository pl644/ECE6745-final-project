#=========================================================================
# Proc_branch_test.py
#=========================================================================
# It is as simple as inheriting from FL tests and change the ProcType.

from proc.Proc import Proc
from proc.test.ProcFL_branch_test import Tests as ProcFL_branch_TestsBaseClass

#-------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------

class Tests( ProcFL_branch_TestsBaseClass ):

  @classmethod
  def setup_class( cls ):
    cls.ProcType = Proc

