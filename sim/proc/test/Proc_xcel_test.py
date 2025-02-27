#=========================================================================
# Proc_xcel_test.py
#=========================================================================
# It is as simple as inheriting from FL tests and change the ProcType.

from proc.Proc import Proc
from proc.test.ProcFL_xcel_test import Tests as ProcFL_xcel_TestsBaseClass

#-------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------

class Tests( ProcFL_xcel_TestsBaseClass ):

  @classmethod
  def setup_class( cls ):
    cls.ProcType = Proc

