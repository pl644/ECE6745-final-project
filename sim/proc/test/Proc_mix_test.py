#=========================================================================
# Proc_mix_test.py
#=========================================================================
# It is as simple as inheriting from FL tests and change the ProcType.

from proc.Proc import Proc
from proc.test.ProcFL_mix_test import Tests as ProcFL_mix_TestsBaseClass

#-------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------

class Tests( ProcFL_mix_TestsBaseClass ):

  @classmethod
  def setup_class( cls ):
    cls.ProcType = Proc

