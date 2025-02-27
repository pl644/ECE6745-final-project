#=========================================================================
# SortXcel_test
#=========================================================================

import pytest

from lab2_xcel.test.SortXcelFL_test import test_case_table, run_test, run_test_multiple
from lab2_xcel.SortXcel import SortXcel

@pytest.mark.parametrize( **test_case_table )
def test( cmdline_opts, test_params ):
  run_test( SortXcel(), cmdline_opts, test_params )

def test_multiple( cmdline_opts ):
  run_test_multiple( SortXcel(), cmdline_opts )

