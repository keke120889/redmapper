from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import unittest
import numpy.testing as testing
import numpy as np
import fitsio

from redmapper import ColorBackground

class ColorBackgroundTestCase(unittest.TestCase):
    def runTest(self):

        file_name = 'test_dr8_col_bkg.fit'
        file_path = 'data_for_tests'

        cbkg = ColorBackground('%s/%s' % (file_path, file_name))

        col1 = np.array([0.572300, 1.39560])
        col2 = np.array([0.7894, 0.9564])
        refmags = np.array([17.587, 18.956])

        refmagindex = np.array([258, 395])
        col1index = np.array([1, 17])
        col2index = np.array([15, 19])

        idl_bkg1 = np.array([0.148366, 0.165678])
        idl_bkg2 = np.array([0.00899471, 0.0201531])
        idl_bkg12 = np.array([0.0111827, 0.0719981])

        # Test color1
        py_outputs = cbkg.lookup_diagonal(1, col1, refmags)
        testing.assert_almost_equal(py_outputs, idl_bkg1, decimal=5)

        # Test color2
        py_outputs = cbkg.lookup_diagonal(2, col2, refmags)
        testing.assert_almost_equal(py_outputs, idl_bkg2, decimal=5)

        # Test off-diagonal
        py_outputs = cbkg.lookup_offdiag(1, 2, col1, col2, refmags)
        testing.assert_almost_equal(py_outputs, idl_bkg12, decimal=5)

