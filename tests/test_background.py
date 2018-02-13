import unittest
import numpy.testing as testing
import numpy as np
import fitsio

from redmapper.background import Background
from redmapper.background import ZredBackground

class BackgroundTestCase(unittest.TestCase):

    def runTest(self):
        """
        Part one of this file checks that the background file
        exists and is formatted correctly.

        Then we load the background file and pull out some sources
        in the background, and compare the IDL outputs to thoe
        of the sigma_g attribute of the background. The initialization
        of the sigma_g values as well as the 'lookup' or spline used
        over these valus is also tested. Note that the background
        is characterized by sigma_g as functions of (z,chisq,refmag).
        """
        file_name, file_path = 'test_bkg.fit', 'data_for_tests'
        # test that we fail if we try a non-existent file
        self.assertRaises(IOError, Background, 'nonexistent.fit')
        # test that we fail if we read a non-fits file
        self.assertRaises(IOError, Background,'%s/testconfig.yaml' % (file_path))
        # test that we fail if we try a file without the right header info
        self.assertRaises(IOError, Background,
                          '%s/test_dr8_pars.fit' % (file_path))
        bkg = Background('%s/%s' % (file_path, file_name))

        # test creation of lookup table
        inputs = [(172,15,64), (323,3,103), (9,19,21), (242,4,87),
                  (70,12,58), (193,6,39), (87,14,88), (337,5,25), (333,8,9)]
        py_outputs = np.array([bkg.sigma_g[idx] for idx in inputs])
        idl_outputs = np.array([0.32197464, 6.4165196, 0.0032830855, 
                                1.4605126, 0.0098356586, 0.79848081, 
                                0.011284498, 9.3293247, 8.7064905])
        testing.assert_almost_equal(py_outputs, idl_outputs, decimal=1)

        # test functionality of lookup table
        z = 0.23185321
        chisq = np.array([0.13315917, 3.57059131, 3.71567741, 2.46307987,
                          9.16647519, 8.24240144, -1., 1.19503491])
        refmag = np.array([1000., 15.05129281, 16.81049236, 18.07566359,
                        19.88279, 15.56617587, 18.55626717, 15.00271158])

        py_outputs = bkg.sigma_g_lookup(z, chisq, refmag)
        idl_outputs = np.array([np.inf, 0.0012997627, 0.56412143, 6.4126010, 
                                43.4550, 0.012194233, np.inf, 0.0])
        # idl_ouputs[4] = 42.555183
        testing.assert_almost_equal(py_outputs, idl_outputs, decimal=4)

        ###########################################
        ## And test the zred background code
        ###########################################

        zredbkg = ZredBackground('%s/%s' % (file_path, file_name))

        # test creation of lookup table
        inputs = [(60, 50), (200, 100), (300, 120)]
        py_outputs = np.array([zredbkg.sigma_g[idx] for idx in inputs])
        idl_outputs = np.array([1.16810, 28.4379, 373.592])

        testing.assert_almost_equal(py_outputs, idl_outputs, decimal=4)

        # test functionality of lookup table
        zred = np.array([0.2154, 0.2545, 0.2876])
        refmag = np.array([18.015,18.576,19.234])
        idl_outputs = np.array([710.17102,1000.1127,1718.0394])
        py_outputs = zredbkg.sigma_g_lookup(zred, refmag)
        testing.assert_almost_equal(py_outputs, idl_outputs, decimal=3)


if __name__=='__main__':
        unittest.main()
