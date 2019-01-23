from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import unittest
import numpy.testing as testing
import numpy as np
import fitsio
from numpy import random

from redmapper import HPMask
from redmapper import Configuration
from redmapper.utilities import apply_errormodels

class ApplyErrormodelsTestCase(unittest.TestCase):
    """
    Test the apply_errormodels() function in mask.py.
    """
    def runTest(self):
        """
        Run the apply_errormodels() test.
        """
        file_path = 'data_for_tests'
        conf_filename = 'testconfig.yaml'
        config = Configuration(file_path + '/' + conf_filename)

        mask = HPMask(config) #Create the mask
        maskgal_index = mask.select_maskgals_sample()

        #set all the necessary inputs from test file
        mask.maskgals.exptime = 100.
        mask.maskgals.limmag  = 20.
        mask.maskgals.zp[0]   = 22.5
        mask.maskgals.nsig[0] = 10.
        #necessary as mask.maskgals.exptime has shape (6000,)
        mag_in                = np.full(6000, 1, dtype = float)
        mag_in[:6]            = np.array([16., 17., 18., 19., 20., 21.])

        #test without noise
        mag, mag_err = apply_errormodels(mask.maskgals, mag_in, nonoise = True)
        idx = np.array([0, 1, 2, 3, 4, 5])
        mag_idl     = np.array([16., 17., 18., 19., 20., 21.])
        mag_err_idl = np.array([0.00602535, 0.0107989, 0.0212915, 0.0463765, 0.108574, 0.264390])
        testing.assert_almost_equal(mag[idx], mag_idl)
        testing.assert_almost_equal(mag_err[idx], mag_err_idl, decimal = 6)

        #test with noise and set seed
        seed = 0
        random.seed(seed = seed)
        mag, mag_err = apply_errormodels(mask.maskgals, mag_in)

        idx = np.array([0, 1, 2, 3, 4, 5, 1257, 2333, 3876])
        mag_test = np.array([15.98942267, 16.99568733, 17.97935868, 
                             18.90075284, 19.81409659, 21.29508236,  
                             0.99999373,  1.00000663,  1.00000807])
        mag_err_test = np.array([5.96693051e-03, 1.07560575e-02, 2.08905241e-02, 
                                 4.23251692e-02, 9.14877522e-02, 3.46958444e-01,
                                 5.44154045e-06, 5.44160510e-06, 5.44161230e-06])
        testing.assert_almost_equal(mag[idx], mag_test)
        testing.assert_almost_equal(mag_err[idx], mag_err_test)

if __name__=='__main__':
    unittest.main()
