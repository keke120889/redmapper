import unittest
import numpy.testing as testing
import numpy as np
import fitsio
from numpy import random

from redmapper.mask import HPMask
from redmapper.config import Configuration

class ClusterFiltersTestCase(unittest.TestCase):
    """
    Test the apply_errormodels() function in mask.py.
    """
    def runTest(self):
        file_path = 'data_for_tests'
        conf_filename = 'testconfig.yaml'
        confstr = Configuration(file_path + '/' + conf_filename)
        
        mask = HPMask(confstr) #Create the mask
        
        #set all the necessary inputs from test file
        #mask.maskgals.exptime = hdr['']
        #mask.maskgals.limmag  = hdr['']
        #mask.maskgals.zp[0]   = hdr['']
        #mask.maskgals.nsig[0] = hdr['']
        #mag_in                = hdr['']
        #
        ##test without noise
        #mag, mag_err = mask.apply_errormodels(mag_in, nonoise = True)
        #
        #testing.assert_almost_equal(mag, mag_idl)
        #testing.assert_almost_equal(mag_err, mag_err_idl)
        #
        ##test with noise and set seed
        #seed = 0
        #random.seed(seed = seed)
        #mag, mag_err = mask.apply_errormodels(mag_in)
        #
        #idx = np.array([])
        #mag_test = np.array([])
        #mag_err_test = np.array([])
        #testing.assert_almost_equal(mag, mag_test)
        #testing.assert_almost_equal(mag_err, mag_err_test)
        
if __name__=='__main__':
    unittest.main()