import unittest
import numpy.testing as testing
import numpy as np
import fitsio

from redmapper.mask import HPMask
from redmapper.config import Configuration

class MaskTestCase(unittest.TestCase):
    def runTest(self):
        """
        This tests the mask.py module. Since this
        module is still in developement, so too are
        these unit tests.

        First test to see if nside, offset, and npix are 
        the values that we expect them to be. These are 
        variables that tell us about how healpix is formatted.

        Next test
        """
        file_path = "data_for_tests"
        conf_filename = "testconfig.yaml"
        confstr = Configuration(file_path + "/" + conf_filename)
        
        #Healpix mask
        mask = HPMask(confstr)

        #First test the healpix configuration
        testing.assert_equal(mask.nside,2048)
        testing.assert_equal(mask.offset,2100800)
        testing.assert_equal(mask.npix,548292)

        print mask.fracgood.shape

        #Other masks below... TODO

if __name__=='__main__':
    unittest.main()
