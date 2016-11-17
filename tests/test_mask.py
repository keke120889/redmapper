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

        Next test the fracgood array and its features, including
        the fracgood_float and various fracgood entries.

        Next test to see if the mask has maskgals.

        Next test the compute_radmask() function which
        finds if an array of (RA,DEC) pairs are in or out
        of the mask. TODO

        Next test the set_radmask() function. TODO
        """
        file_path = "data_for_tests"
        conf_filename = "testconfig.yaml"
        confstr = Configuration(file_path + "/" + conf_filename)
        
        # Healpix mask
        mask = HPMask(confstr)

        # First test the healpix configuration
        testing.assert_equal(mask.nside,2048)
        testing.assert_equal(mask.offset,2100800)
        testing.assert_equal(mask.npix,548292)

        # Next test that the fracgood is working properly
        # indices = [] #some set of pixels #TODO
        # fracgoods = np.array([]) #known fracgoods at indices #TODO
        testing.assert_equal(mask.fracgood_float,1)
        testing.assert_equal(mask.fracgood.shape[0],548292)
        #testing.assert_equal(mask.fracgood[indices],fracgoods)

        # Tom - I believe the fracgood only has 0s and 1s, yes?
        print mask.fracgood.shape
        print mask.fracgood_float

        # See if the mask has maskgals
        testing.assert_equal(hasattr(mask,'maskgals'),True)

        # Next test the compute_radmask() function
        #RAs  = np.array([some values])
        #DECs = np.array([some values])
        #booleans = np.array([known boolean values for RA/DECs])
        #testing.assert_equal(mask.compute_radmask(RAs,DECs),booleans)

        # Next test the set_radmask() function
        #cluster = make a cluster
        #mpscale = something
        #How do I test to see if the function executed at all?
        #See if the mask.maskgals['MASKED'] attribute exists
        #See that the maskgals shape has the same shape as the 
        #cluster RA and DECs.

        # Other masks below... TODO

if __name__=='__main__':
    unittest.main()
