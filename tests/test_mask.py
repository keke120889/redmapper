import unittest
import numpy.testing as testing
import numpy as np
import fitsio

from redmapper.mask import Mask,HPMask
from redmapper.config import Configuration

class MaskTestCase(unittest.TestCase):
    def runTest(self):
        """
        This tests the mask.py module. Since this
        module is still in developement, so too are
        these unit tests.

        First test the red_maskgals() and gen_maskgals() functions.
        Note: the gen_maskgals() function isn't written yet, so it
        can't be tested. TODO

        Next create a HPMask, check that it has maskgals and then
        test to see if nside, offset, and npix are 
        the values that we expect them to be. These are 
        variables that tell us about how healpix is formatted.

        Next test the fracgood array and its features, including
        the fracgood_float and various fracgood entries.

        Next test the compute_radmask() function which
        finds if an array of (RA,DEC) pairs are in or out
        of the mask. TODO

        Next test the set_radmask() function. TODO

        """
        file_path = "data_for_tests"
        conf_filename = "testconfig.yaml"
        confstr = Configuration(file_path + "/" + conf_filename)

        # Test the read_maskgals() and the gen_maskgals() functions
        # of the Mask superclass.
        mask = Mask(confstr)
        mask.read_maskgals(confstr.maskgalfile)
        testing.assert_equal(hasattr(mask,'maskgals'),True)
        # Clear the maskgals attribute and then run gen_maskgals()
        #delattr(mask,'maskgals')
        #mask.gen_maskgals()
        testing.assert_equal(hasattr(mask,'maskgals'),True)
        
        # Healpix mask and see if it has maskgals
        mask = HPMask(confstr)
        testing.assert_equal(hasattr(mask,'maskgals'),True)

        # First test the healpix configuration
        testing.assert_equal(mask.nside,2048)
        testing.assert_equal(mask.offset,2100800)
        testing.assert_equal(mask.npix,548292)

        # Next test that the fracgood is working properly
        indices = [396440, 445445, 99547, 354028, 516163] #Random indices
        true_fracgoods = np.array([0,0,0.828125,0.796875,0.828125]) #known fracgoods at indices
        testing.assert_equal(mask.fracgood_float,1)
        testing.assert_equal(mask.fracgood.shape[0],548292)
        testing.assert_equal(mask.fracgood[indices],true_fracgoods)

        # Next test the compute_radmask() function
        # Note: RA and DECs are in degrees here
        RAs  = np.array([0.0,293.9,134.9,164.1,281.1,107.5])
        DECs = np.array([0.0,67.6,159.0,132.4,178.7,35.5]) - 90.0
        booleans = np.array([False,False,False,False,False,False])#known boolean values for RA/DECs
        testing.assert_equal(mask.compute_radmask(RAs,DECs),booleans)

        # Next test the set_radmask() function
        #cluster = make a cluster
        #mpscale = something
        #How do I test to see if the function executed at all?
        #See if the mask.maskgals['MASKED'] attribute exists
        #See that the maskgals shape has the same shape as the 
        #cluster RA and DECs.
        # actually, maskgals size will be 6000 always 
        #(which is the default config value at least).  
        #It's independent of the cluster itself -- just a list of random points.


        # Other masks below... TODO

if __name__=='__main__':
    unittest.main()
