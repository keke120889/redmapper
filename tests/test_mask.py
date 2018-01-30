import unittest
import numpy.testing as testing
import numpy as np
import fitsio

#from redmapper.mask import Mask,HPMask
from redmapper.mask import get_mask, Mask, HPMask
from redmapper.configuration import Configuration

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
        config = Configuration(file_path + "/" + conf_filename)

        # First, test a bare mask

        config.mask_mode = 0
        mask = get_mask(config)
        testing.assert_equal(hasattr(mask, 'maskgals'), True)
        testing.assert_equal(isinstance(mask, Mask), True)
        testing.assert_equal(isinstance(mask, HPMask), False)

        # And the healpix mask

        config.mask_mode = 3
        mask = get_mask(config)
        testing.assert_equal(hasattr(mask, 'maskgals'), True)
        testing.assert_equal(isinstance(mask, Mask), True)
        testing.assert_equal(isinstance(mask, HPMask), True)

        # When ready, add in test of gen_maskgals()

        # Test the healpix configuration
        testing.assert_equal(mask.nside,2048)
        testing.assert_equal(mask.offset,2100800)
        testing.assert_equal(mask.npix,548292)

        # Next test that the fracgood is working properly
        indices = [396440, 445445, 99547, 354028, 516163] #Random indices
        true_fracgoods = np.array([0, 0, 0.828125, 0.796875, 0.828125]) #known fracgoods at indices
        testing.assert_equal(mask.fracgood_float, 1)
        testing.assert_equal(mask.fracgood.shape[0], 548292)
        testing.assert_equal(mask.fracgood[indices], true_fracgoods)

        # Next test the compute_radmask() function
        # Note: RA and DECs are in degrees here

        RAs = np.array([142.10934, 142.04090, 142.09242, 142.11448, 50.0])
        Decs = np.array([65.022666, 65.133844, 65.084844, 65.109541, 50.0])

        comp = np.array([True, True, True, True, False])

        testing.assert_equal(mask.compute_radmask(RAs, Decs), comp)

if __name__=='__main__':
    unittest.main()
