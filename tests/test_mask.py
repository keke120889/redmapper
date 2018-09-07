from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import unittest
import numpy.testing as testing
import numpy as np
import fitsio
import tempfile
import shutil
import os
from numpy import random

from redmapper import get_mask, Mask, HPMask
from redmapper import Configuration

class MaskTestCase(unittest.TestCase):
    def test_readmask(self):
        """
        Test reading of mask files
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

        # And test that we're getting the right numbers from a sub-mask
        config2 = Configuration(file_path + "/" + conf_filename)
        config2.d.hpix = 582972
        config2.d.nside = 1024
        config2.border = 0.02
        mask2 = get_mask(config2)

        comp = np.array([False, True, True, True, False])
        testing.assert_equal(mask2.compute_radmask(RAs, Decs), comp)

    def test_maskgals(self):
        """
        Test generation of maskgals file
        """

        # Note that due to historical reasons, this is testing the
        # new generation of maskgals with some spot checks.  Independently,
        # it has been checked that the distributions are the same as for
        # the old IDL code which was used to generate the reference used
        # in the cluster tests.

        file_path = "data_for_tests"
        conf_filename = "testconfig.yaml"
        config = Configuration(os.path.join(file_path, conf_filename))

        config.mask_mode = 0
        mask = get_mask(config)

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')
        config.outpath = self.test_dir

        maskgalfile = os.path.join(self.test_dir, 'testmaskgal.fit')

        random.seed(seed=12345)

        # This will generate the file if it isn't there
        mask.read_maskgals(maskgalfile)

        maskgals, hdr = fitsio.read(maskgalfile, ext=1, header=True)

        self.assertEqual(maskgals.size, config.maskgal_ngals)
        self.assertEqual(hdr['VERSION'], 5)
        self.assertEqual(hdr['R0'], config.percolation_r0)
        self.assertEqual(hdr['BETA'], config.percolation_beta)
        self.assertEqual(hdr['STEPSIZE'], config.maskgal_rad_stepsize)
        self.assertEqual(hdr['NMAG'], config.nmag)
        self.assertEqual(hdr['NGALS'], config.maskgal_ngals)
        self.assertEqual(hdr['CHISQMAX'], config.chisq_max)
        self.assertEqual(hdr['LVALREF'], config.lval_reference)
        self.assertEqual(hdr['EXTRA'], config.maskgal_dmag_extra)
        self.assertEqual(hdr['ALPHA'], config.calib_lumfunc_alpha)
        self.assertEqual(hdr['RSIG'], config.rsig)
        self.assertEqual(hdr['ZREDERR'], config.maskgal_zred_err)

        testing.assert_almost_equal(maskgals['r'][0: 3], [0.66900003, 0.119, 0.722])
        testing.assert_almost_equal(maskgals['phi'][0: 3], [1.73098969, 2.53610063, 4.2362957])
        testing.assert_almost_equal(maskgals['x'][0: 3], [-0.1067116, -0.09784444, -0.33090013])
        testing.assert_almost_equal(maskgals['m'][0: 3], [0.46200001, 1.778, -1.43599999])
        testing.assert_almost_equal(maskgals['chisq'][0: 3], [8.63599968, 2.28399992, 1.55799997])
        testing.assert_almost_equal(maskgals['cwt'][0: 3], [0.02877194, 0.1822518, 0.17872778])
        testing.assert_almost_equal(maskgals['nfw'][0: 3], [0.15366785, 0.32543495, 0.1454625])
        testing.assert_almost_equal(maskgals['dzred'][0: 3], [-0.03090504, 0.00847131, -0.01800639])
        testing.assert_almost_equal(maskgals['zwt'][0: 3], [6.0447073, 18.23568916, 13.30043507])
        testing.assert_almost_equal(maskgals['lumwt'][0: 3], [0.39371657, 0.62304342, 0.01774093])
        testing.assert_almost_equal(maskgals['theta_r'][0: 3, 3], [0.73237121, 1., 0.3299689])
        testing.assert_almost_equal(maskgals['radbins'][0, 0: 3], [0.40000001, 0.5, 0.60000002])
        testing.assert_almost_equal(maskgals['nin_orig'][0, 0: 3], [2213., 2663., 3066.])
        testing.assert_almost_equal(maskgals['nin'][0, 0: 3], [2203.3347168, 2651.54467773, 3062.44628906])

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)


if __name__=='__main__':
    unittest.main()
