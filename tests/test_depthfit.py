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

from redmapper import DepthLim
from redmapper import Configuration
from redmapper import GalaxyCatalog
from redmapper import get_mask
from redmapper.depth_fitting import calcErrorModel

class DepthFitTestCase(unittest.TestCase):
    def test_depthfit(self):
        """
        Test depth fitting
        """

        file_path = "data_for_tests"
        conf_filename = "testconfig.yaml"
        config = Configuration(file_path + "/" + conf_filename)

        gals = GalaxyCatalog.from_galfile(config.galfile)

        np.random.seed(seed=12345)

        # First creation of depth object
        dlim = DepthLim(gals.refmag, gals.refmag_err)

        # This has been inspected that it makes sense.
        # Also, one should really be using a depth map

        testing.assert_almost_equal(dlim.initpars['EXPTIME'], 104.782066, 0)
        testing.assert_almost_equal(dlim.initpars['LIMMAG'], 20.64819717, 0)

        # And take a subpixel

        gals = GalaxyCatalog.from_galfile(config.galfile, hpix=8421, nside=128)

        config.mask_mode = 0
        mask = get_mask(config)
        mask.read_maskgals(config.maskgalfile)

        dlim.calc_maskdepth(mask.maskgals, gals.refmag, gals.refmag_err)

        pars, fail = calcErrorModel(gals.refmag, gals.refmag_err, calcErr=False)

        testing.assert_almost_equal(pars['EXPTIME'], 63.73879623, 0)
        testing.assert_almost_equal(pars['LIMMAG'], 20.68231583, 0)

        # And make sure the maskgals are getting the right constant value
        testing.assert_almost_equal(pars['EXPTIME'], mask.maskgals.exptime.min())
        testing.assert_almost_equal(pars['EXPTIME'], mask.maskgals.exptime.max())
        testing.assert_almost_equal(pars['LIMMAG'], mask.maskgals.limmag.min())
        testing.assert_almost_equal(pars['LIMMAG'], mask.maskgals.limmag.max())


if __name__=='__main__':
    unittest.main()
