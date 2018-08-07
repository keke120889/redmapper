from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import matplotlib
matplotlib.use('Agg')

import unittest
import os
import shutil
import numpy.testing as testing
import numpy as np
import fitsio
import tempfile
from numpy import random

from redmapper.configuration import Configuration
from redmapper.calibration import WcenCalibrator

class CenteringCalibratorTestCase(unittest.TestCase):
    def test_centeringcal(self):
        file_path = 'data_for_tests'
        conf_filename = 'testconfig_wcen.yaml'
        config = Configuration(os.path.join(file_path, conf_filename))

        randcatfile = os.path.join(file_path, 'test_dr8_randcat.fit')
        randsatcatfile = os.path.join(file_path, 'test_dr8_randsatcat.fit')

        test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')
        config.outpath = test_dir

        config.wcenfile = os.path.join(config.outpath, '%s_testwcen.fit' % (config.d.outbase))

        # Get repeatability here
        random.seed(seed=1000)

        wc = WcenCalibrator(config, 1, randcatfile=randcatfile, randsatcatfile=randsatcatfile)
        wc.run(testing=True)

        # check outputs...

        # First, the schechter monte carlo.
        # These are very approximate, but checking for any unexpected changes
        testing.assert_almost_equal(wc.phi1_mmstar_m, -1.13451727, 5)
        testing.assert_almost_equal(wc.phi1_mmstar_slope, -0.37794289, 5)
        testing.assert_almost_equal(wc.phi1_msig_m, 0.49644922, 5)
        testing.assert_almost_equal(wc.phi1_msig_slope, -0.13314551, 5)

        # Make sure the output file is there...
        self.assertTrue(os.path.isfile(config.wcenfile))

        # Test the reading from the config.
        vals = config._wcen_vals()
        config._set_vars_from_dict(vals)

        testing.assert_almost_equal(config.wcen_Delta0, -1.4110523635495327, 5)
        testing.assert_almost_equal(config.wcen_Delta1, -0.32437965720455519, 5)
        testing.assert_almost_equal(config.wcen_sigma_m, 0.36547313080408494, 5)
        testing.assert_almost_equal(config.wcen_pivot, 30.0, 5)
        testing.assert_almost_equal(config.lnw_fg_mean, -0.26910934330307562, 5)
        testing.assert_almost_equal(config.lnw_fg_sigma, 0.29529674952749607, 5)
        testing.assert_almost_equal(config.lnw_sat_mean, 0.027343501521674522, 5)
        testing.assert_almost_equal(config.lnw_sat_sigma, 0.27339568749678567, 5)
        testing.assert_almost_equal(config.lnw_cen_mean, 0.21916480960693957, 5)
        testing.assert_almost_equal(config.lnw_cen_sigma, 0.13680977279433526, 5)

        if os.path.exists(test_dir):
            shutil.rmtree(test_dir, True)

if __name__=='__main__':
    unittest.main()
