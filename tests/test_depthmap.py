from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import unittest
import numpy.testing as testing
import numpy as np
import healpy as hp
import fitsio

from redmapper import DepthMap
from redmapper import Configuration

class DepthMapTestCase(unittest.TestCase):
    def runTest(self):
        """
        """
        file_path = "data_for_tests"
        conf_filename = "testconfig.yaml"
        config = Configuration(file_path + "/" + conf_filename)

        # Check the regular depth
        depthstr = DepthMap(config)

        RAs = np.array([142.10934, 142.04090, 142.09242, 142.11448, 50.0])
        Decs = np.array([65.022666, 65.133844, 65.084844, 65.109541, 50.0])

        comp_limmag = np.array([20.6847, 20.5915, 20.5966, 20.5966, -1.63750e+30], dtype='f4')
        comp_exptime = np.array([70.3742, 63.5621, 63.5621, 63.5621, -1.63750e+30], dtype='f4')
        comp_m50 = np.array([20.8964, 20.8517, 20.8568, 20.8568, -1.63750e+30], dtype='f4')

        limmag, exptime, m50 = depthstr.get_depth_values(RAs, Decs)

        testing.assert_almost_equal(limmag, comp_limmag, 4)
        testing.assert_almost_equal(exptime, comp_exptime, 4)
        testing.assert_almost_equal(m50, comp_m50, 4)

        config2 = Configuration(file_path + "/" + conf_filename)
        config2.hpix = 582972
        config2.nside = 1024
        config2.border = 0.02
        depthstr2 = DepthMap(config2)

        limmag2, exptime2, m502 = depthstr2.get_depth_values(RAs, Decs)
        comp_limmag[0] = hp.UNSEEN
        comp_exptime[0] = hp.UNSEEN
        comp_m50[0] = hp.UNSEEN

        testing.assert_almost_equal(limmag2, comp_limmag, 4)
        testing.assert_almost_equal(exptime2, comp_exptime, 4)
        testing.assert_almost_equal(m502, comp_m50, 4)

if __name__=='__main__':
    unittest.main()
