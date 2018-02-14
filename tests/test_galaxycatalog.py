from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import unittest
import numpy.testing as testing
import numpy as np
import fitsio
from numpy import random
import healpy as hp

from redmapper import Configuration
from redmapper import GalaxyCatalog


class GalaxyCatalogTestCase(unittest.TestCase):
    """
    """
    def runTest(self):

        file_path = 'data_for_tests'

        galfile = 'pixelized_dr8_test/dr8_test_galaxies_master_table.fit'

        gals_all = GalaxyCatalog.from_galfile(file_path + '/' + galfile)

        # check that we got the expected number...
        testing.assert_equal(gals_all.size, 14449)

        # read in a subregion, no border
        gals_sub = GalaxyCatalog.from_galfile(file_path + '/' + galfile,
                                              hpix=2163, nside=64)

        theta = (90.0 - gals_all.dec) * np.pi/180.
        phi = gals_all.ra * np.pi/180.
        ipring_all = hp.ang2pix(64, theta, phi)
        use, = np.where(ipring_all == 2163)

        testing.assert_equal(gals_sub.size, use.size)

        # read in a subregion, with border
        gals_sub = GalaxyCatalog.from_galfile(file_path + '/' + galfile,
                                              hpix=9218, nside=128, border=0.1)

        # this isn't really a big enough sample catalog to fully test...
        testing.assert_equal(gals_sub.size, 7950)

        # and test the matching...

        indices, dists = gals_all.match_one(140.5, 65.0, 0.2)
        testing.assert_equal(indices.size, 521)
        testing.assert_array_less(dists, 0.2)

        i0, i1, dists = gals_all.match_many([140.5,141.2],
                                            [65.0, 65.2], [0.2,0.1])
        testing.assert_equal(i0.size, 666)
        test, = np.where(i0 == 0)
        testing.assert_equal(test.size, 521)
        testing.assert_array_less(dists[test], 0.2)
        test, = np.where(i0 == 1)
        testing.assert_equal(test.size, 666 - 521)
        testing.assert_array_less(dists[test], 0.1)
