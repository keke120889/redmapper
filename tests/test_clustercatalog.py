from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import unittest
import numpy.testing as testing
import numpy as np
import fitsio
from numpy import random
import healpy as hp
from esutil.cosmology import Cosmo

from redmapper import Cluster
from redmapper import ClusterCatalog
from redmapper import Configuration
from redmapper import GalaxyCatalog
from redmapper import DataObject
from redmapper import RedSequenceColorPar
from redmapper import Background
from redmapper import HPMask
from redmapper import DepthMap



class ClusterCatalogTestCase(unittest.TestCase):
    """
    """
    def runTest(self):

        random.seed(seed=12345)

        file_path = 'data_for_tests'
        conffile = 'testconfig.yaml'

        config = Configuration(file_path + '/' + conffile)

        gals_all = GalaxyCatalog.from_galfile(config.galfile)

        zred_filename = 'test_dr8_pars.fit'
        zredstr = RedSequenceColorPar(file_path + '/' + zred_filename, fine=True)

        bkg_filename = 'test_bkg.fit'
        bkg = Background('%s/%s' % (file_path, bkg_filename))

        mask = HPMask(config)
        maskgal_index = mask.select_maskgals_sample(maskgal_index=0)
        depthstr = DepthMap(config)

        testcatfile = 'test_cluster_pos.fit'
        cat = ClusterCatalog.from_catfile(file_path + '/' + testcatfile,
                                          zredstr=zredstr,
                                          config=config,
                                          bkg=bkg)

        # test single neighbors...
        c0 = cat[0]
        c0.find_neighbors(0.2, gals_all)
        c1 = cat[1]
        c1.find_neighbors(0.2, gals_all)

        testing.assert_equal(c0.neighbors.size, 580)
        testing.assert_equal(c1.neighbors.size, 298)
        testing.assert_array_less(c0.neighbors.dist, 0.2)
        testing.assert_array_less(c1.neighbors.dist, 0.2)

        # and multi-match...
        i0, i1, dist = gals_all.match_many(cat.ra, cat.dec, 0.2)

        u0, = np.where(i0 == 0)
        testing.assert_equal(c0.neighbors.size, u0.size)

        u1, = np.where(i0 == 1)
        testing.assert_equal(c1.neighbors.size, u1.size)

        # and compute the richness on the first one...
        mask.set_radmask(c0)

        depthstr.calc_maskdepth(mask.maskgals, c0.ra, c0.dec, c0.mpc_scale)

        richness = c0.calc_richness(mask)

        # Make sure the numbers were propagated to the parent catalog
        testing.assert_equal(richness, cat.Lambda[0])
        testing.assert_equal(c0.Lambda_e, cat.Lambda_e[0])
        testing.assert_equal(c0.scaleval, cat.scaleval[0])

        # And make sure the numbers are correct
        #testing.assert_almost_equal(richness, 23.86299324)
        testing.assert_almost_equal(richness, 24.4121723)

        # and we're done

if __name__=='__main__':
    unittest.main()

