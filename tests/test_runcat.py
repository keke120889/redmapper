from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import unittest
import numpy.testing as testing
import numpy as np
import fitsio
from numpy import random
import healpy as hp

from redmapper import Cluster
from redmapper import ClusterCatalog
from redmapper import Configuration
from redmapper import GalaxyCatalog
from redmapper import DataObject
from redmapper import RedSequenceColorPar
from redmapper import Background
from redmapper import HPMask
from redmapper import DepthMap
from redmapper import RunCatalog

class RuncatTestCase(unittest.TestCase):
    """
    """
    def runTest(self):
        random.seed(seed=12345)

        file_path = 'data_for_tests'
        conffile = 'testconfig.yaml'
        catfile = 'test_cluster_pos.fit'

        config = Configuration(file_path + '/' + conffile)
        config.catfile = file_path + '/' + catfile

        runcat = RunCatalog(config)

        runcat.run(do_percolation_masking=False)

        testing.assert_equal(runcat.cat.mem_match_id, [1, 2, 3])
        testing.assert_almost_equal(runcat.cat.Lambda, [24.4121723, 26.8377132, 13.3675709])
        testing.assert_almost_equal(runcat.cat.lambda_e, [2.5175705, 4.8330407, 2.4651196])
        testing.assert_almost_equal(runcat.cat.z_lambda, [0.2278536, 0.3225681, 0.2176394])
        testing.assert_almost_equal(runcat.cat.z_lambda_e, [0.0063111, 0.0135445, 0.0098461])

        runcat.run(do_percolation_masking=True)

        testing.assert_equal(runcat.cat.mem_match_id, [1, 2, 3])
        testing.assert_almost_equal(runcat.cat.Lambda, [24.2748775, 26.8224583, -1.])
        testing.assert_almost_equal(runcat.cat.lambda_e, [2.507231, 4.8306675, -1.])
        testing.assert_almost_equal(runcat.cat.z_lambda, [0.2278541, 0.322565, -1.])
        testing.assert_almost_equal(runcat.cat.z_lambda_e, [0.0063084, 0.0135308, -1.])

if __name__=='__main__':
    unittest.main()
