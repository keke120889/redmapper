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
        testing.assert_almost_equal(runcat.cat.Lambda, [24.4121723, 26.8987331, 13.3675709])
        testing.assert_almost_equal(runcat.cat.lambda_e, [2.5175705, 4.8441205, 2.4651196])
        testing.assert_almost_equal(runcat.cat.z_lambda, [0.2278651, 0.3226278, 0.2231186])

        testing.assert_almost_equal(runcat.cat.z_lambda_e, [0.0062949, 0.013563, 0.0096974])

        runcat.run(do_percolation_masking=True)

        testing.assert_equal(runcat.cat.mem_match_id, [1, 2, 3])
        testing.assert_almost_equal(runcat.cat.Lambda, [24.4121723, 26.8529682, -1.0])
        testing.assert_almost_equal(runcat.cat.lambda_e, [2.5175705, 4.8369522, -1.0])
        testing.assert_almost_equal(runcat.cat.z_lambda, [0.2278651, 0.3226319, -1.0])
        testing.assert_almost_equal(runcat.cat.z_lambda_e, [0.0062967, 0.013576, -1.0])

if __name__=='__main__':
    unittest.main()
