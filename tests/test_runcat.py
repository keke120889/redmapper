import unittest
import numpy.testing as testing
import numpy as np
import fitsio
from numpy import random

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
    Tests of redmapper.RunCatalog, which computes richness for an input catalog with
    ra/dec/z.
    """
    def runTest(self):
        """
        Run the redmapper.RunCatalog tests.
        """
        random.seed(seed=12345)

        file_path = 'data_for_tests'
        conffile = 'testconfig.yaml'
        catfile = 'test_cluster_pos.fit'

        config = Configuration(file_path + '/' + conffile)
        config.catfile = file_path + '/' + catfile
        config.bkg_local_compute = True

        runcat = RunCatalog(config)

        runcat.run(do_percolation_masking=False)

        testing.assert_equal(runcat.cat.mem_match_id, [1, 2, 3])
        testing.assert_almost_equal(runcat.cat.Lambda, [24.168093, 26.929243, 13.367571], 5)
        testing.assert_almost_equal(runcat.cat.lambda_e, [2.5000322, 4.8504086, 2.4651196], 5)
        testing.assert_almost_equal(runcat.cat.z_lambda, [0.2278546, 0.3225739, 0.2176394], 5)
        testing.assert_almost_equal(runcat.cat.z_lambda_e, [0.0063102, 0.0135351, 0.0098461], 5)
        testing.assert_almost_equal(runcat.cat.bkg_local, [1.2288319, 1.6887128, 1.7223835])

        runcat.run(do_percolation_masking=True)

        testing.assert_equal(runcat.cat.mem_match_id, [1, 2, 3])
        testing.assert_almost_equal(runcat.cat.Lambda, [24.122328, 26.913988, -1.], 5)
        testing.assert_almost_equal(runcat.cat.lambda_e, [2.4962583, 4.8480325, -1.], 5)
        testing.assert_almost_equal(runcat.cat.z_lambda, [0.2278544,  0.3225641, -1.], 5)
        testing.assert_almost_equal(runcat.cat.z_lambda_e, [0.0063079,  0.0135317, -1.], 5)
        testing.assert_almost_equal(runcat.cat.bkg_local, [1.18146, 1.73055, 0.], 5)

if __name__=='__main__':
    unittest.main()
