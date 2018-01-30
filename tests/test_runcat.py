import unittest
import numpy.testing as testing
import numpy as np
import fitsio
from numpy import random
import healpy as hp

from redmapper.cluster import Cluster
from redmapper.cluster import ClusterCatalog
from redmapper.configuration import Configuration
from redmapper.galaxy import GalaxyCatalog
from redmapper.catalog import DataObject
from redmapper.redsequence import RedSequenceColorPar
from redmapper.background import Background
from redmapper.mask import HPMask
from redmapper.depthmap import DepthMap
from redmapper.runcat import RunCatalog

class RuncatTestCase(unittest.TestCase):
    """
    """
    def runTest(self):

        file_path = 'data_for_tests'
        conffile = 'testconfig.yaml'
        catfile = 'test_cluster_pos.fit'

        config = Configuration(file_path + '/' + conffile)
        config.catfile = file_path + '/' + catfile

        runcat = RunCatalog(config)

        runcat.run(do_percolation_masking=False)

        testing.assert_equal(runcat.cat.mem_match_id, [1, 2, 3])
        testing.assert_almost_equal(runcat.cat.Lambda, [ 23.86299324,  17.39488411, 13.36757088])
        testing.assert_almost_equal(runcat.cat.lambda_e, [ 2.47804546,  2.00936174, 2.4651196])
        testing.assert_almost_equal(runcat.cat.z_lambda, [ 0.22786506,  0.32121494, 0.22311865])
        testing.assert_almost_equal(runcat.cat.z_lambda_e, [ 0.00629484,  0.01389629, 0.00969736])

        runcat.run(do_percolation_masking=True)

        testing.assert_equal(runcat.cat.mem_match_id, [1, 2, 3])
        testing.assert_almost_equal(runcat.cat.Lambda, [ 23.86299324,  17.39488411, -1.0])
        testing.assert_almost_equal(runcat.cat.lambda_e, [ 2.47804546,  2.00936174, -1.0])
        testing.assert_almost_equal(runcat.cat.z_lambda, [ 0.22786506,  0.32121494, -1.0])
        testing.assert_almost_equal(runcat.cat.z_lambda_e, [ 0.00629484,  0.01389629, -1.0])

