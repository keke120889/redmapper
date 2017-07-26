import unittest
import numpy.testing as testing
import numpy as np
import fitsio
from esutil.cosmology import Cosmo

from redmapper.depthmap import DepthMap
from redmapper.config import Configuration
from redmapper.cluster import Cluster
from redmapper.mask import HPMask
from redmapper.galaxy import GalaxyCatalog


class TestDepthMap(unittest.TestCase):
    """
    Unittest for depthmap.py
    
    not sure if this is really necessary as depthmap results are  tested in test_cluster.py.
    """
    def runTest(self):
        """
        """
        self.file_path = 'data_for_tests'
        conf_filename = 'testconfig.yaml'
        confstr = Configuration(self.file_path + '/' + conf_filename)
        
        self.cluster = Cluster(confstr)
        filename = 'test_cluster_members.fit'
        self.cluster.neighbors = GalaxyCatalog.from_fits_file(self.file_path + '/' + filename)
        cosmo = Cosmo()
        
        mpc_scale = np.radians(1.) * cosmo.Dl(0, self.cluster.neighbors.z) / (1 + self.cluster.neighbors.z)**2
        
        depthmap = DepthMap(confstr)
        
        mask = HPMask(confstr)
        #depthmap.calc_maskdepth(mask.maskgals, self.cluster.neighbors.ra, self.cluster.neighbors.dec, mpc_scale)
        
if __name__=='__main__':
    unittest.main()