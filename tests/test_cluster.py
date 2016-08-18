import unittest
import numpy.testing as testing
import numpy as np
import fitsio

from redmapper.cluster import Cluster
from redmapper.galaxy import GalaxyCatalog

NUM_CHECK = 8

class ClusterFiltersTestCase(unittest.TestCase):

    def test_nfw(self):
        test_indices = np.array([46, 38,  1,  1, 11, 24, 25, 16])
        py_nfw = self._calc_radial_profile()[indices]
        idl_nfw = np.array([0.29360449, 0.14824243, 0.14721203, 0.14721203, 
                            0.23459411, 0.31615007, 0.29307860, 0.29737136])
        testing.assert_almost_equal(py_nfw, idl_nfw)

    def test_luminosity(self): pass
    def test_background(self): pass

    def setUp(self):
        self.cluster = Cluster(np.empty(0))
        file_name, file_path = 'test_cluster_members.fit', 'data'
        self.cluster.members = GalaxyCatalog.from_fits_file(file_path 
                                                            + '/' + file_name)
        self.cluster.z = cluster.members.z[0]


        
class ClusterMembersTestCase(unittest.TestCase):

    def test_member_finding(self): pass
    def test_richness(self): pass

if __name__=='__main__':
        unittest.main()