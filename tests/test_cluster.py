import unittest
import numpy.testing as testing
import numpy as np
import fitsio

from redmapper.cluster import Cluster
from redmapper.galaxy import GalaxyCatalog

class ClusterFiltersTestCase(unittest.TestCase):

    def test_nfw(self): pass
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