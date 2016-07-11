import fitsio
import esutil as eu
import numpy as np
import itertools


class Cluster(object):

    """ docstring """

    def __init__(self, clust_tuple):
        self._read_clust(clust_tuple)
        self.members = self.centers = None

    def _read_clust(self, clust_tuple):
        self.index, self.ra, self.dec, self.z = clust_tuple

    def find_members(self, radius=None, galcat=None):
        if self._member_probs and radius is None:
            return self.member_probs['GAL']
        if galcat is None:
            raise ValueError("A GalaxyCatalog object must be specified.")
        if radius is None or radius < 0 or radius > 360:
            raise ValueError("A radius in degrees must be specified.")
        indices, dists = galaxy_list._match(self.ra, self.dec, radius)
        ndarray = np.array(np.zeros(len(indices)), 
                                dtype=[('DIST', 'f8'), ('PMEM', 'f8')])
        self.members = GalaxyCatalog(galcat[indices], ndarray)

    def calc_lambda(self): pass


class ClusterCatalog(object):

    """ docstring """

    def __init__(self, filename):
        self._clust_ndarray = fitsio.read(filename)
        self._clust_list = {}

    def __getattribute__(self, attr):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:  # attr must be a fieldname
            pass
        if attr.upper() in self._clust_ndarray.dtype.names:
            return self._clust_ndarray[attr.upper()]
        return object.__getattribute__(self, attr)

    def dtype(self): return self._clust_ndarray.dtype

    def __getitem__(self, key): 
        if key is not in self._clust_list:
            self._clust_list[key] = Cluster(self._clust_ndarray[key])
        return self._clust_list[key]

    def __len__(self): return len(self._clust_ndarray)

