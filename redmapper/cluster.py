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
        array = np.array(np.zeros(len(indices)), 
                                dtype=[('DIST', 'f8'), ('PMEM', 'f8')])
        self.members = GalaxyCatalog(galcat[indices], array)

    def calc_lambda(self): pass


class ClusterCatalog(object):

    """ docstring """

    def __init__(self, *arrays, depth=10):
        if any([len(arr) != len(arrays[0]) for arr in arrays]):
            raise ValueError("Input arrays must have the same length.")
        self._clust_arrays = list(arrays)
        self.dtype = itertools.chain([arr.dtype for arr in self._clust_arrays])

    @classmethod
    def fromfilename(cls, filename):
        array = fitsio.read(filename)
        return cls(array)

    def __getattribute__(self, attr):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:  # attr must be a fieldname
            pass
        for arr in self._clust_arrays:
            if attr.upper() in arr.dtype.names:
                return arr[attr.upper()]
        return object.__getattribute__(self, attr)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return np.array([self[i] for i in xrange(*key.indices(len(self)))],
                                dtype=self.dtype)
        if isinstance(key, list):
            return np.array([self[i] for i in key, dtype=self.dtype])
        return tuple(itertools.chain([arr for arr in self._clust_arrays]))

    def __len__(self): return len(self._clust_arrays[0])

    def add_fields(self, array):
        if len(array) != len(self):
            raise ValueError("Input arrays must have the same length.")
        self._clust_arrays.append(array)

