import fitsio
import esutil as eu
import numpy as np
import itertools


class Galaxy:

    """ docstring """

    def __init__(self, gal_tuple):
        self._read_gal(gal_tuple)

    def _read_gal(self, gal_tuple): pass

    def distance_from(self, galaxy):
        return eu.coords.sphdist(self.ra, self.dec, galaxy.ra, galaxy.dec)


class GalaxyCatalog(object):

    """ docstring """

    def __init__(self, *ndarrays, depth=10):
        if any([len(ndarr) != len(ndarrays[0]) for ndarr in ndarrays]):
            raise ValueError("Input arrays must have the same length.")
        self._gal_ndarrays = list(ndarrays) # do I need to copy?? Dont think so...
        self._htm_matcher = None

    @classmethod
    def fromfilename(cls, filename):
        ndarray = fitsio.read(filename)
        return cls(ndarray)

    def __getattribute__(self, attr):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:  # attr must be a fieldname
            pass
        for ndarr in self._gal_ndarrays:
            if attr.upper() in ndarr.dtype.names:
                return ndarr[attr.upper()]
        return object.__getattribute__(self, attr)

    def __getitem__(self, key):
        return tuple(itertools.chain([ndarr for ndarr in self._gal_ndarrays]))

    def __len__(self): return len(self._gal_ndarrays)

    def add_fields(self, ndarray):
        if len(ndarray) != len(self):
            raise ValueError("Input arrays must have the same length.")
        self._gal_ndarrays.append(ndarray)

    def _match(self, galaxy, radius):
        if self._htm_matcher is None:
            self._htm_matcher = eu.Matcher(self.ra, self.dec, depth)
        _, indices, dists = self._htm_matcher._match(galaxy.ra, galaxy.dec, 
                                                        radius, maxmatch=-1)
        return indices, dists


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

