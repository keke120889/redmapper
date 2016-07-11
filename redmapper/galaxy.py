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

    def __init__(self, *arrays, depth=10):
        if any([len(arr) != len(arrays[0]) for arr in arrays]):
            raise ValueError("Input arrays must have the same length.")
        self._gal_arrays = list(arrays)
        self.dtype = itertools.chain([arr.dtype for arr in self._gal_arrays])
        self._htm_matcher = None

    @classmethod
    def fromfilename(cls, filename):
        array = fitsio.read(filename)
        return cls(array)

    def __getattribute__(self, attr):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:  # attr must be a fieldname
            pass
        for arr in self._gal_arrays:
            if attr.upper() in arr.dtype.names:
                return arr[attr.upper()]
        return object.__getattribute__(self, attr)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return np.array([self[i] for i in xrange(*key.indices(len(self)))],
                                dtype=self.dtype)
        if isinstance(key, list):
            return np.array([self[i] for i in key, dtype=self.dtype])
        return tuple(itertools.chain([arr for arr in self._gal_arrays]))

    def __len__(self): return len(self._gal_arrays[0])

    def add_fields(self, array):
        if len(array) != len(self):
            raise ValueError("Input arrays must have the same length.")
        self._gal_arrays.append(array)

    def _match(self, galaxy, radius):
        if self._htm_matcher is None:
            self._htm_matcher = eu.Matcher(self.ra, self.dec, depth)
        _, indices, dists = self._htm_matcher._match(galaxy.ra, galaxy.dec, 
                                                        radius, maxmatch=-1)
        return indices, dists

