import fitsio
import esutil as eu
import numpy as np
import itertools
from catalog import Catalog


class Galaxy(object):

    """ docstring """

    def __init__(self, gal_tuple):
        self._read_gal(gal_tuple)

    def _read_gal(self, gal_tuple): pass

    def distance_from(self, galaxy):
        return eu.coords.sphdist(self.ra, self.dec, galaxy.ra, galaxy.dec)


class GalaxyCatalog(Catalog):

    """ docstring """

    def __init__(self, *arrays, depth=10):
        super(GalaxyCatalog, self).__init__(arrays)
        self._htm_matcher = None
        self.depth = depth

    def match(self, galaxy, radius):
        if self._htm_matcher is None:
            self._htm_matcher = eu.Matcher(self.ra, self.dec, self.depth)
        _, indices, dists = self._htm_matcher._match(galaxy.ra, galaxy.dec, 
                                                        radius, maxmatch=-1)
        return indices, dists

