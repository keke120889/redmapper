import fitsio
import esutil as eu
import numpy as np
import itertools
from catalog import Catalog, DataEntry


class GalaxyCatalog(Catalog):

    singleton_class = Galaxy

    def __init__(self, depth=10, *arrays):
        super(GalaxyCatalog, self).__init__(arrays)
        self._htm_matcher = None
        self.depth = depth

    @classmethod
    def from_custom_file(cls, filename): pass

    def match(self, galaxy, radius):
        if self._htm_matcher is None:
            self._htm_matcher = eu.Matcher(self.ra, self.dec, self.depth)
        _, indices, dists = self._htm_matcher._match(galaxy.ra, galaxy.dec, 
                                                        radius, maxmatch=-1)
        return indices, dists


class Galaxy(DataEntry): pass

