import fitsio
import esutil as eu
from esutil.htm import Matcher
import numpy as np
import itertools
from catalog import Catalog, Entry


class Galaxy(Entry): pass


class GalaxyCatalog(Catalog):

    entry_class = Galaxy

    def __init__(self, *arrays, **kwargs):
        super(GalaxyCatalog, self).__init__(*arrays)
        self._htm_matcher = None
        self.depth = 10 if 'depth' not in kwargs else kwargs['depth']

    @classmethod
    def from_custom_file(cls, filename): pass

    def match(self, galaxy, radius):
        if self._htm_matcher is None:
            self._htm_matcher = Matcher(self.depth, self.ra, self.dec)
        _, indices, dists = self._htm_matcher.match(galaxy.ra, galaxy.dec, 
                                                        radius, maxmatch=0)
        return indices, dists

