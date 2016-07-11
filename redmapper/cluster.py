import fitsio
import esutil as eu
import numpy as np
import itertools
import galaxy


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
        indices, dists = galaxy_list.match(self.ra, self.dec, radius)
        new_fields_array = p.array(np.zeros(len(indices)), 
                                dtype=[('DIST', 'f8'), ('PMEM', 'f8')])
        self.members = galcat[indices].add_fields(new_fields_array)

    def calc_lambda(self): pass

