import fitsio
import esutil as eu
import numpy as np
import itertools
from catalog import Catalog, Entry


class Cluster(Entry):

    def find_members(self, radius=None, galcat=None):
        if galcat is None:
            raise ValueError("A GalaxyCatalog object must be specified.")
        if radius is None or radius < 0 or radius > 180:
            raise ValueError("A radius in degrees must be specified.")
        indices, dists = galcat.match(self, radius)
        self.members = galcat[indices]
        new_fields_array = np.array(zip(dists, np.zeros(len(indices))),
                                        dtype=[('DIST', 'f8'), ('PMEM', 'f8')])
        self.members.add_fields(new_fields_array)

    def calc_lambda(self): pass


class ClusterCatalog(Catalog): entry_class = Cluster

