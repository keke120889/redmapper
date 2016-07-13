import fitsio
import esutil as eu
from esutil.htm import Matcher
import numpy as np
import itertools
from catalog import Catalog, Entry


class Cluster(Entry):

    def find_members(self, radius=None, galcat=None):
        if galcat is None:
            raise ValueError("A GalaxyCatalog object must be specified.")
        if radius is None or radius < 0 or radius > 360:
            raise ValueError("A radius in degrees must be specified.")
        indices, dists = galcat.match(self, radius)
        new_fields_array = p.array(np.zeros(len(indices)), 
                                dtype=[('DIST', 'f8'), ('PMEM', 'f8')])
        self.members = galcat[indices].add_fields(new_fields_array)

    def calc_lambda(self): pass


class ClusterCatalog(Catalog): entry_class = Cluster

