import fitsio
import esutil as eu
import numpy as np
import itertools
from catalog import Catalog, Entry


class Cluster(Entry):
    """Docstring."""

    def find_members(self, radius=None, galcat=None):
        if galcat is None:
            raise ValueError("A GalaxyCatalog object must be specified.")
        if radius is None or radius < 0 or radius > 180:
            raise ValueError("A radius in degrees must be specified.")
        indices, dists = galcat.match(self, radius) # techincally need to pass in a Galaxy
        self.members = galcat[indices]
        new_fields = np.array(zip(dists, np.zeros(len(indices))),
                                    dtype=[('DIST', 'f8'), ('PMEM', 'f8')])
        self.members.add_fields(new_fields)

    def calc_lambda(self): pass


class ClusterCatalog(Catalog): 
    """Dosctring."""
    
    entry_class = Cluster

