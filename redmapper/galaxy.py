import fitsio


class Galaxy:

    """ docstring """

    def __init__(self, gal_string):
        self._read_gal(gal_string)
        self.neighbors = None

    def _read_gal(self, gal_string): pass

    # populates dictionary of galaxies to distances from current galaxy
    # but returns just the keys (the galaxies themselves)
    def get_neighbors(self, galaxy_list):
        if self.neighbors: return self.neighbors.keys()
        # else, use htm_find_neighbors

    def distance_from(self, galaxy):
        if self.neighbors and galaxy in self.neighbors: 
            return self.neighbors[galaxy]


class Cluster:

    ### TO DISCUSS WITH ELI:
    ### I think incat should just be a list of galaxy IDs

    """ docstring """

    def __init__(self, clust_tuple):
        self._read_clust(clust_tuple)
        self.members = None
        # self.center = None

    def _read_clust(self, clust_tuple):
        self.index, self.ra, self.dec, self.z = clust_tuple

    # def ra(self): return self.center.ra
    # def dec(self): return self.center.dec

    def get_potential_members(self):
        if self.members: return self.members.keys()
        self.members = self.center.get_neighbors()
        return self.members

    def calc_lambda(self): pass


def read_galaxies(filename):
    galaxies = []
    for gal in fitsio.read(filename): galaxies.append(Galaxy(gal))
    return galaxies


def read_clusters(filename):
    clusters = []
    for clust in fitsio.read(filename): clusters.append(Cluster(clust))
    return clusters

