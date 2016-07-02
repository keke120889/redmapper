import fitsio


class Galaxy:

    """ docstring """

    def __init__(self, gal_tuple):
        self._read_gal(gal_tuple)
        self._neighbor_dists = None

    def _read_gal(self, gal_tuple): pass

    # populates dictionary of galaxies to distances from current galaxy
    # but returns just the keys (the galaxies themselves)
    def get_neighbors(self, galaxy_list = None):
        if self._neighbor_dists: return self._neighbor_dists.keys()
        # else, use logic of htm_find_neighbors
        self._neighbor_dists = {} # this is dummy implementation though
        return self._neighbor_dists.keys()

    def distance_from(self, galaxy):
        if self._neighbor_dists and galaxy in self._neighbor_dists: 
            return self._neighbor_dists[galaxy]


class Cluster:

    ### TO DISCUSS WITH ELI:
    ### I think incat should just be a list of galaxy IDs

    """ docstring """

    def __init__(self, clust_tuple):
        self._read_clust(clust_tuple)
        self.member_probs = None 

    def _read_clust(self, clust_tuple):
        self.index, self.ra, self.dec, self.z = clust_tuple
        # following is just dummy implementation to make code work
        self.center = Galaxy((self.ra, self.dec, self.z))

    # def ra(self): return self.center.ra
    # def dec(self): return self.center.dec

    # I envision members to be a dict from member galaxies to 
    # respective membership probabilities but get_members gets keys
    def get_members(self):
        if self.member_probs: return self.member_probs.keys()
        self.member_probs = {gal:None for gal in self.center.get_neighbors()}
        return self.member_probs.keys()

    def calc_lambda(self): pass


def read_galaxies(filename):
    galaxies = []
    for gal in fitsio.read(filename): galaxies.append(Galaxy(gal))
    return galaxies


def read_clusters(filename):
    clusters = []
    for clust in fitsio.read(filename): clusters.append(Cluster(clust))
    return clusters

