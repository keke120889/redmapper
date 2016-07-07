import fitsio
import esutil as eu


class Galaxy:

    """ docstring """

    def __init__(self, gal_tuple):
        self._read_gal(gal_tuple)

    def _read_gal(self, gal_tuple): pass

    def distance_from(self, galaxy):
        return eu.coords.sphdist(self.ra, self.dec, galaxy.ra, galaxy.dec)


class Cluster:

    """ docstring """

    def __init__(self, clust_tuple):
        self._read_clust(clust_tuple)
        self.member_probs = self.centers = None

    def _read_clust(self, clust_tuple):
        self.index, self.ra, self.dec, self.z = clust_tuple

    def get_members(self, radius=None, galcat=None):
        if self._member_probs and radius is None:
            return self.member_probs['GAL']
        if galcat is None:
            raise ValueError("A GalaxyCatalog object must be specified.")
        if radius is None or radius < 0 or radius > 360:
            raise ValueError("A radius in degrees must be specified.")
        indices = galaxy_list._match(self.ra, self.dec, radius)
        self.member_probs = np.array(zip([galcat[i] for i in indices], 
                np.zeros(len(indices))), dtype=[('GAL', 'O'), ('PMEM', 'f4')])
        return self.member_probs['GAL']

    def calc_lambda(self): pass


class GalaxyCatalog:

    """ docstring """

    def __init__(self, filename, depth=10):
        self._gal_recarray = fitsio.read(filename)
        self._gal_list = {}
        self._htm_matcher = eu.Matcher(self._gal_recarray['RA'], 
                                            self._gal_recarray['DEC'], depth)
        
    def __getitem__(self, key):
        if key is not in self._gal_list:
            self._gal_list[key] = Galaxy(self._gal_recarray[key])
        return self._gal_list[key]

    def __len__(self): return len(self._gal_recarray)

    def ra(self): return self._gal_recarray['RA']
    def dec(self): return self._gal_recarray['DEC']

    def match(self, galaxy, radius, maxmatch):
        _, indices, dists = self._htm_matcher._match(galaxy.ra, galaxy.dec, 
                                                        radius, maxmatch=-1)
        return indices


class ClusterCatalog:

    """ docstring """

    def __init__(self, filename):
        self._clust_recarray = fitsio.read(filename)
        self._clust_list = {}

    def __getitem__(self, key): 
        if key is not in self._clust_list:
            self._clust_list[key] = Cluster(self._clust_recarray[key])
        return self._clust_list[key]

    def __len__(self): return len(self._clust_recarray)

    def ra(self): return self._clust_recarray['RA']
    def dec(self): return self._clust_recarray['DEC']
    def z(self): return self._clust_recarray['Z']

