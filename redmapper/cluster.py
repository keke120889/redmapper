import fitsio
import esutil as eu
import numpy as np
import itertools
from solver_nfw.solver_nfw_lib import Solver
from catalog import Catalog, Entry
from utilities import chisq_pdf


class Cluster(Entry):
    """Docstring."""

    def find_members(self, radius=None, galcat=None):
        if galcat is None:
            raise ValueError("A GalaxyCatalog object must be specified.")
        if radius is None or radius < 0 or radius > 180:
            raise ValueError("A radius in degrees must be specified.")
        indices, dists = galcat.match(self, radius) # pass in a Galaxy?
        self.members = galcat[indices]
        new_fields = [('DIST', 'f8'), ('R', 'f8'), ('PMEM', 'f8'), 
                        ('CHISQ', 'f8')]
        self.members.add_fields(new_fields)
        self.members.dist = dists

    def _calc_bkg_density(self, bkg):
        sigma_g = bkg.sigma_g_lookup(self.z, self.members.chisq, 
                                                    self.members.refmag)

    def calc_richness(self, zredstr, bkg, r0, beta, mpc_scale):
        self.members.r = self.members.dist * mpc_scale
        self.members.chisq = zredstr.calculate_chisq(self.members, self.z)
        rho = chisq_pdf(self.members.chisq, zredstr.ncol)
        nfw = 0 # two dimensional cluster galaxy density profile (NFW)
        phi = 0 # cluster luminosity function
        ucounts = (2*np.pi*self.members.r) * nfw * phi * rho
        bcounts = self._calc_bkg_density(bkg)
        
        w = 0

        richness_obj = Solver(r0, beta, ucounts, bcounts, r, w)
        return richness_obj.solve_nfw()


class ClusterCatalog(Catalog): 
    """Dosctring."""
    entry_class = Cluster

