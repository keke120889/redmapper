import fitsio
import esutil as eu
import numpy as np
import itertools
from solver_nfw import Solver
from catalog import Catalog, Entry
from utilities import chisq_pdf
from scipy.special import erf

class Cluster(Entry):
    """

    Class for a single galaxy cluster, with methods to perform
    computations on individual clusters

    parameters
    ----------
    (TBD)

    """
    def find_neighbors(self, radius, galcat):
        """
        parameters
        ----------
        radius: float
            radius in degrees to look for neighbors
        galcat: GalaxyCatalog
            catalog of galaxies

        This method is not finished or tested.

        """
        if galcat is None:
            raise ValueError("A GalaxyCatalog object must be specified.")
        if radius is None or radius < 0 or radius > 180:
            raise ValueError("A radius in degrees must be specified.")
        indices, dists = galcat.match(self, radius) # self refers to the cluster
        self.neighbors = galcat[indices]
        #Dist is arcmin???, R is Mpc/h
        new_fields = [('DIST', 'f8'), ('R', 'f8'), ('PMEM', 'f8'), 
                        ('CHISQ', 'f8')]
        self.neighbors.add_fields(new_fields)
        self.neighbors.dist = dists

    def _calc_radial_profile(self, rscale=0.15):
        """
        internal method for computing radial profile weights

        parameters
        ----------
        rscale: float
            r_s for nfw profile

        returns
        -------
        sigx: array of floats
           sigma(x)
        """
        corer = 0.1
        x, corex = self.neighbors.r/rscale, corer/rscale
        sigx = np.zeros(self.neighbors.r.size)

        low, = np.where(x < corex)
        mid, = np.where((x >= corex) & (x < 1.0))
        high, = np.where((x >= 1.0) & (x < 10.0/rscale))
        other, = np.where((x > 0.999) & (x < 1.001))

        if low.size > 0:
            arg = np.sqrt((1. - corex)/(1. + corex))
            pre = 2./(np.sqrt(1. - corex**2))
            front = 1./(corex**2 - 1)
            sigx[low] = front * (1. - pre*0.5*np.log((1.+arg)/(1.-arg)))

        if mid.size > 0:
            arg = np.sqrt((1. - x[mid])/(1. + x[mid]))
            pre = 2./(np.sqrt(1. - x[mid]**2))
            front = 1./(x[mid]**2 - 1.)
            sigx[mid] = front * (1. - pre*0.5*np.log((1.+arg)/(1.-arg)))

        if high.size > 0:
            arg = np.sqrt((x[high] - 1.)/(x[high] + 1.))
            pre = 2./(np.sqrt(x[high]**2 - 1.))
            front = 1./(x[high]**2 - 1)
            sigx[high] = front * (1. - pre*np.arctan(arg))

        if other.size > 0:
            xlo, xhi = 0.999, 1.001
            arglo, arghi = np.sqrt((1-xlo)/(1+xlo)), np.sqrt((xhi-1)/(xhi+1))
            prelo, prehi = 2./np.sqrt(1.-xlo**2), 2./np.sqrt(xhi**2 - 1)
            frontlo, fronthi = 1./(xlo**2 - 1), 1./(xhi**2 - 1)
            testlo = frontlo * (1 - prelo*0.5*np.log((1+arglo)/(1-arglo)))
            testhi = fronthi * (1 - prehi*np.arctan(arghi))
            sigx[other] = (testlo + testhi)/2.

        return sigx

    def _calc_luminosity(self, zredstr, normmag):
        """
        Internal method to compute luminosity filter

        parameters
        ----------
        zredstr: RedSequenceColorPar
            Red sequence object
        normmag: float
            Normalization magnitude

        returns
        -------
        phi: float array
            phi(x) filter for the cluster

        """
        zind = zredstr.zindex(self.z)
        refind = zredstr.lumrefmagindex(normmag)
        normalization = zredstr.lumnorm[refind, zind]
        mstar = zredstr.mstar(self.z)
        phi_term_a = 10. ** (0.4 * (zredstr.alpha+1.) 
                                 * (mstar-self.neighbors.refmag))
        phi_term_b = np.exp(-10. ** (0.4 * (mstar-self.neighbors.refmag)))
        return phi_term_a * phi_term_b / normalization

    def _calc_bkg_density(self, bkg, cosmo):
        """
        Internal method to compute background filter

        parameters
        ----------
        bkg: Background object
           background
        cosmo: Cosmology object
           cosmology scaling info

        returns
        -------

        bcounts: float array
            b(x) for the cluster
        """
        mpc_scale = np.radians(1.) * cosmo.Dl(0, self.z) / (1 + self.z)**2
        sigma_g = bkg.sigma_g_lookup(self.z, self.neighbors.chisq, 
                                                    self.neighbors.refmag)
        return 2 * np.pi * self.neighbors.r * (sigma_g/mpc_scale**2)

    def calc_richness(self, zredstr, bkg, cosmo, confstr, r0=1.0, beta=0.2):
        """
        compute richness for a cluster

        parameters
        ----------
        zredstr: RedSequenceColorPar object
            Red sequence parameters
        bkg: Background object
            background lookup table
        cosmo: Cosmology object
            From esutil
        confstr: Configuration object
            config info
        r0: float, optional
            Radius -- richness scaling amplitude (default = 1.0 Mpc)
        beta: float, optional
            Radius -- richness scaling index (default = 0.2)

        returns
        -------
        TBD

        """
        maxmag = zredstr.mstar(self.z) - 2.5*np.log10(confstr.lval_reference)
        self.neighbors.r = np.radians(self.neighbors.dist) * cosmo.Dl(0, self.z)
        self.neighbors.chisq = zredstr.calculate_chisq(self.neighbors, self.z)
        rho = chisq_pdf(self.neighbors.chisq, zredstr.ncol)
        nfw = self._calc_radial_profile()
        phi = self._calc_luminosity(zredstr, maxmag) #phi is lumwt in the IDL code
        ucounts = (2*np.pi*self.neighbors.r) * nfw * phi * rho
        bcounts = self._calc_bkg_density(bkg, cosmo)

        #Calculate theta_i. This is reproduced from calclambda_chisq_theta_i.pr
        theta_i = np.ones((len(self.neighbors))) #Default to 1 for theta_i
        eff_lim = np.clip(maxmag,0,zredstr.limmag)
        dmag = eff_lim - self.neighbors.refmag
        calc = dmag < 5.0
        N_calc = np.count_nonzero(calc==True)
        if N_calc > 0: theta_i[calc] = 0.5 + 0.5*erf(dmag[calc]/(np.sqrt(2)*self.neighbors.refmag_err[calc]))
        hi = self.neighbors.refmag > zredstr.limmag
        N_hi = np.count_nonzero(hi==True)
        if N_hi > 0: theta_i[hi] = 0.0
        try:
            w = theta_i * self.neighbors.wvals
        except AttributeError:
            w = np.ones_like(ucounts) * theta_i
        richness_obj = Solver(r0, beta, ucounts, bcounts, self.neighbors.r, w)

        #Call the solving routine
        #this returns three items: lam_obj, p_obj, wt_obj, rlam_obj, theta_r
        lam,p_obj,wt,rlam,theta_r = richness_obj.solve_nfw()
        self.neighbors.theta_i = theta_i
        self.neighbors.w = w
        self.neighbors.wt = wt
        self.neighbors.theta_r = theta_r
        self.richness = lam
        self.rlambda = rlam
        print "p_obj  :",p_obj[:5]
        print "wt_obj :",wt[:5]
        print "theta_r:",theta_r[:5]
        #Record lambda, record p_obj onto the neighbors, 
        return lam


class ClusterCatalog(Catalog): 
    """
    Class to hold a catalog of Clusters

    TBD

    """
    entry_class = Cluster

