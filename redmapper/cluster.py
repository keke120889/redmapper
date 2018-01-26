import fitsio
import esutil as esutil
import numpy as np
import itertools
import scipy.optimize
import scipy.integrate
import copy

from solver_nfw import Solver
from catalog import Catalog, Entry
from utilities import chisq_pdf, calc_theta_i
from mask import HPMask
from chisq_dist import ChisqDist
from redmapper.redsequence import RedSequenceColorPar
from esutil.cosmology import Cosmo
from galaxy import GalaxyCatalog

cluster_dtype_base = [('RA', 'f8'),
                      ('DEC', 'f8'),
                      ('Z', 'f4'),
                      ('REFMAG', 'f4'),
                      ('REFMAG_ERR', 'f4'),
                      ('LAMBDA', 'f4'),
                      ('LAMBDA_E', 'f4'),
                      ('Z_LAMBDA', 'f4'),
                      ('Z_LAMBDA_E', 'f4'),
                      ('CG_SPEC_Z', 'f4'),
                      ('Z_SPEC_INIT', 'f4'),
                      ('Z_INIT', 'f4'),
                      ('R_LAMBDA', 'f4'),
                      ('SCALEVAL', 'f4'),
                      ('MASKFRAC', 'f4'),
                      ('CHISQ', 'f4'),
                      ('Z_LAMBDA_NITER', 'i2'),
                      ('EBV_MEAN', 'f4'),
                      ('LNLAMLIKE', 'f4'),
                      ('LNCGLIKE', 'f4'),
                      ('LNLIKE', 'f4'),
                      ('RA_ORIG', 'f8'),
                      ('DEC_ORIG', 'f8'),
                      ('DLAMBDA_DZ', 'f4'),
                      ('DLAMBDA_DZ2', 'f4'),
                      ('DLAMBDAVAR_DZ', 'f4'),
                      ('DLAMBDAVAR_DZ2', 'f4'),
                      ('Z_LAMBDA_RAW', 'f4'),
                      ('Z_LAMBDA_E_RAW', 'f4'),
                      ('LIM_EXPTIME', 'f4'),
                      ('LIM_LIMMAG', 'f4'),
                      ('LIM_LIMMAG_HARD', 'f4')]

class Cluster(Entry):
    """

    Class for a single galaxy cluster, with methods to perform
    computations on individual clusters

    parameters
    ----------
    (TBD)

    """
    def __init__(self, cat_vals=None, r0=None, beta=None, confstr=None, zredstr=None, bkg=None, neighbors=None, cosmo=None):

        if cat_vals is None:
            #cat_vals = np.zeros(1, dtype=[('RA', 'f8'),
            #                              ('DEC', 'f8'),
            #                              ('Z', 'f8')])
            if confstr is not None:
                cat_vals = np.zeros(1, dtype=confstr.cluster_dtype)
            else:
                # This might lead to bugs down the line, but let's try
                cat_vals = np.zeros(1, dtype=cluster_dtype_base)

        # Start by taking catalog values and stuffing them into a nice Entry format
        # we need to extend if necessary?  Or just the catalog?
        super(Cluster, self).__init__(cat_vals)

        self.r0 = 1.0 if r0 is None else r0
        self.beta = 0.2 if beta is None else beta

        # FIXME: this should explicitly set our default cosmology
        self.cosmo = cosmo
        if self.cosmo is None: self.cosmo = Cosmo()
        self.confstr = confstr
        self.zredstr = zredstr
        self.bkg = bkg
        self.set_neighbors(neighbors)

    def reset(self):
        """
        """

        # reset values to defaults
        self.Lambda = -1.0
        self.z_lambda = -1.0

    def set_neighbors(self, neighbors):
        """
        """

        if (neighbors.__class__ is not GalaxyCatalog and neighbors is not None):
            raise ValueError("Cluster neighbors must be a GalaxyCatalog")

        self.neighbors = None
        if (neighbors is not None):
            self.neighbors = copy.deepcopy(neighbors)

            # extra fields
            neighbor_extra_dtype = [('R', 'f8'),
                                    ('DIST', 'f8'),
                                    ('CHISQ', 'f8'),
                                    ('PFREE', 'f8'),
                                    ('THETA_I', 'f8'),
                                    ('THETA_R', 'f8'),
                                    ('P', 'f8'),
                                    ('PCOL', 'f8'),
                                    ('PMEM', 'f8'),
                                    ('INDEX', 'i8')]

            dtype_augment = [dt for dt in neighbor_extra_dtype if dt[0] not in self.neighbors.dtype.names]
            if len(dtype_augment) > 0:
                self.neighbors.add_fields(dtype_augment)

            if 'PFREE' in [dt[0] for dt in dtype_augment]:
                # The PFREE is new, so we must set it to 1s
                self.neighbors.pfree[:] = 1.0

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
        #indices, dists = galcat.match(self, radius) # self refers to the cluster
        indices, dists = galcat.match_one(self.ra, self.dec, radius)

        #self.neighbors = galcat[indices]
        #Dist is arcmin???, R is Mpc/h

        ## FIXME: check and add fields
        #new_fields = [('DIST', 'f8'), ('R', 'f8'), ('PMEM', 'f8'),
        #                ('CHISQ', 'f8')]
        #self.neighbors.add_fields(new_fields)
        #self.neighbors.dist = dists
        self.set_neighbors(galcat[indices])
        self.neighbors.dist = dists
        self.neighbors.index = indices

    def clear_neighbors(self):
        """
        """

        # Clear out the memory used by the neighbors.
        # Here or elsewhere to copy to members?
        self.neighbors = None

    def neighbors_to_members(self):
        """
        """

        # copy the neighbors to a members subset
        pass

    def _calc_radial_profile(self, idx=None, rscale=0.15):
        """
        internal method for computing radial profile weights

        parameters
        ----------
        idx: integer array (optional)
           indices to compute
        rscale: float
           r_s for nfw profile

        returns
        -------
        sigx: array of floats
           sigma(x)
        """
        if idx is None:
            idx = np.arange(len(self.neighbors))

        corer = 0.1
        x, corex = self.neighbors.r[idx]/rscale, corer/rscale
        sigx = np.zeros(self.neighbors.r[idx].size)

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

    def _calc_luminosity(self, normmag, idx=None):
        """
        Internal method to compute luminosity filter

        parameters
        ----------
        self.zredstr: RedSequenceColorPar
            Red sequence object
        normmag: float
            Normalization magnitude
        idx: int array (optional)
            Indices to compute

        returns
        -------
        phi: float array
            phi(x) filter for the cluster

        """

        if idx is None:
            idx = np.arange(len(self.neighbors))

        zind = self.zredstr.zindex(self.z)
        refind = self.zredstr.lumrefmagindex(normmag)
        normalization = self.zredstr.lumnorm[refind, zind]
        mstar = self.zredstr.mstar(self.z)
        phi_term_a = 10. ** (0.4 * (self.zredstr.alpha+1.)
                                 * (mstar-self.neighbors.refmag[idx]))
        phi_term_b = np.exp(-10. ** (0.4 * (mstar-self.neighbors.refmag[idx])))
        return phi_term_a * phi_term_b / normalization

    def _calc_bkg_density(self, r, chisq, refmag):
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
        mpc_scale = np.radians(1.) * self.cosmo.Da(0, self.z)
        sigma_g = self.bkg.sigma_g_lookup(self.z, chisq, refmag)
        return 2 * np.pi * r * (sigma_g/mpc_scale**2)

    def calc_richness(self, mask, calc_err=True, index=None, record_values=True):
        """
        compute richness for a cluster

        parameters
        ----------
        mask:  mask object
        calc_err: if False, no error calculated
        index: only use part of self.neighbor data

        returns
        -------
        lam: cluster richness

        """
        #set index for slicing self.neighbors
        if index is not None:
            idx = index
        else:
            idx = np.arange(len(self.neighbors))

        self.mstar = self.zredstr.mstar(self.z)

        maxmag = self.mstar - 2.5*np.log10(self.confstr.lval_reference)
        self.neighbors.r = np.radians(self.neighbors.dist) * self.cosmo.Da(0, self.z)

        # need to clip r at > 1e-6 or else you get a singularity
        self.neighbors.r[idx] = self.neighbors.r[idx].clip(min=1e-6)

        self.neighbors.chisq[idx] = self.zredstr.calculate_chisq(self.neighbors[idx], self.z)
        rho = chisq_pdf(self.neighbors.chisq[idx], self.zredstr.ncol)
        nfw = self._calc_radial_profile(idx=idx)
        phi = self._calc_luminosity(maxmag, idx=idx) #phi is lumwt in the IDL code
        ucounts = (2*np.pi*self.neighbors.r[idx]) * nfw * phi * rho
        bcounts = self._calc_bkg_density(self.neighbors.r[idx], self.neighbors.chisq[idx],
                                         self.neighbors.refmag[idx])

        theta_i = calc_theta_i(self.neighbors.refmag[idx], self.neighbors.refmag_err[idx],
                               maxmag, self.zredstr.limmag)

        cpars = mask.calc_maskcorr(self.mstar, maxmag, self.zredstr.limmag)

        try:
            #w = theta_i * self.neighbors.wvals[idx]
            w = theta_i * self.neighbors.pfree[idx]
        except AttributeError:
            w = theta_i * np.ones_like(ucounts)

        richness_obj = Solver(self.r0, self.beta, ucounts, bcounts, self.neighbors.r[idx], w,
                              cpars = cpars, rsig = self.confstr.rsig)

        # Call the solving routine
        # this returns five items: lam_obj, p, pmem, rlam, theta_r
        # pmem = p * pfree * theta_i * theta_r
        lam, p, pmem, rlam, theta_r = richness_obj.solve_nfw()

        # error
        bar_pmem = np.sum(pmem**2.0)/np.sum(pmem)
        # cval = np.sum(cpars*rlam**np.arange(cpars.size, dtype=float)) > 0.0
        cval = np.clip(np.sum(cpars * rlam**np.arange(cpars.size, dtype=float)),
                       0.0, None)

        if calc_err:
            lam_cerr = self.calc_lambdaerr(mask.maskgals, self.mstar,
                                           lam, rlam, cval, self.confstr.dldr_gamma)
        else:
            lam_cerr = 0.0

        self.scaleval = np.absolute(lam/np.sum(pmem))

        lam_unscaled = lam/self.scaleval

        if (lam < 0.0):
            lam_err = -1.0
        else:
            lam_err = np.sqrt((1-bar_pmem) * lam_unscaled * self.scaleval**2. + lam_cerr**2.)

        # calculate pcol -- color only.  Don't need to worry about nfw norm!
        ucounts = rho*phi

        pcol = ucounts * lam/(ucounts * lam + bcounts)
        bad, = np.where((self.neighbors.r[idx] > rlam) | (self.neighbors.refmag[idx] > maxmag) |
                        (self.neighbors.refmag[idx] > self.zredstr.limmag) | (~np.isfinite(pcol)))
        pcol[bad] = 0.0

        # reset before setting subsets
        self.neighbors.theta_i[:] = 0.0
        self.neighbors.theta_r[:] = 0.0
        self.neighbors.p[:] = 0.0
        self.neighbors.pcol[:] = 0.0
        self.neighbors.pmem[:] = 0.0

        # and set the values
        self.neighbors.theta_i[idx] = theta_i
        self.neighbors.theta_r[idx] = theta_r
        self.neighbors.p[idx] = p
        self.neighbors.pcol[idx] = pcol
        self.neighbors.pmem[idx] = pmem

        self.Lambda = lam
        self.r_lambda = rlam
        if calc_err:
            self.Lambda_e = lam_err
        else:
            self.Lambda_e = 0.0

        return lam

    def calc_lambdaerr(self, maskgals, mstar, lam, rlam, cval, gamma):
        """
        Calculate richness error

        parameters
        ----------
        maskgals : maskgals object
        lam      : Richness
        rlam     :
        cval     :
        gamma    : Local slope of the richness profile of galaxy clusters

        returns
        -------
        lam_err

        """
        dof = self.zredstr.ncol
        limmag = self.zredstr.limmag

        use, = np.where(maskgals.r < rlam)

        mark    = maskgals.mark[use]
        refmag  = mstar + maskgals.m[use]
        cwt     = maskgals.cwt[use]
        nfw     = maskgals.nfw[use]
        lumwt   = maskgals.lumwt[use]
        chisq   = maskgals.chisq[use]
        r       = maskgals.r[use]

        # normalizing nfw
        logrc   = np.log(rlam)
        norm    = np.exp(1.65169 - 0.547850*logrc + 0.138202*logrc**2. - 
            0.0719021*logrc**3. - 0.0158241*logrc**4.-0.000854985*logrc**5.)
        nfw     = norm*nfw

        ucounts = cwt*nfw*lumwt

        #Set too faint galaxy magnitudes close to limiting magnitude
        faint, = np.where(refmag >= limmag)
        refmag_for_bcounts = np.copy(refmag)
        refmag_for_bcounts[faint] = limmag-0.01

        bcounts = self._calc_bkg_density(r, chisq , refmag_for_bcounts)

        out, = np.where((refmag > limmag) | (mark == 0))

        if out.size == 0 or cval < 0.01:
            lam_err = 0.0
        else:
            p_out       = lam*ucounts[out]/(lam*ucounts[out]+bcounts[out])
            varc0       = (1./lam)*(1./use.size)*np.sum(p_out)
            sigc        = np.sqrt(varc0 - varc0**2.)
            k           = lam**2./np.sum(lambda_p**2.)
            lam_err  = k*sigc/(1.-self.beta*gamma)

        return lam_err

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        # This returns a copy of the cluster, and note that the neighbors will
        # be deepcopied which is what we want.
        return Cluster(r0=self.r0,
                       beta=self.beta,
                       confstr=self.confstr,
                       zredstr=self.zredstr,
                       bkg=self.bkg,
                       neighbors=self.neighbors,
                       cosmo=self.cosmo)

class ClusterCatalog(Catalog):
    """
    Class to hold a catalog of Clusters

    TBD

    """
    entry_class = Cluster

    # Okay, here's the plan for the cluster catalog...

    # It is simply an array of numbers ... all the fields we want
    #   So we need to override and append all those values when we load a catalog
    # And it will also have an array of members, if necessary
    # And on demand it will return a cluster object which it needs to initialize...


    def __init__(self, *arrays, **kwargs):
        super(ClusterCatalog, self).__init__(*arrays)

        self.r0 = None if 'r0' not in kwargs else kwargs['r0']
        self.beta = None if 'beta' not in kwargs else kwargs['beta']
        self.zredstr = None if 'zredstr' not in kwargs else kwargs['zredstr']
        self.confstr = None if 'confstr' not in kwargs else kwargs['confstr']
        self.bkg = None if 'bkg' not in kwargs else kwargs['bkg']
        self.cosmo = None if 'cosmo' not in kwargs else kwargs['cosmo']

        if self.confstr is not None:
            cluster_dtype = self.confstr.cluster_dtype
        else:
            cluster_dtype = cluster_dtype_base

        # and if confstr is set then use that cluster_dtype because that
        #  will have all the other stuff filled as well.
        dtype_augment = [dt for dt in cluster_dtype if dt[0] not in self._ndarray.dtype.names]
        if len(dtype_augment) > 0:
            self.add_fields(dtype_augment)

    @classmethod
    def from_catfile(cls, filename, **kwargs):
        """
        """
        cat = fitsio.read(filename, ext=1)

        return cls(cat, **kwargs)

    def __getitem__(self, key):
        if isinstance(key, int):
            # Note that if we have members, we can associate them with the cluster
            #  here.
            return Cluster(cat_vals=self._ndarray.__getitem__(key),
                           r0=self.r0,
                           beta=self.beta,
                           zredstr=self.zredstr,
                           confstr=self.confstr,
                           bkg=self.bkg,
                           cosmo=self.cosmo)
        else:
            raise ValueError("Don't know what to do with this key type yet?")
