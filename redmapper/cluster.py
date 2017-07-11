import fitsio
import esutil as eu
import numpy as np
import itertools
from solver_nfw import Solver
from catalog import Catalog, Entry
from utilities import chisq_pdf
from scipy.special import erf
from numpy import random
from mask import HPMask

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

    def calc_richness(self, zredstr, bkg, cosmo, confstr, mask, r0=1.0, beta=0.2, noerr = True):
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

        # need to clip r at > 1e-6 or else you get a singularity
        self.neighbors.r = self.neighbors.r.clip(min=1e-6)

        self.neighbors.chisq = zredstr.calculate_chisq(self.neighbors, self.z)
        rho = chisq_pdf(self.neighbors.chisq, zredstr.ncol)
        nfw = self._calc_radial_profile()
        phi = self._calc_luminosity(zredstr, maxmag) #phi is lumwt in the IDL code
        ucounts = (2*np.pi*self.neighbors.r) * nfw * phi * rho
        bcounts = self._calc_bkg_density(bkg, cosmo)
        
        theta_i = self.calc_theta_i(self.neighbors.refmag, self.neighbors.refmag_err, maxmag, zredstr.limmag)
        
        cpars = mask.calc_maskcorr(zredstr.mstar(self.z), maxmag, zredstr.limmag, confstr)
        #should this be handed a different limmag?
        
        try:
            w = theta_i * self.neighbors.wvals
        except AttributeError:
            w = np.ones_like(ucounts) * theta_i
        
        richness_obj = Solver(r0, beta, ucounts, bcounts, self.neighbors.r, w, cpars = cpars, rsig = confstr.rsig)
        #DELETE ONCE VALUES ARE FIXED
        richness_obj = Solver(r0, beta, ucounts, bcounts, self.neighbors.r, w)

        #Call the solving routine
        #this returns three items: lam_obj, p_obj, wt_obj, rlam_obj, theta_r
        lam,p_obj,wt,rlam,theta_r = richness_obj.solve_nfw()
        #---> does this replace remaining IDL code?
        
        #error
        bar_p = np.sum(wt**2.0)/np.sum(wt) #ASSUME wtvals = wt
        cval = np.sum(cpars*rlam**np.arange(cpars.size, dtype=float)) > 0.0
        
        alpha = 1.0 #WHAT IS ALPHA?
        dof = 1.0 #WHAT IS DOF?
        gamma = 1.0 #WHAT IS gamma?
        if not noerr:
            lam_cerr = self.calc_maskcorr_lambdaerr(mask.maskgals, zredstr.mstar(self.z), alpha ,maxmag ,dof, zredstr.limmag, 
                lam, rlam ,self.z ,bkg, wt, cval, r0, beta, gamma, cosmo)
        else:
            lam_cerr = 0.0
        self.neighbors.theta_i = theta_i
        self.neighbors.w = w
        self.neighbors.wt = wt
        self.neighbors.theta_r = theta_r
        self.richness = lam
        self.rlambda = rlam
        #Record lambda, record p_obj onto the neighbors, 
        return lam
    
    def calc_theta_i(self, mag, mag_err, maxmag, limmag):
        """
        Calculate theta_i. This is reproduced from calclambda_chisq_theta_i.pr
        
        parameters
        ----------
        mag:
        mag_err:
        maxmag:
        limmag:

        returns
        -------
        theta_i:
        """
 
        theta_i = np.ones((len(mag))) #Default to 1 for theta_i
        eff_lim = np.clip(maxmag,0,limmag)
        dmag = eff_lim - mag
        calc = dmag < 5.0
        N_calc = np.count_nonzero(calc==True)
        if N_calc > 0: theta_i[calc] = 0.5 + 0.5*erf(dmag[calc]/(np.sqrt(2)*mag_err[calc]))
        hi = mag > limmag
        N_hi = np.count_nonzero(hi==True)
        if N_hi > 0: theta_i[hi] = 0.0
        return theta_i
    
        
    def calc_maskcorr_lambdaerr(self, maskgals, mstar, alpha ,maxmag ,dof, limmag, 
                lam, rlam ,z ,bkg, wt, cval, r0, beta, gamma, cosmo):
        """
        
        parameters
        ----------
        maskgals :
        mstar    :
        alpha    :
        maxmag   :
        dof      :
        limmag   :
        lam      :
        rlam     :
        z        :
        bkg      :
        wt       :
        cval     :
        r0       :
        beta     :
        gamma    :
        cosmo    :

        returns
        -------
        lambda_err:
        
        """
        use, = np.where(maskgals.r < rlam)
        
        mark    = maskgals.mark[use]
        refmag  = mstar+maskgals.m[use]
        cwt     = maskgals.cwt[use]
        nfw     = maskgals.nfw[use]
        lumwt   = maskgals.lumwt[use]
        chisq   = maskgals.chisq[use]
        r       = maskgals.r[use]
        
        logrc   = np.log(rlam)
        norm    = np.exp(1.65169 - 0.547850*logrc + 0.138202*logrc**2. -0.0719021*logrc**3.- 0.0158241*logrc**4.-0.000854985*logrc**5.)
        nfw     = norm*maskgals.nfw[use]

        ucounts = cwt*nfw*lumwt

        faint, = np.where(refmag >= limmag)
        refmag_for_bcounts = refmag
        if faint.size > 0:
             refmag_for_bcounts[faint] = limmag-0.01
             
        bcounts = self.calc_bcounts(z, r, chisq , refmag_for_bcounts, bkg, cosmo)
        
        out = np.where((refmag > limmag) or (mark == 0)) # ,comp=in) - necessary?
        
        if (out.size == 0 or cval > 0.01):
            lambda_err = 0.0
        else:
        
            p_out = lam*ucounts[out]/(lam*ucounts[out]+bcounts[out])
            varc0 = (1./lam)*(1./use.size)*np.sum(p_out)
            sigc = np.sqrt(varc0 - varc0**2.)
            k = lam**2./total(lambda_p**2.)
            lambda_err = k*sigc/(1.-beta*gamma)
        
        return lambda_err

    def calc_bcounts(self, z, r, chisq, refmag_for_bcounts, bkg, cosmo, allow0='allow0'):
        """
        
        parameters
        ----------         :
        z                  :
        r                  :
        chisq              :
        refmag_for_bcounts :
        bkg                :
        cosmo              :
        allow0             :

        returns
        -------
        bcounts:
        
        """
        H0 = cosmo._H0
        nchisqbins  = bkg.chisqbins.size
        chisqindex  = np.around((chisq-bkg.chisqbins[0])*nchisqbins/((bkg.chisqbins[nchisqbins-1]+bkg.chisqbinsize)-bkg.chisqbins[0]))
        nrefmagbins = bkg.refmagbins.size
        refmagindex = np.around((self.neighbors.refmag-bkg.refmagbins[0])*nrefmagbins/((bkg.refmagbins[nrefmagbins-1]+bkg.refmagbinsize)-bkg.refmagbins[0]))
        #assume refmag = self.neighbors.refmag
        print self.neighbors.refmag, self.neighbors.refmag.shape
        #check for overruns
        badchisq, = np.where((chisqindex < 0) | (chisqindex >= nchisqbins))
        if (badchisq.size > 0): # $ important?
          chisqindex[badchisq] = 0
        badrefmag, = np.where((refmagindex < 0) | (refmagindex >= nrefmagbins))
        if (badrefmag.size > 0): # $ important?
          refmagindex[badrefmag] = 0
        
        #print np.around((z-bkg.zbins[0])/(bkg.zbins[1]-bkg.zbins[0]))
        ind = np.clip(np.around((z-bkg.zbins[0])/(bkg.zbins[1]-bkg.zbins[0])), 0, (bkg.zbins.size-1))
        print ind, chisqindex.size, refmagindex.size, bkg.sigma_g.shape
        #FIXME
        sigma_g = bkg.sigma_g[np.full(chisqindex.size, ind), chisqindex, refmagindex]
        print sigma_g

        #no matter what, these should be infinities
        if (badchisq.size >  0):
            sigma_g[badchisq]= float("inf")
        if (badrefmag.size > 0):
            sigma_g[badrefmag] = float("inf")
        
        
        if not allow0:
            badcombination = np.where((sigma_g == 0.0) & (chisq > 5.0))
            if (badcombination.size > 0):
                sigma_g[badcombination] = float("inf")
        
        bcounts=2. * np.pi * r * (sigma_g / c**2.) #WHAT IS C?

        return bcounts


class ClusterCatalog(Catalog): 
    """
    Class to hold a catalog of Clusters

    TBD

    """
    entry_class = Cluster

