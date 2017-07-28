import fitsio
import esutil as eu
import numpy as np
import itertools
import scipy.optimize
import scipy.integrate

from solver_nfw import Solver
from catalog import Catalog, Entry
from utilities import chisq_pdf, calc_theta_i, gaussFunction
from mask import HPMask
from chisq_dist import ChisqDist
from redmapper.redsequence import RedSequenceColorPar
from esutil.cosmology import Cosmo

class Cluster(object):
    """

    Class for a single galaxy cluster, with methods to perform
    computations on individual clusters

    parameters
    ----------
    (TBD)

    """
    def __init__(self, r0 = 1.0, beta = 0.2):
        self.r0     = r0
        self.beta   = beta
        # this should explicitly set our default cosmology
        self.cosmo = Cosmo()
        
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

    def _calc_luminosity(self, normmag):
        """
        Internal method to compute luminosity filter

        parameters
        ----------
        self.zredstr: RedSequenceColorPar
            Red sequence object
        normmag: float
            Normalization magnitude

        returns
        -------
        phi: float array
            phi(x) filter for the cluster

        """
        zind = self.zredstr.zindex(self.z)
        refind = self.zredstr.lumrefmagindex(normmag)
        normalization = self.zredstr.lumnorm[refind, zind]
        mstar = self.zredstr.mstar(self.z)
        phi_term_a = 10. ** (0.4 * (self.zredstr.alpha+1.) 
                                 * (mstar-self.neighbors.refmag))
        phi_term_b = np.exp(-10. ** (0.4 * (mstar-self.neighbors.refmag)))
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
        mpc_scale = np.radians(1.) * self.cosmo.Dl(0, self.z) / (1 + self.z)**2
        sigma_g = self.bkg.sigma_g_lookup(self.z, chisq, refmag)
        return 2 * np.pi * r * (sigma_g/mpc_scale**2)

    def calc_richness(self, mask, noerr = False, index = None):
        """
        compute richness for a cluster

        parameters
        ----------
        mask:  mask object
        noerr: if True, no error calculated
        index: only use part of self.neighbor data

        returns
        -------
        lam: cluster richness

        """
        #set index for sclicing self.neighbors
        if index is not None:
            idx = index
        else:
            idx = np.arange(len(self.neighbors))
            
        maxmag = self.zredstr.mstar(self.z) - 2.5*np.log10(self.confstr.lval_reference)
        self.neighbors.r = np.radians(self.neighbors.dist[idx]) * self.cosmo.Dl(0, self.z)

        # need to clip r at > 1e-6 or else you get a singularity
        self.neighbors.r[idx] = self.neighbors.r[idx].clip(min=1e-6)

        self.neighbors.chisq[idx] = self.zredstr.calculate_chisq(self.neighbors[idx], self.z)
        rho = chisq_pdf(self.neighbors.chisq[idx], self.zredstr.ncol)
        nfw = self._calc_radial_profile()
        phi = self._calc_luminosity(maxmag) #phi is lumwt in the IDL code
        ucounts = (2*np.pi*self.neighbors.r[idx]) * nfw * phi * rho
        bcounts = self._calc_bkg_density(self.neighbors.r[idx], self.neighbors.chisq[idx], 
            self.neighbors.refmag[idx])
        
        theta_i = calc_theta_i(self.neighbors.refmag[idx], self.neighbors.refmag_err[idx], 
            maxmag, self.zredstr.limmag)

        cpars = mask.calc_maskcorr(self.zredstr.mstar(self.z), maxmag, self.zredstr.limmag)
        
        try:
            w = theta_i * self.neighbors.wvals[idx]
        except AttributeError:
            w = np.ones_like(ucounts) * theta_i
    
        richness_obj = Solver(self.r0, self.beta, ucounts, bcounts, self.neighbors.r[idx], w, 
            cpars = cpars, rsig = self.confstr.rsig)
        #Call the solving routine
        #this returns three items: lam_obj, p_obj, wt_obj, rlam_obj, theta_r
        lam, p_obj, wt, rlam, theta_r = richness_obj.solve_nfw()
        
        #error
        bar_p = np.sum(wt**2.0)/np.sum(wt)
        cval = np.sum(cpars*rlam**np.arange(cpars.size, dtype=float)) > 0.0
        
        if not noerr:
            lam_cerr = self.calc_maskcorr_lambdaerr(mask.maskgals, self.zredstr.mstar(self.z), 
                lam, rlam, cval, self.confstr.dldr_gamma)
        else:
            lam_cerr = 0.0
        
        scaleval = np.absolute(lam/np.sum(wt))
        
        lam_unscaled = lam/scaleval
        
        if (lam < 0.0):
            raise ValueError('Richness < 0!')
        else:
           elam = np.sqrt((1-bar_p) * lam_unscaled * scaleval**2. + lam_cerr**2.)
        
        # calculate pcol -- color only.  Don't need to worry about nfw norm!
        ucounts = rho*phi
        
        pcol = ucounts * lam/(ucounts * lam + bcounts)
        bad = np.where((self.neighbors.r[idx] > rlam) | (self.neighbors.refmag[idx] > maxmag) | 
            (self.neighbors.refmag[idx] > self.zredstr.limmag) | (np.isfinite(pcol) == False))
        pcol[bad] = 0.0
        
        #create w, wt
        self.neighbors.w = np.zeros(len(self.neighbors))
        self.neighbors.wt = np.zeros(len(self.neighbors))
        
        self.neighbors.theta_i[idx]  = theta_i
        self.neighbors.w[idx]        = w
        self.neighbors.wt[idx]       = wt
        self.neighbors.theta_r[idx]  = theta_r
        self.richness                = lam
        self.rlambda                 = rlam
        self.elambda                 = elam
        self.cpars                   = cpars
        self.pcol                    = pcol
        #Record lambda, record p_obj onto the neighbors, 
        
        return lam
    
    def calc_maskcorr_lambdaerr(self, maskgals, mstar, lam, rlam, cval, gamma):
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
        lambda_err
        
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
            lambda_err = 0.0
        else:
            p_out       = lam*ucounts[out]/(lam*ucounts[out]+bcounts[out])
            varc0       = (1./lam)*(1./use.size)*np.sum(p_out)
            sigc        = np.sqrt(varc0 - varc0**2.)
            k           = lam**2./np.sum(lambda_p**2.)
            lambda_err  = k*sigc/(1.-self.beta*gamma)
        
        return lambda_err
        
    def redmapper_zlambda(self, zin, mask, maxmag_in=None, calcpz=False, noerr=False, 
        correction = False, ncross=None):
        """
        from redmapper_zlambda.pro
        
        parameters
        ----------
        zin: input redshift
        mask: mask object
        noerr: if True, no error calculated

        returns
        -------
        z_lambda
        """
        
        z_lambda = zin

        maxmag = self.zredstr.mstar(self.z) - 2.5*np.log10(self.confstr.lval_reference)
        if maxmag_in is not None:
            if maxmag_in.size == 1:
                maxmag = maxmag_in
            
        maxrad = 1.2 * self.r0 * 3.**self.beta
        
        i = 0
        niter = 0
        pzdone = False
        
        if noerr:
            z_lambda_e = 0.0
        for pi in range(0, 2):
            #skip second iteration if we're already done
            if pzdone: break
            
            while i < self.confstr.zlambda_maxiter:
                mpc_scale = np.radians(1.) * self.cosmo.Dl(0, z_lambda) / (1 + z_lambda)**2
                
                r = self.neighbors.dist * mpc_scale
        
                in_r, = np.where(r < maxrad)
                
                if in_r.size < 1:
                    z_lambda = -1.0
                    break
                
                lam = self.calc_richness(mask, noerr = True, index = in_r)
                        
                if lam < self.confstr.percolation_minlambda:
                    z_lambda = -1.0
                    break
                    
                wtvals_mod = self.pcol
                
                r_lambda = self.r0 * (lam/100.)**self.beta
                
                if maxmag_in is not None:
                   maxmag = (self.zredstr.mstar(z_lambda) - 
                       2.5 * np.log10(self.confstr.lval_reference))
                
                self.select_neighbors(wtvals_mod, maxrad, maxmag)
                
                #break out of loop if too few neighbors
                if self.zlambda_fail is True:
                    z_lambda_new = -1
                    break
                else:   
                    z_lambda_new = self.zlambda_calcz(z_lambda)
                
                #check for convergence
                if (np.absolute(z_lambda_new-z_lambda) < self.confstr.zlambda_tol or 
                    z_lambda_new < 0.0):
                    break
                    
                z_lambda = z_lambda_new
                i += 1
                
            niter = i
            
            if z_lambda > 0.0 and not noerr:
                if not calcpz:
                    #regular Gaussian error   
                    z_lambda_e = self.zlambda_err(z_lambda)
                    #and check for an error
                    if z_lambda_e < 0.0:
                        self.z_lambda = -1.0
                    pzdone = 1
                    
                else:
                    # set pz and pzbins
                    self.zlambda_pz(z_lambda, wtvals_mod, r_lambda, maxmag)
            
                    #check for bad values and do slow mode if necessary
                    if (self.zlambda_pz[0]/self.zlambda_pz[(self.confstr.npzbins-1)/2] > 0.01 and 
                        self.zlambda_pzbins[0] >= np.amin(self.zredstr.z) + 0.01) or \
                        (self.zlambda_pz[self.confstr.npzbins-1]/
                        self.zlambda_pz[(self.confstr.npzbins-1)/2] > 0.01 and 
                        self.zlambda_pzbins[self.confstr.npzbins-1] <= np.amax(self.zredstr.z)-0.01):
                        
                        self.zlambda_pz(self.confstr, z_lambda, self.confstr.npzbins, 
                            wtvals_mod, r_lambda, maxmag, slow = True)
                        
                    if self.zlambda_pz[0] < 0:
                        #this is bad
                        z_lambda   = -1.0
                        z_lambda_e = -1.0
                    else:
                        m = np.argmax(self.zlambda_pz)
                        p0 = np.array([self.zlambda_pz[m], self.zlambda_pzbins[m], 0.01])
                        
                        coeff, varMatrix = scipy.optimize.curve_fit(gaussFunction, 
                            self.zlambda_pzbins, self.zlambda_pz, p0=p0)
                            
                        if coeff[2] > 0 or coeff[2] > 0.2:
                            z_lambda_e = coeff[2]
                        else:
                            z_lambda_e = self.zlambda_err(z_lambda)
                            
                    # check peak of p(z)...       
                    pmind = np.argmax(self.zlambda_pz)
                    if (np.absolute(self.zlambda_pzbins[pmind] - z_lambda) 
                        < self.confstr.zlambda_tol):
                        pzdone = 1
                    else:
                        print('Warning: z_lambda / p(z) inconsistency detected.')
                        z_lambda = self.zlambda_pzbins[pmind]
                        pzdone = 0
            else:
                z_lambda_e = -1.0
                pzdone = 1
                
                
        #and apply the correction if necessary...
        if correction and z_lambda > 0.0:
            z_lambda, z_lambda_e = zlambda_apply_correction(corrstr, np.sum(wtvals_in), z_lambda ,z_lambda_e)
            #NOT READY - MISSING corrstr
        
        
        self.z_lambda     = z_lambda
        self.z_lambda_err = z_lambda_e
        return self.z_lambda

    def select_neighbors(self, wtvals, maxrad, maxmag):
        """
        select neighbours internal to r < maxrad
        
        parameters
        ----------
        wtvals: weights
        maxrad: maximum radius for considering neighbours
        maxmag: maximum magnitude for considering neighbours
        
        returns
        -------
        sets zrefmagbin, refmag, refmag_err, mag, mag_err, c, pw, targval
        for selected neighbors  
        """
        topfrac = self.confstr.zlambda_topfrac
        
        #we need the zrefmagbin
        nzrefmag    = self.zredstr.refmagbins.size  #zredstr.refmagbins[0].size
        zrefmagbin  = np.clip(np.around(nzrefmag*(self.neighbors.refmag - self.zredstr.refmagbins[0])/
            (self.zredstr.refmagbins[nzrefmag-2] - self.zredstr.refmagbins[0])), 0, nzrefmag-1)
        
        ncount = topfrac*np.sum(wtvals)
        use,   = np.where((self.neighbors.r < maxrad) & (self.neighbors.refmag < maxmag))
        
        if ncount < 3:
            ncount = 3
        
        #exit variable in case use.size < 3
        self.zlambda_fail = False
        if use.size < 3:
            self.zlambda_fail = True
        
        if use.size < ncount:
            ncount = use.size
        
        st      = np.argsort(wtvals[use])[::-1]
        pthresh = wtvals[use[st[np.int(np.around(ncount)-1)]]]
        
        pw  = 1./(np.exp((pthresh-wtvals[use])/0.04)+1)
        gd, = np.where(pw > 1e-3)
        
        self.zlambda_in_rad = use[gd]
        
        self.zlambda_zrefmagbin    = zrefmagbin[self.zlambda_in_rad]
        self.zlambda_refmag        = self.neighbors.refmag[self.zlambda_in_rad]
        self.zlambda_refmag_err    = self.neighbors.refmag_err[self.zlambda_in_rad]
        self.zlambda_mag           = self.neighbors.mag[self.zlambda_in_rad,:]
        self.zlambda_mag_err       = self.neighbors.mag_err[self.zlambda_in_rad,:]
        self.zlambda_c             = self.neighbors.galcol[self.zlambda_in_rad,:]
        self.zlambda_pw            = pw[gd]
        self.zlambda_targval       = 0
    
    def bracket_fn(self, z):
        """
        bracketing function
        """
        likelihoods = self.zredstr.calculate_chisq(self.neighbors[self.zlambda_in_rad], 
            z, calc_lkhd=True)
        t = -np.sum(self.zlambda_pw*likelihoods)
        return t
        
    def delta_bracket_fn(self, z):
        t  = self.bracket_fn(z)
        dt = np.absolute(t-self.zlambda_targval)
        return dt
        
    def zlambda_calcz(self, z_lambda):
        """
        calculate z_lambda
        
        parameters
        ----------
        z_lambda: input
        
        returns
        -------
        z_lambda: output
        """
        nsteps = 10
        steps = np.linspace(0., nsteps*self.confstr.zlambda_parab_step, num = nsteps, 
            dtype = np.float64)+z_lambda-self.confstr.zlambda_parab_step*(nsteps-1)/2
        likes = np.zeros(nsteps)
        for i in range(0, nsteps):
             likes[i] = self.bracket_fn(steps[i])
        fit = np.polyfit(steps, likes, 2)
        
        if fit[0] > 0.0:
            z_lambda = -fit[1]/(2.0 * fit[0])
        else:
            z_lambda = -1.0
        
        z_lambda = np.clip(z_lambda, (steps[0]-self.confstr.zlambda_parab_step), 
            (steps[nsteps-1]+self.confstr.zlambda_parab_step))
        z_lambda = np.clip(z_lambda, self.zredstr.z[0], self.zredstr.z[-2])
        
        return z_lambda
          
    def zlambda_err(self, z_lambda):
        """
        calculate z_lambda error
        parameters
        ----------
        z_lambda: input
        
        returns
        -------
        z_lambda_e: z_lambda error
        """
        minlike = self.bracket_fn(z_lambda) # of course this is negative
        #now we want to aim for minlike+1
        self.zlambda_targval = minlike+1
        
        z_lambda_lo = scipy.optimize.minimize_scalar(self.delta_bracket_fn, 
            bracket = (z_lambda-0.1, z_lambda-0.001), method='bounded', 
            bounds = (z_lambda-0.1, z_lambda-0.001))
        z_lambda_hi = scipy.optimize.minimize_scalar(self.delta_bracket_fn, 
            bracket = (z_lambda+0.001, z_lambda+0.1), method='bounded', 
            bounds = (z_lambda+0.001, z_lambda+0.1))
        z_lambda_e = (z_lambda_hi.x-z_lambda_lo.x)/2.
        
        return z_lambda_e
        
    def zlambda_pz(self, z_lambda, wtvals, maxrad, maxmag, slow = True):
        '''
        set pz and pzbins
        
        parameters
        ----------
        z_lambda: input
        wtvals: weights
        maxrad: maximum radius for considering neighbours
        maxmag: maximum magnitude for considering neighbours
        slow: slow or fast mode
        '''
        minlike = self.bracket_fn(z_lambda)
        #4 sigma
        self.zlambda_targval=minlike+16
        
        if not slow:
            #do we need both directions?  For speed, just do one...
            #also don't need as tight tolerance.
            #This is very approximate, but is fine...
            
            #z_lambda_hi = scipy.optimize.minimize_scalar(self.delta_bracket_fn, 
            #bracket = (z_lambda+0.001,z_lambda+0.05,z_lambda+0.15), method='brent', tol=0.001)
            #WORKS,  but above doesn't so use this here to stay consistent
            z_lambda_hi = scipy.optimize.minimize_scalar(self.delta_bracket_fn, 
                bracket = (z_lambda+0.001, z_lambda+0.15), method='bounded', 
                bounds = (z_lambda+0.001, z_lambda+0.15))
                
            dz = np.clip((z_lambda_hi.x - z_lambda), 0.005, 0.15) # minimal to consider
            pzbinsize = 2.*dz/(self.confstr.npzbins-1)
            pzbins = pzbinsize*np.arange(self.confstr.npzbins)+z_lambda - dz
            
        else:
            #super-slow-mode
            #find the center
            pk  = -self.bracket_fn(z_lambda) #REPLACE WITH *minlike
            pz0 = self.zredstr.volume_factor[self.zredstr.zindex(z_lambda)]
            
            #go to lower redshift
            dztest = 0.05
            
            lowz  = z_lambda - dztest
            ratio = 1.0
            while (lowz >= np.amin(self.zredstr.z) and (ratio > 0.01)):
                val = - self.bracket_fn(lowz)
                
                ln_lkhd = val - pk
                pz = np.exp(val - pk)*self.zredstr.volume_factor[self.zredstr.zindex(lowz)]

                ratio=pz/pz0

                if (ratio > 0.01):
                    lowz=lowz-dztest
                    
            lowz = np.clip(lowz, np.amin(self.zredstr.z), None)
            
            highz = z_lambda + dztest
            ratio = 1.0
            while (highz <= np.amax(self.zredstr.z) and (ratio > 0.01)):
                val  = - self.bracket_fn(highz)
        
                ln_lkhd = val - pk
                pz = np.exp(ln_lkhd)*self.zredstr.volume_factor[self.zredstr.zindex(highz)]

                ratio=pz/pz0

                if (ratio > 0.01):
                    highz=highz+dztest
            
            highz = np.clip(highz, None, np.amax(self.zredstr.z))

            pzbinsize = (highz - lowz)/(self.confstr.npzbins-1)

            pzbins = pzbinsize*np.arange(self.confstr.npzbins) + lowz

            # and finally offset so that we're centered on z_lambda.  Important!
            zmind = np.argmin(np.absolute(pzbins - z_lambda))
            pzbins = pzbins - (pzbins[zmind] - z_lambda)
            
        ln_lkhd = np.zeros(self.confstr.npzbins)
        for i in range(0, self.confstr.npzbins):
            likelihoods = self.zredstr.calculate_chisq(self.neighbors[self.zlambda_in_rad], 
                pzbins[i], calc_lkhd=True)
            ln_lkhd[i] = np.sum(self.zlambda_pw*likelihoods)
            
        ln_lkhd = ln_lkhd - np.amax(ln_lkhd)
        pz = np.exp(ln_lkhd) * self.zredstr.volume_factor[self.zredstr.zindex(pzbins)]
        
        #now normalize
        n = scipy.integrate.simps(pz, pzbins)
        pz=pz/n
        
        self.zlambda_pzbins     = pzbins
        self.zlambda_pzbinsize  = pzbinsize
        self.zlambda_pz         = pz
        
    def zlambda_apply_correction(corrstr, lambda_in, z_lambda, z_lambda_e, noerr=False):
        """
        apply corrections to modify z_lambda & uncertainty, pz and pzbins
        NOT READY - MISSING corrstr
        
        parameters
        ----------
        corrstr: correction object
        z_lambda: input
        z_lambda_e: error
        noerr: if True, no error calculated
        """
        
        niter = corrstr.offset[0].size

        for i in range(0, niter):
    
            correction = (corrstr.offset[i] + corrstr.slope[i] * 
                np.log(lambda_in/confstr.zlambda_pivot))
            extra_err = np.interp(corrstr.scatter[i], corrstr.z, z_lambda)        

            dz = np.interp(correction, corrstr.z, z_lambda)
    
            z_lambda_new = z_lambda + dz
            
            #and recalculate z_lambda_e
            if not noerr:
                z_lambda_e_new = np.sqrt(z_lambda_e**2 + extra_err**2.)
            else:
                z_lambda_e_new = z_lambda_e

            if self.confstr.npzbins is not None:
                #do space density expansion...        
                #modify width of bins by expansion...
                #also shift the center to the new z_lambda...

                #allow for an offset between the peak and the center...
                offset  = self.zlambda_pzbins[(self.confstr.npzbins-1)/2] - z_lambda
                pdz     = self.zlambda_pzbinsize*np.sqrt(extra_err**2.+z_lambda_e**2)/z_lambda_e
        
                #centered on z_lambda...
                self.zlambda_pzbins = (pdz*np.arange(self.confstr.npzbins) + z_lambda_new - 
                    pdz*(self.confstr.npzbins-1)/2. + offset*pdz/self.zlambda_pzbinsize)
    
                #and renormalize
                n = scipy.integrate.simps(self.zlambda_pzbins, self.zlambda_pz)
                self.zlambda_pz = self.zlambda_pz/n
                
        return z_lambda_new, z_lambda_e_new
    
        
class ClusterCatalog(Catalog): 
    """
    Class to hold a catalog of Clusters

    TBD

    """
    entry_class = Cluster

