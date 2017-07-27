import fitsio
import esutil as eu
import numpy as np
import itertools
import scipy.optimize
import scipy.integrate

from solver_nfw import Solver
from catalog import Catalog, Entry
from utilities import chisq_pdf, calc_theta_i
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
    def __init__(self,confstr, r0 = 1.0, beta = 0.2):
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

    def calc_richness(self, confstr, mask, 
        noerr = False, index = None):
        """
        compute richness for a cluster

        parameters
        ----------
        self.zredstr: RedSequenceColorPar object
            Red sequence parameters
        bkg: Background object
            background lookup table
        cosmo: Cosmology object
            From esutil
        confstr: Configuration object
            containing configuration info
        r0: float, optional
            Radius -- richness scaling amplitude (default = 1.0 Mpc)
        beta: float, optional
            Radius -- richness scaling index (default = 0.2)
        noerr: 
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
            
        maxmag = self.zredstr.mstar(self.z) - 2.5*np.log10(confstr.lval_reference)
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
            cpars = cpars, rsig = confstr.rsig)
        #Call the solving routine
        #this returns three items: lam_obj, p_obj, wt_obj, rlam_obj, theta_r
        lam, p_obj, wt, rlam, theta_r = richness_obj.solve_nfw()
        
        #error
        bar_p = np.sum(wt**2.0)/np.sum(wt)
        cval = np.sum(cpars*rlam**np.arange(cpars.size, dtype=float)) > 0.0
        
        if not noerr:
            lam_cerr = self.calc_maskcorr_lambdaerr(mask.maskgals, self.zredstr.mstar(self.z), 
                lam, rlam, cval, confstr.dldr_gamma)
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
    
    def calc_maskcorr_lambdaerr(self, maskgals, mstar,
         lam, rlam, cval, gamma):
        """
        Calculate richness error
        
        parameters
        ----------
        mstar    :
        lam      : Richness
        rlam     :
        cval     :
        gamma    : Local slope of the richness profile of galaxy clusters
        cosmo    : Cosmology object
                    From esutil
        

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
            p_out = lam*ucounts[out]/(lam*ucounts[out]+bcounts[out])
            varc0 = (1./lam)*(1./use.size)*np.sum(p_out)
            sigc = np.sqrt(varc0 - varc0**2.)
            k = lam**2./total(lambda_p**2.)
            lambda_err = k*sigc/(1.-self.beta*gamma)
        
        return lambda_err
        
    def redmapper_zlambda(self, confstr, zin, mask, z_lambda_e=None,
        maxmag_in=None, corrstr=None, npzbins=None, noerr=None, ncross=None):
        
        """
        from redmapper_zlambda.pro
        
        parameters
        ----------
        

        returns
        -------
        
        """
        
        z_lambda=zin

        maxmag = self.zredstr.mstar(self.z) - 2.5*np.log10(confstr.lval_reference)
        if maxmag_in is not None:
            if maxmag_in.size == 1:
                maxmag = maxmag_in
        if npzbins is None:
            npzbins=0
        else:
            pzbins = np.full(npzbins, -1.0)
            pzvals = pzbins
            
        maxrad = 1.2 * self.r0 * 3.**self.beta
        #300./100. = 3.
        
        i = 0
        done = False
        niter = 0
        pzdone = False
        
        if noerr:
            z_lambda_e = 0.0
        for pi in range(0, 2):
            #skip second iteration if we're already done
            if pzdone: continue
            
            while i < confstr.zlambda_maxiter:
                print z_lambda
                mpc_scale = np.radians(1.) * self.cosmo.Dl(0, z_lambda) / (1 + z_lambda)**2
                
                r = self.neighbors.dist * mpc_scale
        
                in_r, = np.where(r < maxrad)
                
                if in_r.size < 1:
                    z_lambda = -1.0
                    break
                
                lam = self.calc_richness(confstr, mask, noerr = True, index = in_r)
                        
                if lam < confstr.percolation_minlambda:
                    z_lambda = -1.0
                    break
                    
                wtvals_mod = self.pcol
                
                r_lambda=self.r0*(lam/100.)**self.beta
                
                if maxmag_in is not None:
                   maxmag = self.zredstr.mstar(z_lambda)-2.5*np.log10(confstr.lval_reference)
                
                #create class with neighbours internal to r < maxrad
                neighbors_in_r = neighbors_in(self.neighbors[in_r], self.zredstr, confstr, z_lambda, wtvals_mod, maxrad, maxmag)
                
                if neighbors_in_r.exit is True:
                    z_lambda_new = -1
                else:   
                    z_lambda_new = neighbors_in_r.zlambda_calcz()
                
                #check for convergence
                if np.absolute(z_lambda_new-z_lambda) < confstr.zlambda_tol or z_lambda_new < 0.0:
                    break
                    
                z_lambda = z_lambda_new
                i += 1
                
            print z_lambda
            niter = i
            
            if z_lambda > 0.0:
                if npzbins == 0 and not noerr:
                    #regular Gaussian error   
                    z_lambda_e = neighbors_in_r.zlambda_err()
                    #and check for an error
                    if z_lambda_e < 0.0:
                        z_lambda = -1.0
                elif npzbins > 0:
                    pzvals = self.zlambda_pz(confstr, z_lambda, npzbins, wtvals_mod, r_lambda, maxmag, pzbins, in_r)
            
                    #check for bad values
                    if (pzvals[0]/pzvals[(npzbins-1)/2] > 0.01 and 
                        pzbins[0] >= np.amin(self.zredstr.z) + 0.01) or \
                        (pzvals[npzbins-1]/pzvals[(npzbins-1)/2] > 0.01 and 
                        pzbins[npzbins-1] <= np.amax(self.zredstr.z)-0.01):
                        
                        pzvals = self.zlambda_pz(confstr, z_lambda, npzbins, wtvals_mod, r_lambda, maxmag, pzbins, in_r, slow = True)
                        
                    if pzvals[0] < 0:
                        #this is bad
                        z_lambda = -1.0
                        z_lambda_e = -1.0
                    else:
                        #res = gaussfit_rm(pzbins, pzvals, a, nterms=3, status=status)
                        
                        p0 = np.array([pzvals.size, np.median(pzvals), np.std(pzvals)])

                        coeff,varMatrix = scipy.optimize.curve_fit(gaussFunction, pzbins, pzvals, p0=p0)
                        # a = coeff ??
                        
                        if coeff[2] > 0 or coeff[2] > 0.2:
                            z_lambda_e = coeff[2]
                        else:
                            z_lambda_e = neighbors_in_r.zlambda_err()
                            
                # check peak of p(z)...
                if npzbins == 0:
                    # we didn't do p(z) so we have no way to check here.
                    pzdone = 1                
                else:
                    print pzvals, pzbins
                    #pm = np.amax(pzvals)
                    pmind = np.argmax(pzvals)
                    if np.absolute(pzbins[pmind] - z_lambda) < confstr.zlambda_tol:
                        pzdone = 1
                    else:
                        print('Warning: z_lambda / p(z) inconsistency detected.')
                        z_lambda = pzbins[pmind]
                        pzdone = 0
            else:
                z_lambda_e = -1.0
                pzdone = 1
                
                
        #and apply the correction if necessary...
        if corrstr is not None and z_lambda > 0:
            redmapper_zlambda_apply_correction(confstr,corrstr,total(wtvals_in),z_lambda,z_lambda_e,pzbins=pzbins,pzvals=pzvals,noerr=noerr)

        if ncross is not None and z_lambda > confstr.zrange[0] and z_lambda < confstr.zrange[1]:
            ncross = redmapper_zlambda_ncross(confstr,self.zredstr,bkg,zin,refmag_total,refmag_total_err,col_or_flux_arr,magerr_or_ivar_arr,dis,ebv,r0,beta,maskgals,z_lambda,wvals=wvals,maxmag_in=maxmag_in,zreds=zreds,zred_errs=zred_errs,zred_chisqs=zred_chisqs)
        
        self.z_lambda = z_lambda
        self.z_lambda_err = z_lambda_e
        return z_lambda

    def gaussFunction(x, *p):
       A, mu, sigma = p
       return A*np.exp(-(x-mu)**2./(2.*sigma**2))
            
class neighbors_in(object):
    """
    class with neighbours internal to r < maxrad for z_lambda calculation
    """
    def __init__(self, neighbors, zredstr, confstr, z_in, wtvals, maxrad, maxmag):
        topfrac=confstr.zlambda_topfrac
        
        #we need the zrefmagbin
        nzrefmag = zredstr.refmagbins.size  #zredstr.refmagbins[0].size
        zrefmagbin = np.clip(np.around(nzrefmag*(neighbors.refmag - zredstr.refmagbins[0])/
            (zredstr.refmagbins[nzrefmag-2] - zredstr.refmagbins[0])), 0, nzrefmag-1)
        
        
        ncount=topfrac*np.sum(wtvals)
        use, = np.where((neighbors.r < maxrad) & (neighbors.refmag < maxmag))
        
        if ncount < 3:
            ncount = 3
        
        #exit variable in case use.size < 3
        self.exit = False
        if use.size < 3:
            self.exit = True
        
        if use.size < ncount:
            ncount = use.size
        
        st = np.argsort(wtvals[use])[::-1]
        pthresh = wtvals[use[st[np.int(np.around(ncount)-1)]]]
        
        pw = 1./(np.exp((pthresh-wtvals[use])/0.04)+1)
        gd, = np.where(pw > 1e-3)
        
        self.z_lambda      = z_in
        self.zredstr       = zredstr
        self.zrefmagbin    = zrefmagbin[use[gd]]
        self.refmag        = neighbors.refmag[use[gd]]
        self.refmag_err    = neighbors.refmag_err[use[gd]]
        self.mag           = neighbors.mag[use[gd],:]
        self.mag_err       = neighbors.mag_err[use[gd],:]
        self.c             = neighbors.galcol(zredstr.ncol)[use[gd],:]
        self.pw            = pw[gd]
        self.targval       = 0
        self.parab_step  = confstr.zlambda_parab_step
        
    def bracket_fn(self, z):
        likelihoods = self.zredstr.calculate_chisq(self, z, calc_lkhd=True)
        t=-np.sum(self.pw*likelihoods)
        return t
        
    def delta_bracket_fn(self, z):
        t  = self.bracket_fn(z)
        dt = np.absolute(t-self.targval)
        return dt
        
    def zlambda_calcz(self):
        #calculate z_lambda
        nsteps = 10
        steps = np.linspace(0., nsteps*self.parab_step, num = nsteps, dtype = np.float64)+self.z_lambda-self.parab_step*(nsteps-1)/2
        likes = np.zeros(nsteps)
        for i in range(0, nsteps):
             likes[i] = self.bracket_fn(steps[i])
        #print steps, likes
        #plt.plot(steps, likes)
        #plt.show()
        fit = np.polyfit(steps,likes,2)
        
        if fit[0] > 0.0:
            z_lambda = -fit[1]/(2.0*fit[0])
        else:
            z_lambda = -1.0
        
        z_lambda = np.clip(z_lambda, (steps[0]-self.parab_step), (steps[nsteps-1]+self.parab_step))
        z_lambda = np.clip(z_lambda, self.zredstr.z[0], self.zredstr.z[-2])
        
        return z_lambda
            
    def zlambda_err(self):
        #calculate error
        minlike = self.bracket_fn(self.z_lambda) # of course this is negative
        #now we want to aim for minlike+1
        self.targval = minlike+1
        
        z_lambda_lo, fval_lo = scipy.optimize.minimize_scalar(self.delta_bracket_fn, bracket = (self.z_lambda-0.1,self.z_lambda-0.02,self.z_lambda-0.001), method='brent', tol=0.0002)
        z_lambda_hi, fval_hi = scipy.optimize.minimize_scalar(self.delta_bracket_fn, bracket = (self.z_lambda+0.001,self.z_lambda+0.02,self.z_lambda+0.1), method='brent', tol=0.0002)
        z_lambda_e = (z_lambda_hi-z_lambda_lo)/2.
        
        return z_lambda_e

    def zlambda_pz(self, confstr, z_lambda, npzbins, wtvals, maxrad, maxmag,pzbins,idx,slow=False):
        '''
        NEEDS FIXING
        '''
        minlike = self.bracket_fn(z_lambda)
        #4 sigma
        neighbors_in_str.targval=minlike+16
        
        if not slow:
            #do we need both directions?  For speed, just do one...
            #also don't need as tight tolerance.
            #This is very approximate, but is fine...
            
            z_lambda_hi, fval_hi = scipy.optimize.minimize_scalar(self.delta_bracket_fn, bracket = (self.z_lambda+0.001,self.z_lambda+0.05,self.z_lambda+0.15),method='brent', tol=0.001)
            dz = np.clip((z_lambda_hi-self.z_lambda), 0.005, 0.15) # minimal to consider
            
            pzbinsize = 2.*dz/(npzbins-1)
            
            pzbins = pzbinsize*np.arange(npzbins)+ self.z_lambda - dz
        else:
            #super-slow-mode
            #find the center
            pk=-1*neighbors_in_str.bracket_fn(self.z_lambda) #REPLACE WITH *minlike
            zbin = np.clip(np.round((self.z_lambda-self.zredstr.z[0])/(self.zredstr.z[1]-self.zredstr.z[0])), 0, (self.zredstr.z.size-1))
            
            pz0=self.zredstr.volume_factor[zbin]
            
            #go to lower redshift
            dztest = 0.05
            
            lowz = self.z_lambda - dztest
            ratio = 1.0
            while (lowz >= np.amin(self.zredstr.z) and (ratio > 0.01)):
                val = - self.bracket_fn(lowz)
        
                zbin = np.clip(np.round((lowz-self.zredstr.z[0])/(self.zredstr.z[1]-self.zredstr.z[0])), 0, (self.zredstr.z.size-1))
                
                ln_lkhd = val - pk
                pz=np.exp(val - pk)*self.zredstr.volume_factor[zbin]

                ratio=pz/pz0

                if (ratio > 0.01):
                    lowz=lowz-dztest
                    
            lowz = np.clip(lowz, np.amin(zredstr.z), None)
            
            highz = self.z_lambda + dztest
            ratio = 1.0
            while (highz <= np.amax(self.zredstr.z) and (ratio > 0.01)):
                val  = - self.bracket_fn(highz)
                zbin = np.clip(np.around((lowz-self.zredstr.z[0])/(self.zredstr.z[1]-self.zredstr.z[0])), 0, (self.zredstr.z.size-1))
        
                ln_lkhd = val - pk
                pz=exp(ln_lkhd)*self.zredstr.volume_factor[zbin]
                endelse

                ratio=pz/pz0

                if (ratio > 0.01):
                    highz=highz+dztest
            
            highz = np.clip(highz, None, np.amax(self.zredstr.z))

            pzbinsize = (highz - lowz)/(npzbins-1)

            pzbins = pzbinsize*np.arange(npzbins) + lowz

            # and finally offset so that we're centered on z_lambda.  Important!
            #zm = np.amin(np.absolute(pzbins-z_lambda))
            zmind = np.argmin(np.absolute(pzbins-z_lambda))
            pzbins=pzbins-(pzbins[zmind]-z_lambda)
            
        #if sconst or scol:
        #    ch=fltarr(npzbins)
        #
        #    for i in range(0, npzbins):
        #        if not scol:
        #            #regular
        #            chisqs=calclambda_chisq_dist(np.transpose(self.zredstr[zbins[i]].covmat[self.zrefmagbin,:,:]),
        #                                         self.zredstr[zbins[i]].c,self.zredstr[zbins[i]].slope,
        #                                         self.zredstr[zbins[i]].pivotmag,self.refmag,self.merr,
        #                                         self.c,chisq = True,
        #                                         lupcorr = self.zredstr[zbins[i]].lupcorr[:,self.zrefmagbin].flatten())    
        #    
        #        else:
        #            # single color
        #            d=reform(c_arr) - (self.zredstr[zbin].slope[0]*(neighbors_in_str.refmag - self.zredstr[zbins[i]].pivotmag) + self.zredstr[zbins[i]].c[0])
        #            ctot=(self.zredstr[zbins[i]].covmat[neighbors_in_str.zrefmagbin] + neighbors_in_str.merr[0]**2. + neighbors_in_str.merr[1]**2.)
        #            #chisqs=d*(1./ctot)*d
        #            chisqs=d**2 / ctot
        #            
        #        
        #        #this may be correct...
        #        ch[i] = np.sum(self.pw*chisqs)
        #        
        #    ch = ch-np.amin(ch)
        #    pz = np.exp(-ch/2.)*self.zredstr.volume_factor[zbins]
        #    
        #else:
        ln_lkhd = np.zeros(npzbins)
        for i in range(0, npzbins):
            likelihoods = self.zredstr.calculate_chisq(self, pzbins, calc_lkhd=True)
            ln_lkhd[i] = np.sum(self.pw*likelihoods)
            
        ln_lkhd = ln_lkhd - np.amax(ln_lkhd)
        pz = np.exp(ln_lkhd) * self.zredstr.volume_factor[zbins]
        
        #now normalize
        n = scipy.integrate.simps(pz, pzbins)
        pz=pz/n

        return pz
        
        
class ClusterCatalog(Catalog): 
    """
    Class to hold a catalog of Clusters

    TBD

    """
    entry_class = Cluster

