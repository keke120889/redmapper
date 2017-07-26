import fitsio
import esutil as eu
import numpy as np
import itertools
from solver_nfw import Solver
from catalog import Catalog, Entry
from utilities import chisq_pdf, calc_theta_i
from mask import HPMask
from scipy.optimize import brent, minimize_scalar
from scipy.integrate import simps
from chisq_dist import ChisqDist
from redmapper.redsequence import RedSequenceColorPar

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
        self.zredstr = RedSequenceColorPar(confstr.parfile)
        
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

    def _calc_bkg_density(self, bkg, r, chisq, refmag, cosmo):
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
        sigma_g = bkg.sigma_g_lookup(self.z, chisq, refmag)
        return 2 * np.pi * r * (sigma_g/mpc_scale**2)

    def calc_richness(self, bkg, cosmo, confstr, mask, 
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
        self.neighbors.r = np.radians(self.neighbors.dist[idx]) * cosmo.Dl(0, self.z)

        # need to clip r at > 1e-6 or else you get a singularity
        self.neighbors.r[idx] = self.neighbors.r[idx].clip(min=1e-6)

        self.neighbors.chisq[idx] = self.zredstr.calculate_chisq(self.neighbors[idx], self.z)
        rho = chisq_pdf(self.neighbors.chisq[idx], self.zredstr.ncol)
        nfw = self._calc_radial_profile()
        phi = self._calc_luminosity(maxmag) #phi is lumwt in the IDL code
        ucounts = (2*np.pi*self.neighbors.r[idx]) * nfw * phi * rho
        bcounts = self._calc_bkg_density(bkg, self.neighbors.r[idx], self.neighbors.chisq[idx], 
            self.neighbors.refmag[idx], cosmo)
        
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
                lam, rlam,bkg, cval, confstr.dldr_gamma, cosmo)
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
        self.richness           = lam
        self.rlambda            = rlam
        self.elambda            = elam
        self.cpars              = cpars
        self.pcol               = pcol
        #Record lambda, record p_obj onto the neighbors, 
        
        return lam
    
    def calc_maskcorr_lambdaerr(self, maskgals, mstar,
         lam, rlam ,bkg, cval, gamma, cosmo):
        """
        Calculate richness error
        
        parameters
        ----------
        mstar    :
        self.zredstr  : RedSequenceColorPar object
                    Red sequence parameters
        dof      : Degrees of freedom / number of collumns
        limmag   : Limiting Magnitude
        lam      : Richness
        rlam     :
        bkg      : Background object
                   background lookup table
        cval     :
        beta     :
        gamma    : Local slope of the richness profile of galaxy clusters
        cosmo    : Cosmology object
                    From esutil
        refmag   : Reference magnitude

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
        
        bcounts = self._calc_bkg_density(bkg, r, chisq , refmag_for_bcounts, cosmo)
        
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
        
    def redmapper_zlambda(self, confstr, bkg, zin, mask, cosmo, z_lambda_e=None,
        maxmag_in=None, corrstr=None, npzbins=None, noerr=None, ncross=None):
        #refmag_total,refmag_total_err,refmag_rs,refmag_rs_err,col_or_flux_arr,magerr_or_ivar_arr,dis,ebv,r0,beta,
        
        """
        MISSING:
        refmag_total,refmag_total_err, = self.neighbors.refmag
        
        col_or_flux_arr,
        magerr_or_ivar_arr
        
        
        """
        
        z_lambda=zin

        #m = np.amin(self.neighbors.dist)
        #minind = np.argmin(self.neighbors.dist) #assuming self.neighbors.dist = dis
        maxmag = self.zredstr.mstar(self.z) - 2.5*np.log10(confstr.lval_reference)
        if maxmag_in is not None:
            if maxmag_in.size == 1:
                maxmag = maxmag_in
        if npzbins is not None:
            if npzbins.size == 0:
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
        for pi in range(0, 1):
            #skip second iteration if we're already done
            if pzdone: continue
            
            while i < confstr.zlambda_maxiter and not done:
                print z_lambda
                mpc_scale = np.radians(1.) * cosmo.Dl(0, z_lambda) / (1 + z_lambda)**2
                
                r = self.neighbors.dist * mpc_scale
        
                in_r, = np.where(r < maxrad)
                
                if in_r.size < 1:
                    z_lambda = -1.0
                    done = 1
                    continue
                
                lam = self.calc_richness(bkg, cosmo, confstr, mask, 
                    noerr = True, index = in_r)
                #print lam
                        
                if lam < confstr.percolation_minlambda:
                    z_lambda = -1.0
                    done = 1
                    continue
                    
                wtvals_mod = self.pcol
                
                r_lambda=self.r0*(lam/100.)**self.beta
                
                if maxmag_in is not None:
                   maxmag = self.zredstr.mstar(z_lambda)-2.5*np.log10(confstr.lval_reference)
                
                z_lambda_new = self.zlambda_calcz(confstr,z_lambda,
                                wtvals_mod,r_lambda,maxmag, in_r)
                z_lambda_new = np.clip(z_lambda_new, self.zredstr.z[0], self.zredstr.z[-1])
                
                #check for convergence
                if np.absolute(z_lambda_new-z_lambda) < confstr.zlambda_tol or z_lambda_new < 0.0:
                    done = 1
                    
                z_lambda = z_lambda_new
                i += 1
            print confstr.zlambda_tol
            niter = i
            if z_lambda > 0.0:
                if npzbins == 0 and not noerr:
                    #regular Gaussian error   
                    z_lambda_e = self.zlambda_err(confstr, z_lambda, wtvals_mod, r_lambda, maxmag, in_r)
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
                        res = gaussfit_rm(pzbins, pzvals, a, nterms=3, status=status)
                
                        if status == 0 and (a[2] > 0 or a[2] > 0.2):
                            z_lambda_e = a[2]
                        else:
                            z_lambda_e = self.zlambda_err(confstr, z_lambda, wtvals_mod, r_lambda, maxmag, in_r)
                            
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

        return z_lambda

    def zlambda_calcz(self, confstr, z_in, wtvals, maxrad, maxmag, idx):
        neighbors_in_str = neighbors_in(self.neighbors[idx], self.zredstr, confstr, z_in, wtvals, maxrad, maxmag)
        
        #calculate z_lambda
        nsteps = 10
        steps = np.linspace(0., nsteps*confstr.zlambda_parab_step, num = nsteps, dtype = np.float64)+z_in-confstr.zlambda_parab_step*(nsteps-1)/2
        likes = np.zeros(nsteps)
        for i in range(0, nsteps-1):
             likes[i] = neighbors_in_str.bracket_fn(steps[i])
        
        fit = np.polyfit(steps,likes,2)
        
        if fit[0] > 0.0:
            z_lambda = -fit[1]/(2.0*fit[0])
        else:
            z_lambda = -1.0
        
        z_lambda = np.clip(z_lambda, (steps[0]-confstr.zlambda_parab_step), (steps[nsteps-1]+confstr.zlambda_parab_step))
        
        return z_lambda
            
    def zlambda_err(self, confstr, z_lambda, wtvals, maxrad, maxmag, idx):
        neighbors_in_str = neighbors_in(self.neighbors[idx], self.zredstr, confstr, z_lambda, wtvals, maxrad, maxmag)
        
        #calculate error
        minlike = neighbors_in_str.bracket_fn(z_lambda) # of course this is negative
        #now we want to aim for minlike+1
        neighbors_in_str.targval = minlike+1
        
        print neighbors_in_str(z_lambda-0.1),neighbors_in_str(z_lambda-0.02),neighbors_in_str(z_lambda-0.001)
        
        z_lambda_lo, fval_lo = minimize_scalar(neighbors_in_str, bracket = (z_lambda-0.1,z_lambda-0.02,z_lambda-0.001), method='brent', tol=0.0002)
        z_lambda_hi, fval_hi = minimize_scalar(neighbors_in_str, bracket = (z_lambda+0.001,z_lambda+0.02,z_lambda+0.1), method='brent', tol=0.0002)
        z_lambda_e = (z_lambda_hi-z_lambda_lo)/2.
        
        return z_lambda_e

    def zlambda_pz(self, confstr, z_lambda, npzbins, wtvals, maxrad, maxmag,pzbins,idx,slow=False):
        
        neighbors_in_str = neighbors_in(self.neighbors[idx], self.zredstr, confstr, z_lambda, wtvals, maxrad, maxmag)
        
        minlike = neighbors_in_str.bracket_fn(z_lambda)
        #4 sigma
        neighbors_in_str.targval=minlike+16
        
        if not slow:
            #do we need both directions?  For speed, just do one...
            #also don't need as tight tolerance.
            #This is very approximate, but is fine...
            
            z_lambda_hi, fval_hi = minimize_scalar(neighbors_in_str, bracket = (z_lambda+0.001,z_lambda+0.05,z_lambda+0.15),method='brent', tol=0.001)
            dz = np.clip((z_lambda_hi-z_lambda), 0.005, 0.15) # minimal to consider
            
            pzbinsize = 2.*dz/(npzbins-1)
            
            pzbins = pzbinsize*np.arange(npzbins)+ z_lambda-dz
        else:
            #super-slow-mode
            #find the center
            pk=-1*neighbors_in_str.bracket_fn(z_lambda) #REPLACE WITH *minlike
            zbin = np.clip(np.round((z_lambda-self.zredstr[0].z)/(self.zredstr[1].z-self.zredstr[0].z)), 0, (self.zredstr.size-1))
            
            pz0=self.zredstr.volume_factor[zbin]
            
            #go to lower redshift
            dztest = 0.05
            
            lowz=z_lambda-dztest
            ratio = 1.0
            while (lowz >= np.amin(self.zredstr.z) and (ratio > 0.01)):
                val = -1 * neighbors_in_str.bracket_fn(lowz)
        
                zbin = np.clip(np.round((lowz-self.zredstr.z[0])/(self.zredstr.z[1]-self.zredstr.z[0])), 0, (self.zredstr.size-1))

                if sconst:
                    #ch = val-pk
                    pz = np.exp(-(val-pk)/2.)*self.zredstr.volume_factor[zbin]
                else:
                    #ln_lkhd = val - pk
                    pz=np.exp(val - pk)*self.zredstr.volume_factor[zbin]

                ratio=pz/pz0

                if (ratio > 0.01):
                    lowz=lowz-dztest
            
            highz = np.clip(highz, None, np.amax(self.zredstr.z))

            pzbinsize = (highz - lowz)/(npzbins-1)

            pzbins = pzbinsize*np.arange(npzbins) + lowz

            # and finally offset so that we're centered on z_lambda.  Important!
            #zm = np.amin(np.absolute(pzbins-z_lambda))
            zmind = np.argmin(np.absolute(pzbins-z_lambda))
            pzbins=pzbins-(pzbins[zmind]-z_lambda)
            
        zbins = np.clip(np.round((pzbins-self.zredstr.z[0])/(self.zredstr[1].z-self.zredstr[0].z)), 0, (self.zredstr.size-1))
            
        if sconst or scol:
            ch=fltarr(npzbins)

            for i in range(0, npzbins-1):
                if not scol:
                    #regular
                    chisqs=calclambda_chisq_dist(np.transpose(self.zredstr[zbins[i]].covmat[neighbors_in_str.zrefmagbin,:,:]),
                                                 self.zredstr[zbins[i]].c,self.zredstr[zbins[i]].slope,
                                                 self.zredstr[zbins[i]].pivotmag,neighbors_in_str.refmag,neighbors_in_str.merr,
                                                 neighbors_in_str.c,chisq = True,
                                                 lupcorr = self.zredstr[zbins[i]].lupcorr[:,neighbors_in_str.zrefmagbin].flatten())    
            
                else:
                    # single color
                    d=reform(c_arr) - (self.zredstr[zbin].slope[0]*(neighbors_in_str.refmag - self.zredstr[zbins[i]].pivotmag) + self.zredstr[zbins[i]].c[0])
                    ctot=(self.zredstr[zbins[i]].covmat[neighbors_in_str.zrefmagbin] + neighbors_in_str.merr[0]**2. + neighbors_in_str.merr[1]**2.)
                    #chisqs=d*(1./ctot)*d
                    chisqs=d**2 / ctot
                    
                
                #this may be correct...
                ch[i] = np.sum(neighbors_in_str.pw*chisqs)
                
            ch = ch-np.amin(ch)
            pz = np.exp(-ch/2.)*self.zredstr.volume_factor[zbins]
            
        else:
            ln_lkhd = np.zeros(npzbins)
            for i in range(0, npzbins-1):
                likelihoods=calclambda_chisq_dist(transpose(self.zredstr[zbins[i]].covmat[neighbors_in_str.zrefmagbin,:,:]),
                                                  self.zredstr[zbins[i]].c,self.zredstr[zbins[i]].slope,
                                                  self.zredstr[zbins[i]].pivotmag,neighbors_in_str.refmag,neighbors_in_str.merr,
                                                  neighbors_in_str.c,
                                                  lupcorr = self.zredstr[zbins[i]].lupcorr[:,neighbors_in_str.zrefmagbin].flatten())

                ln_lkhd[i] = np.sum(neighbors_in_str.pw*likelihoods)
                
            ln_lkhd = ln_lkhd - np.amax(ln_lkhd)
            pz = np.exp(ln_lkhd) * self.zredstr.volume_factor[zbins]
            
            #now normalize
            n = simps(pz, pzbins)
            pz=pz/n

            return pz
            
class neighbors_in(object):
    def __init__(self, neighbors, zredstr, confstr, z_in, wtvals, maxrad, maxmag):
        #is this a constant scatter model?
        topfrac=confstr.zlambda_topfrac
        
        #we need the zrefmagbin
        nzrefmag = zredstr.refmagbins.size  #zredstr.refmagbins[0].size
        zrefmagbin = np.clip(np.around(nzrefmag*(neighbors.refmag - zredstr.refmagbins[0])/
            (zredstr.refmagbins[nzrefmag-2] - zredstr.refmagbins[0])), 0, nzrefmag-1)
        
        
        ncount=topfrac*np.sum(wtvals)
        use, = np.where((neighbors.r < maxrad) & (neighbors.refmag < maxmag))
        
        if ncount < 3:
            ncount = 3

        if use.size < 3:
            return -1

        if use.size < ncount:
            ncount = use.size

        st = np.argsort(wtvals[use])[::-1]
        pthresh = wtvals[use[st[np.int(np.around(ncount)-1)]]] #???
        
        pw = 1./(np.exp((pthresh-wtvals[use])/0.04)+1)
        gd, = np.where(pw > 1e-3)
        
        self.zredstr       = zredstr
        self.zrefmagbin    = zrefmagbin[use[gd]]
        self.refmag        = neighbors.refmag[use[gd]]
        self.refmag_err    = neighbors.refmag_err[use[gd]]
        self.mag           = neighbors.mag[use[gd],:]
        self.mag_err       = neighbors.mag_err[use[gd],:]
        self.c             = neighbors.galcol[use[gd],:]
        self.pw            = pw[gd]
        self.targval       = 0
        
    def bracket_fn(self, p):
        likelihoods = self.zredstr.calculate_chisq(self, p, calc_lkhd=True)
        t=-np.sum(self.pw*likelihoods)
        return t
        
    def __call__(self, p):
        t  = self.bracket_fn(p)
        dt = np.absolute(t-self.targval)
        return dt
        
        
        
class ClusterCatalog(Catalog): 
    """
    Class to hold a catalog of Clusters

    TBD

    """
    entry_class = Cluster

