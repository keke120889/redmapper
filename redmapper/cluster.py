import fitsio
import esutil as eu
import numpy as np
import itertools
from solver_nfw import Solver
from catalog import Catalog, Entry
from utilities import chisq_pdf, calc_theta_i
from mask import HPMask

class Cluster(object):
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

    def calc_richness(self, zredstr, bkg, cosmo, confstr, mask, r0=1.0, beta=0.2, 
        noerr = False):
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
            containing configuration info
        r0: float, optional
            Radius -- richness scaling amplitude (default = 1.0 Mpc)
        beta: float, optional
            Radius -- richness scaling index (default = 0.2)

        returns
        -------
        lam: cluster richness

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
        bcounts = self._calc_bkg_density(bkg, self.neighbors.r, self.neighbors.chisq, 
            self.neighbors.refmag, cosmo)
        
        theta_i = calc_theta_i(self.neighbors.refmag, self.neighbors.refmag_err, 
            maxmag, zredstr.limmag)

        cpars = mask.calc_maskcorr(zredstr.mstar(self.z), maxmag, zredstr.limmag)
        
        try:
            w = theta_i * self.neighbors.wvals
        except AttributeError:
            w = np.ones_like(ucounts) * theta_i
    
        richness_obj = Solver(r0, beta, ucounts, bcounts, self.neighbors.r, w, 
            cpars = cpars, rsig = confstr.rsig)
        #Call the solving routine
        #this returns three items: lam_obj, p_obj, wt_obj, rlam_obj, theta_r
        lam, p_obj, wt, rlam, theta_r = richness_obj.solve_nfw()
        
        #error
        bar_p = np.sum(wt**2.0)/np.sum(wt)
        cval = np.sum(cpars*rlam**np.arange(cpars.size, dtype=float)) > 0.0
        
        if not noerr:
            lam_cerr = self.calc_maskcorr_lambdaerr(mask.maskgals, zredstr.mstar(self.z), 
                zredstr, lam, rlam,bkg, cval, beta, confstr.dldr_gamma, cosmo)
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
        bad = np.where((self.neighbors.r > rlam) | (self.neighbors.refmag > maxmag) | 
            (self.neighbors.refmag > zredstr.limmag) | (np.isfinite(pcol) == False))
        pcol[bad] = 0.0
        
        self.neighbors.theta_i  = theta_i
        self.neighbors.w        = w
        self.neighbors.wt       = wt
        self.neighbors.theta_r  = theta_r
        self.richness           = lam
        self.rlambda            = rlam
        self.elambda            = elam
        self.cpars              = cpars
        self.pcol               = pcol
        #Record lambda, record p_obj onto the neighbors, 
        return lam
    
    def calc_maskcorr_lambdaerr(self, maskgals, mstar, zredstr,
         lam, rlam ,bkg, cval, beta, gamma, cosmo):
        """
        Calculate richness error
        
        parameters
        ----------
        mstar    :
        zredstr  : RedSequenceColorPar object
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
        dof = zredstr.ncol
        limmag = zredstr.limmag
        
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
            lambda_err = k*sigc/(1.-beta*gamma)
        
        return lambda_err
        
    def redmapper_zlambda(confstr,zredstr,bkg,zin,mask,cosmo,z_lambda_e='z_lambda_e',maxmag_in='maxmag_in',corrstr='corrstr',npzbins='npzbins',pzbins='pzbins',pzvals='pzvals',noerr='noerr',ncross='ncross'):
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

        if maxmag_in is not None:
            if maxmag_in.size == 1:
                maxmag = maxmag_in
        #if wvals is not None or wvals.size == 0:       #HAVE self.neighbors.wvals
        #     wvals = np.ones(refmag_total.size)
        #     #YEAH??
        if npzbins is not None or npzbins.size == 0:
            npzbins=0
        else:
            pzbins = np.full(npzbins, -1.0)
            pzvals = pzbins
            
        maxrad = 1.2 * confstr.percolation_r0 * 3.**confstr.percolation_beta
        #300./100. = 3.
        
        i = 0
        done = False
        niter = 0
        pzdone = 0
        
        if noerr:
            z_lambda_e = 0.0
        for pi in range(0, 1):
            #skip second iteration if we're already done
            if pzdone: continue
            
            while i < confstr.zlambda_maxiter and not done:
                #calclambda_z_rad,z_lambda,mpcscale,1./60.,/rtok,/silent
                #ignore?
                mpc_scale = np.radians(1.) * cosmo.Dl(0, self.z) / (1 + self.z)**2
                #YEAH??
                r=dis*mpcscale
        
                in_r = np.where(r < maxrad)
                #thought about creating an object here that contains all in data
                #- but how would I implement that?
                
                if in_r.size < 1:
                    z_lambda = -1.0
                    done = 1
                    continue
                
                #if confstr.use_lambda_zred: ----> not implementing zred in calc_richness so do we need this at all?
                lam = calc_richness(zredstr, bkg, cosmo, confstr, mask, 
                    r0=confstr.percolation_r0, beta=confstr.percolation_beta, 
                    noerr = True)
                        #confstr,zredstr,bkg,z_lambda,
                        #zreds[in_r],zred_errs[in_r],zred_chisqs[in_r],refmag_total[in_r],
                        #refmag_total_err[in_r],r[in_r],wtvals=wtvals_in,
                        #r0=confstr.percolation_r0,beta=confstr.percolation_beta,
                        #wvals=wvals[in_r],maskgals=maskgals,maxmag_in=maxmag_in,
                        #pcol=pcol_in,noerr = True)
                        #----> so are we trying to only use the ones indexed [in_r]?
            
                #else:   
                #    lam= calclambda_chisq_cluster_lambda(confstr,zredstr,bkg,z_lambda,$
                #           col_or_flux_arr[:,in_r],magerr_or_ivar_arr[:,in_r],$
                #           refmag_total[in_r],refmag_total_err[in_r],$
                #           refmag_rs[in_r],refmag_rs_err[in_r],$
                #           r[in_r],wtvals=wtvals_in,$
                #           r0=confstr.percolation_r0,$
                #           beta=confstr.percolation_beta,$
                #           wvals=wvals[in_r],maskgals=maskgals,$
                #           chisqs=chisqs_in,maxmag_in=maxmag_in,$
                #           pcol=pcol_in,ebv=ebv[in_r],/noerr)
               
                if lam < confstr.percolation_minlambda:
                    z_lambda = -1.0
                    done = 1
                    continue
                    
                wtvals_mod = self.pcol
                
                r_lambda=confstr.percolation_r0*(lam/100.)**confstr.percolation_beta
                if maxmag_in.size == 0:
                   maxmag = zredstr.mstar(z_lambda)-2.5*np.log10(confstr.lval_reference)
                
                z_lambda_new=redmapper_zlambda_calcz(confstr,zredstr,z_lambda,refmag_total[in_r],refmag_rs[in_r],col_or_flux_arr[:,in_r],
                                magerr_or_ivar_arr[:,in_r],r[in_r],ebv[in_r],wtvals_mod,r_lambda,maxmag)
                z_lambda_new = np.clip(z_lambda_new, zredstr[0].z, zredstr[n_elements(zredstr)-1].z)
                if np.absolute(z_lambda_new-z_lambda) < confstr.zlambda_tol or z_lambda_new < 0.0:
                    done = 1
                    
                z_lambda = z_lambda_new
                i += 1
        
            niter = i
            
            if z_lambda > 0.0:
                if npzbins == 0 and not noerr:
                    #regular Gaussian error   
                    z_lambda_e = redmapper_zlambda_err(confstr,zredstr,z_lambda,refmag_total[in_r],refmag_rs[in_r],col_or_flux_arr[:,in_r],magerr_or_ivar_arr[:,in_r],r[in_r],ebv[in_r],wtvals_mod,r_lambda,maxmag)
                    #and check for an error
                    if z_lambda_e < 0.0:
                        z_lambda = -1.0
                elif npzbins > 0:
                    pzvals = redmapper_zlambda_pz(confstr,zredstr,z_lambda,npzbins,refmag_total[in_r],refmag_rs[in_r],col_or_flux_arr[:,in_r],magerr_or_ivar_arr[:,in_r],r[in_r],ebv[in_r],wtvals_mod,r_lambda,maxmag,pzbins)
            
                    #check for bad values
                    if (pzvals[0]/pzvals[(npzbins-1)/2] > 0.01 and 
                        pzbins[0] >= np.amin(zredstr.z) + 0.01) or \
                        (pzvals[npzbins-1]/pzvals[(npzbins-1)/2] > 0.01 and 
                        pzbins[npzbins-1] <= np.amax(zredstr.z)-0.01):
                        
                        pzvals = redmapper_zlambda_pz(confstr,zredstr,z_lambda,npzbins,refmag_total[in_r],refmag_rs[in_r],col_or_flux_arr[:,in_r],magerr_or_ivar_arr[:,in_r],r[in_r],ebv[in_r],wtvals_mod,r_lambda,maxmag,pzbins,slow = True)
                        
                    if pzvals[0] < 0:
                        #this is bad
                        z_lambda = -1.0
                        z_lambda_e = -1.0
                    else:
                        res = gaussfit_rm(pzbins,pzvals,a,nterms=3,status=status)
                
                        if status == 0 and (a[2] > 0 or a[2] > 0.2):
                            z_lambda_e = a[2]
                        else:
                            z_lambda_e = redmapper_zlambda_err(confstr,zredstr,z_lambda,refmag_total[in_r],refmag_rs[in_r],col_or_flux_arr[:,in_r],magerr_or_ivar_arr[:,in_r],r[in_r],ebv[in_r],wtvals_mod,r_lambda,maxmag)
                            
                # check peak of p(z)...
                if npzbins == 0:
                    # we didn't do p(z) so we have no way to check here.
                    pzdone = 1                
                else:
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
            ncross = redmapper_zlambda_ncross(confstr,zredstr,bkg,zin,refmag_total,refmag_total_err,refmag_rs,refmag_rs_err,col_or_flux_arr,magerr_or_ivar_arr,dis,ebv,r0,beta,maskgals,z_lambda,wvals=wvals,maxmag_in=maxmag_in,zreds=zreds,zred_errs=zred_errs,zred_chisqs=zred_chisqs)

        return z_lambda

        
        

class neighbours_in(object):
    def __init__(self, neigbours):
        #self.
        pass

class ClusterCatalog(Catalog): 
    """
    Class to hold a catalog of Clusters

    TBD

    """
    entry_class = Cluster

