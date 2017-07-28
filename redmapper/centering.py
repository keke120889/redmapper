import fitsio
import esutil as eu
import numpy as np
import itertools


from utilities import gaussFunction



class Centering(object):
    """
    Class for cluster centering
    
    
    MISSING: -- some are missing and some are replacements 
        I made, just need confirmation
    
    lambda = self.cluster.richness
    rlambda = self.cluster.rlambda
    zself.cluster = self.cluster.z
    gzreds = self.cluster.neighbors.zred 
    gzrede = self.cluster.neighbors.zred_e
    wtvals = self.cluster. neighbors.wt
    refmag_total = self.cluster.neighbors.refmag
    gchisq = self.cluster.neighbors.chisq
    zcluster = self.cluster.z
    wcen_Delta0: 0.0 in confstr - ok?
    wcen_Delta1: 0.0 in confstr - ok?
    zlambdastr
        zlambdastr.zred_corr
        zlambdastr.z
    pvals = self.cluster.zlambda_pz
    
    set all confstr.* variables in testconfig.yaml
    
    MISSING METHODS:
    gcirc
    close_match_radec
    
    """
    
    def __init__(self, cluster):
        
        self.cluster = cluster 
        
        self.mpcscale = np.radians(1.) * cluster.cosmo.Dl(0, cluster.z) / 
            (1 + cluster.z)**2
            
        self.r = cluster.neighbors.dist * self.mpcscale
        
        pass
        
    def wcen_zred(self):
        """
        from redmapper_centering_wcen_zred.pro
        
        consider galaxies for the center that...
        - are within rlambda
        - have p>0 OR abs(zred-zself.cluster) < 5*zred_err?
        
        #gcirc,1,rac/15d,decc,ras/15d,decs,dis
        # - needed!
        
        #close_match_radec,ras[use],decs[use],ras[u],decs[u],i1,i2,maxrad,nu+1,/silent
        # - needed
        
        """
        #gcirc,1,rac/15d,decc,ras/15d,decs,dis
        # - needed!
        
        #cut down to those considered as candidate centers
        use, = np.where((self.r < self.cluster.rlambda) & 
            ((self.cluster. neighbors.wt > 0.0) | 
            (np.absolute(self.cluster.z-self.cluster.neighbors.zred) 
            < 5.0*self.cluster.neighbors.zred_e)) &
            (self.cluster. neighbors.wt >= self.cluster.confstr.percolation_pbcg_cut) &
            (self.cluster.neighbors.chisq < self.cluster.confstr.wcen_zred_chisq_max))
        
        mstar = self.cluster.zredstr.mstar(self.cluster.z)
        mbar = mstar + (self.cluster.confstr.wcen_Delta0 + self.cluster.confstr.wcen_Delta1 * 
            np.log(self.cluster.richness/self.cluster.confstr.wcen_pivot))
        phi_cen = gaussFunction(self.cluster.neighbors.refmag[use], 
            np.array([(1./(np.sqrt(2.*np.pi)*self.cluster.confstr.wcen_sigma_m)), 
            mbar, self.cluster.confstr.wcen_sigma_m]))
        
        if zlambdastr is not None:
            #we have a zlambda calibration -- uncorrected, I hope.

            if self.cluster.confstr.zlambda_correct_internal:
                #this is the corrected zlambda
                zrmod = np.interp(zlambdastr.zred_corr, zlambdastr.z, self.cluster.z)
            else:
                #uncorrected zlambda
                zrmod = np.interp(zlambdastr.zred_uncorr, zlambdastr.z, self.cluster.z)
            endelse

            gz = gaussFunction(self.cluster.neighbors.zred[use], 
                np.array([(1./(np.sqrt(2.*np.pi)*self.cluster.neighbors.zred[use])), 
                zrmod, self.cluster.neighbors.zred_e[use]]))
        else:
            gz = gaussFunction(self.cluster.z, 
                np.array([(1./(np.sqrt(2.*np.pi)*self.cluster.neighbors.zred[use])), 
                zrmod, self.cluster.neighbors.zred_e[use]]))
        
        #and the w filter:  need w for each galaxy...
        #
        #okay, we need w for each galaxy that is considered as a candidate center.
        #but to calculate w, we need to know all the galaxies that are around it, but
        #within rlambda *of that galaxy*.  This is tricky.
        
        u, = np.where(self.cluster.zlambda_pz > 0.0) # these are the companions

        # whoah -- match radius isn't 1.0 degree, it's 1.1*r_lambda!
        maxrad = 1.1 * rlambda / (mpcscale *3600.0)


        #close_match_radec,ras[use],decs[use],ras[u],decs[u],i1,i2,maxrad,nu+1,/silent
        # - needed!

        """
        etc.
        """
        pass
    
    def bcg(self):
        """
        from redmapper_centering_bcg.pro
        #gcirc,1,rac/15d,decc,ras/15d,decs,dis
        # - needed!
        
        records top 5 - all unknown set to None
        """
        pcut = 0.8

        #gcirc,1,rac/15d,decc,ras/15d,decs,dis
        # - needed!
        
        if self.cluster.neighbors.zred.size > 0:
            use, = np.where((self.r < self.cluster.rlambda) & ((self.cluster.neighbors.wt > pcut) | 
                (np.absolute(self.cluster.neighbors.zred - self.cluster.z) < 2.0 * self.cluster.neighbors.zred_e)))
        else:
            use, = np.where((self.r < self.cluster.rlambda) and (self.cluster.neighbors.wt > pcut))

        if use.size > 0:
            mind = np.argmin(self.cluster.neighbors.refmag[use])
            maxind = use[mind]

            self.ra        = ras[maxind][:self.cluster.confstr.percolation_maxcen]
            self.dec       = decs[maxind][:self.cluster.confstr.percolation_maxcen]
            self.maxind    = maxind
            self.p_cen     = np.full(self.cluster.confstr.percolation_maxcen, 1.0)
            self.q_cen     = np.full(self.cluster.confstr.percolation_maxcen, 1.0)
            self.p_sat     = np.full(self.cluster.confstr.percolation_maxcen, 0.0)
            self.lnlamlike = None
            self.lnbcglike = None
            self.p_fg      = None
            self.q_miss    = None
            self.p_c       = None