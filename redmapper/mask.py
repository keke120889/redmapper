import esutil, fitsio
import healpy as hp
import numpy as np
from catalog import Catalog,Entry
from utilities import TOTAL_SQDEG, SEC_PER_DEG, astro_to_sphere, calc_theta_i
from numpy import random
from scipy.special import erf

class Mask(object):
    """
    A super-class for (pixelized) footprint masks

    This should not be instantiated directly (yet).

    parameters
    ----------
    confstr: Config object
       configuration
    """

    # note: not sure how to organize this.
    #   We need a routine that looks at the mask_mode and instantiates
    #   the correct type.  How is this typically done?

    def __init__(self, confstr):
        try:
            self.read_maskgals(confstr.maskgalfile)
        except:
            # this could throw a ValueError or AttributeError
            self.gen_maskgals()

    def calc_radmask(self, *args, **kwargs): pass
    def read_maskgals(self, maskgalfile):
        self.maskgals = Catalog.from_fits_file(maskgalfile)
    def gen_maskgals(self):
        # this needs to be written to generate maskgals if not from file
        # Tom-where would we generate them from?
        pass


class HPMask(Mask):
    """
    A class to use a healpix mask (mask_mode == 3)

    parameters
    ----------
    confstr: Config object
        Configuration object with maskfile

    """

    def __init__(self, confstr):
        # record for posterity
        self.maskfile = confstr.maskfile
        maskinfo, hdr = fitsio.read(confstr.maskfile, ext=1, header=True)
        # maskinfo converted to a catalog (array of Entrys)
        maskinfo = Catalog(maskinfo)
        nlim, nside, nest = maskinfo.hpix.size, hdr['NSIDE'], hdr['NEST']
        hpix_ring = maskinfo.hpix if nest != 1 else hp.nest2ring(nside, maskinfo.hpix)
        muse = np.arange(nlim)

        # if we have a sub-region of the sky, cut down the mask to save memory
        if confstr.hpix > 0:
            border = confstr.border + hp.nside2resol(nside)
            theta, phi = hp.pix2ang(confstr.nside, confstr.hpix)
            radius = np.sqrt(2) * (hp.nside2resol(confstr.nside)/2. + border)
            pixint = hp.query_disc(nside, hp.ang2vec(theta, phi), 
                                        np.radians(radius), inclusive=False)
            muse, = esutil.numpy_util.match(hpix_ring, pixint)

        offset, ntot = np.min(hpix_ring)-1, np.max(hpix_ring)-np.min(hpix_ring)+3
        self.nside = nside 
        self.offset = offset
        self.npix = ntot

        #ntot = np.max(hpix_ring) - np.min(hpix_ring) + 3
        self.fracgood = np.zeros(ntot,dtype='f4')

        # check if we have a fracgood in the input maskinfo
        try:
            self.fracgood_float = 1
            self.fracgood[hpix_ring-offset] = maskinfo[muse].fracgood
        except AttributeError:
            self.fracgood_float = 0
            self.fracgood[hpix_ring-offset] = 1
        super(HPMask, self).__init__(confstr)

    def compute_radmask(self, ra, dec):
        """
        Determine if a given set of ra/dec points are in or out of mask

        parameters
        ----------
        ra: array of doubles
        dec: array of doubles

        returns
        -------
        radmask: array of booleans

        """
        _ra  = np.atleast_1d(ra)
        _dec = np.atleast_1d(dec)

        if (_ra.size != _dec.size):
            raise ValueError("ra, dec must be same length")

        theta, phi = astro_to_sphere(_ra, _dec)
        ipring = hp.ang2pix(self.nside, theta, phi)
        ipring_offset = np.clip(ipring - self.offset, 0, self.npix-1)
        ref = 0 if self.fracgood_float == 0 else np.random.rand(_ra.size)
        radmask = np.zeros(_ra.size, dtype=np.bool_)
        radmask[np.where(self.fracgood[ipring_offset] > ref)] = True
        return radmask

    def set_radmask(self, cluster, mpcscale):
        """
        Assign mask (0/1) values to maskgals for a given cluster

        parameters
        ----------
        cluster: Cluster object
        mpcscale: float
            scaling to go from mpc to degrees (check units) at cluster redshift

        results
        -------
        sets maskgals['MASKED']

        """

        # note this probably can be in the superclass, no?

        ras = cluster.ra + self.maskgals.x/(mpcscale*SEC_PER_DEG)/np.cos(np.radians(clusters.dec))
        decs = cluster.dec + self.maskgals.y/(mpcscale*SEC_PER_DEG)
        self.maskgals['MASKED'] = self.compute_radmask(ras,decs)
        
    def calc_maskcorr(self, mstar, maxmag, limmag, confstr):
        """
        Obtain mask correction c parameters.
        
        parameters
        ----------
        maskgals :
        mstar    :
        maxmag   :
        limmag   : Limiting Magnitude
        confstr  : Configuration object
                    containing configuration info

        returns
        -------
        cpars:
        
        """
        
        mag_in = self.maskgals.m + mstar
        self.maskgals.refmag = mag_in
        
        if limmag > 0.0: #should this be self.maskgals.limmag[0] (IDL) or zredstr.limmag or confstr.limmag_ref??
        
            #ignore reuse_errormodel
            
            mag, mag_err = self.apply_errormodels(self.maskgals.exptime, 
                self.maskgals.limmag, mag_in, confstr, confstr.b, 
                zp = self.maskgals.zp[0], nsig=self.maskgals.nsig[0])
                #changes mag, mag_err in IDL - make it return them here?!
            
            self.maskgals.refmag_obs = mag
            self.maskgals.refmag_obs_err = mag_err
        else:
            mag = mag_in
            mag_err = 0*mag_in
            raise ValueError('Survey limiting magnitude <= 0!')
            #Raise error here ase this would lead to divide by zero if called.
            
        if (self.maskgals.w[0] < 0) or (self.maskgals.w[0] == 0 and 
            np.amax(self.maskgals.m50) == 0):
            tmode = 0
            theta_i = calc_theta_i(mag, mag_err, maxmag, limmag)
        elif (self.maskgals.w[0] == 0):
            tmode = 1
            theta_i = calc_theta_i(mag, mag_err, maxmag, self.maskgals.m50)
        else:
            tmode = 2
            raise Exception('Unsupported mode!')
        
        p_det = theta_i*self.maskgals.mark
        
        c = 1 - np.dot(p_det, self.maskgals.theta_r) / self.maskgals.nin[0]
        
        cpars = (np.polyfit(self.maskgals.radbins[0], c, 3)).flatten()
        
        return cpars
        
    def apply_errormodels(self, exptime, limmag, mag_in, confstr, b, zp=22.5, 
        nsig=10.0, err_ratio=1.0, fluxmode=False, nonoise=False, inlup=False, 
        errtflux='errtflux'):
        """
        
        parameters
        ----------
        exptime   :
        limmag    : Limiting Magnitude
        mag_in    :
        mag       :
        mag_err   :
        confstr   : Configuration object
            containing configuration info
        nonoise   :
        zp:       :
        nsig:     :
        fluxmode  :
        lnscat    :
        b         :
        inlup     :
        errtflux  :
        err_ratio :

        returns
        -------
        mag       :
        mag_err   :
        
        """           
        
        #ignore extinction bits: ext_factor = 1.0
        
        f1lim = 10.**((limmag - zp)/(-2.5))
        fsky1 = (((f1lim**2.) * exptime)/(nsig**2.) - f1lim) > 0.001
        
        if inlup:   #ignore inlup ?
            bnmgy = b*1e9       #1d9 - number of decimal points or 1e9????
            
            tflux = exptime*2.0*bnmgy*np.sinh(-np.log(b)-0.4*np.log(10.0)*mag_in)
         #---> needed??
        else:
            tflux = exptime*10.**((mag_in - zp)/(-2.5))
        
        noise = err_ratio*np.sqrt(fsky1*exptime + tflux)
        
        if nonoise:
            flux = tflux
        else:
            flux = tflux + noise*random.standard_normal(mag_in.size)
        
        if fluxmode:
            mag = flux/exptime
            mag_err = noise/exptime
        else:
            #set error for now - delete completely after fixed
            error = True
            if b.size > 0 and not error:
                bnmgy = b[0]*1e9        #1d9 - number of decimal points or 1e9????    
                #TAKE b[0] unntil problem fixed
                
                flux_new = flux/exptime
                noise_new = noise/exptime
                
                mag = 2.5*np.log10(1.0/b[0]) - np.arcsinh(0.5*flux_new/bnmgy)/(0.4*np.log(10.0))
                #TAKE b[0] unntil problem fixed
                mag_err = 2.5*noise_new/(2.0*bnmgy*np.log(10.0)
                    *np.sqrt(1.0+(0.5*flux_new/bnmgy)**2.0))
                #PROBLEMS WITH LENGTHS OF ARRAYS HERE: b.size = 5; flux_new.size=6000
            else:
                mag = zp-2.5*np.log10(flux/exptime)
                mag_err = (2.5/np.log(10.0))*(noise/flux)
                
                bad, = np.where(np.isfinite(mag) == False)
                mag[bad] = 99.0
                mag_err[bad] = 99.0
        return mag, mag_err
        
    def calc_maskcorr_lambdaerr(self, cluster, mstar, zredstr ,maxmag ,dof, 
        limmag, lam, rlam ,z ,bkg, wt, cval, r0, beta, gamma, cosmo):
        """
        
        parameters
        ----------
        mstar    :
        zredstr  :
        maxmag   :
        dof      :
        limmag   : Limiting Magnitude
        lam      :
        rlam     :
        z        :
        bkg      : Background object
                   background lookup table
        wt       :
        cval     :
        r0       :
        beta     :
        gamma    :
        cosmo    :
        refmag   :

        returns
        -------
        lambda_err:
        
        """
        
        use, = np.where(self.maskgals.r < rlam)
        
        mark    = self.maskgals.mark[use]
        refmag  = mstar + self.maskgals.m[use]
        cwt     = self.maskgals.cwt[use]
        nfw     = self.maskgals.nfw[use]
        lumwt   = self.maskgals.lumwt[use]
        chisq   = self.maskgals.chisq[use]
        r       = self.maskgals.r[use]
        
        logrc   = np.log(rlam)
        norm    = np.exp(1.65169 - 0.547850*logrc + 0.138202*logrc**2. - 
            0.0719021*logrc**3. - 0.0158241*logrc**4.-0.000854985*logrc**5.)
        nfw     = norm*nfw
        
        ucounts = cwt*nfw*lumwt

        faint, = np.where(refmag >= limmag)
        refmag_for_bcounts = np.copy(refmag)
        refmag_for_bcounts[faint] = limmag-0.01
        
        bcounts = cluster._calc_bkg_density(bkg, r, chisq , refmag_for_bcounts, cosmo)
        
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
        
    #UNNECESSARY COPY OF calclambda_chisq_bcounts - exists already in cluster.py
    
    #def calc_bcounts(self, z, r, chisq, refmag, bkg, cosmo, allow0='allow0'):
    #    """
    #    
    #    parameters
    #    ----------         :
    #    z                  :
    #    r                  :
    #    chisq              :
    #    refmag_for_bcounts :
    #    bkg                : Background object
    #                         background lookup table
    #    cosmo              :
    #    neigbours.refmag   :
    #    allow0             :
    #
    #    returns
    #    -------
    #    bcounts:
    #    
    #    """
    #    H0 = cosmo._H0
    #    nchisqbins  = bkg.chisqbins.size
    #    chisqindex  = np.around((chisq-bkg.chisqbins[0])*nchisqbins/
    #        (bkg.chisqbins[nchisqbins-1]+bkg.chisqbinsize-bkg.chisqbins[0])) 
    #    nrefmagbins = bkg.refmagbins.size
    #    refmagindex = np.around((refmag-bkg.refmagbins[0])*nrefmagbins/
    #        (bkg.refmagbins[nrefmagbins-1]+bkg.refmagbinsize-bkg.refmagbins[0]))
    #    
    #    #check for overruns
    #    badchisq, = np.where((chisqindex < 0) | (chisqindex >= nchisqbins))
    #    if (badchisq.size > 0): # $ important?
    #      chisqindex[badchisq] = 0
    #    badrefmag, = np.where((refmagindex < 0) | (refmagindex >= nrefmagbins))
    #    if (badrefmag.size > 0): # $ important?
    #      refmagindex[badrefmag] = 0
    #    
    #    ind = np.clip(np.around((z-bkg.zbins[0])/(bkg.zbins[1]-bkg.zbins[0])), 0, (bkg.zbins.size-1))
    #
    #    sigma_g = bkg.sigma_g[refmagindex, chisqindex, np.full_like(chisqindex, ind)]
    #    #no matter what, these should be infinities
    #    if (badchisq.size >  0):
    #        sigma_g[badchisq]= np.inf
    #    if (badrefmag.size > 0):
    #        sigma_g[badrefmag] = np.inf
    #        
    #    mpc_scale = np.radians(1.) * cosmo.Dl(0, z) / (1 + z)**2
    #    
    #    if not allow0:
    #        badcombination = np.where((sigma_g == 0.0) & (chisq > 5.0))
    #        if (badcombination.size > 0):
    #            sigma_g[badcombination] = np.inf
    #    
    #    bcounts = 2. * np.pi * r * (sigma_g / mpc_scale**2. ) # / c**2.) #WHAT IS C?
    #    print bcounts
    #    #return bcounts