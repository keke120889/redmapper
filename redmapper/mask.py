from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import esutil
import fitsio
import healpy as hp
import numpy as np
from scipy.special import erf

from .catalog import Catalog,Entry
from .utilities import TOTAL_SQDEG, SEC_PER_DEG, astro_to_sphere, calc_theta_i, apply_errormodels

class Mask(object):
    """
    A super-class for (pixelized) footprint masks

    This should not be instantiated directly (yet).

    parameters
    ----------
    config: Config object
       configuration
    """

    # note: not sure how to organize this.
    #   We need a routine that looks at the mask_mode and instantiates
    #   the correct type.  How is this typically done?

    def __init__(self, config):
        try:
            self.read_maskgals(config.maskgalfile)
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

    def set_radmask(self, cluster):
        """
        Assign mask (0/1) values to maskgals for a given cluster

        parameters
        ----------
        cluster: Cluster object

        results
        -------
        sets maskgals.mark

        """

        # note this probably can be in the superclass, no?
        ras = cluster.ra + self.maskgals.x/(cluster.mpc_scale*SEC_PER_DEG)/np.cos(np.radians(cluster.dec))
        decs = cluster.dec + self.maskgals.y/(cluster.mpc_scale*SEC_PER_DEG)
        self.maskgals.mark = self.compute_radmask(ras,decs)

    def calc_maskcorr(self, mstar, maxmag, limmag):
        """
        Obtain mask correction c parameters. From calclambda_chisq_calc_maskcorr.pro

        parameters
        ----------
        maskgals : Object holding mask galaxy parameters
        mstar    :
        maxmag   : Maximum magnitude
        limmag   : Limiting Magnitude
        config  : Configuration object
                    containing configuration info

        returns
        -------
        cpars

        """

        mag_in = self.maskgals.m + mstar
        self.maskgals.refmag = mag_in

        if self.maskgals.limmag[0] > 0.0:
            mag, mag_err = apply_errormodels(self.maskgals, mag_in)

            self.maskgals.refmag_obs = mag
            self.maskgals.refmag_obs_err = mag_err
        else:
            mag = mag_in
            mag_err = 0*mag_in
            raise ValueError('Survey limiting magnitude <= 0!')
            #Raise error here as this would lead to divide by zero if called.

        #extract object for testing
        #fitsio.write('test_data.fits', self.maskgals._ndarray)

        if (self.maskgals.w[0] < 0) or (self.maskgals.w[0] == 0 and 
            np.amax(self.maskgals.m50) == 0):
            theta_i = calc_theta_i(mag, mag_err, maxmag, limmag)
        elif (self.maskgals.w[0] == 0):
            theta_i = calc_theta_i(mag, mag_err, maxmag, self.maskgals.m50)
        else:
            raise Exception('Unsupported mode!')

        p_det = theta_i*self.maskgals.mark
        np.set_printoptions(threshold=np.nan)
        c = 1 - np.dot(p_det, self.maskgals.theta_r) / self.maskgals.nin[0]

        cpars = np.polyfit(self.maskgals.radbins[0], c, 3)

        return cpars

class HPMask(Mask):
    """
    A class to use a healpix mask (mask_mode == 3)

    parameters
    ----------
    config: Config object
        Configuration object with maskfile

    """

    def __init__(self, config):
        # record for posterity
        self.maskfile = config.maskfile
        maskinfo, hdr = fitsio.read(config.maskfile, ext=1, header=True, upper=True)
        # maskinfo converted to a catalog (array of Entrys)
        maskinfo = Catalog(maskinfo)
        nside_mask = hdr['NSIDE']
        nest = hdr['NEST']

        hpix_ring = maskinfo.hpix if nest != 1 else hp.nest2ring(nside_mask, maskinfo.hpix)

        # if we have a sub-region of the sky, cut down the mask to save memory
        if config.hpix > 0:
            border = np.radians(config.border) + hp.nside2resol(nside_mask)
            theta, phi = hp.pix2ang(config.nside, config.hpix)
            radius = np.sqrt(2) * (hp.nside2resol(config.nside)/2. + border)
            pixint = hp.query_disc(nside_mask, hp.ang2vec(theta, phi),
                                   radius, inclusive=False)
            suba, subb = esutil.numpy_util.match(pixint, hpix_ring)
            hpix_ring = hpix_ring[subb]
            muse = subb
        else:
            muse = np.arange(hpix_ring.size, dtype='i4')

        offset, ntot = np.min(hpix_ring)-1, np.max(hpix_ring)-np.min(hpix_ring)+3
        self.nside = nside_mask
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
        super(HPMask, self).__init__(config)

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


def get_mask(config):
    """
    Convenience function to look at a config file and load the appropriate type of mask.

    parameters
    ----------
    config: Config object
        Configuration object with maskfile
    """

    if config.mask_mode == 0:
        # This is no mask!
        # Return a bare object with maskgal functionality
        return Mask(config)
    elif config.mask_mode == 3:
        # This is a healpix mask
        #  (don't ask about 1 and 2)
        return HPMask(config)

