from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import esutil
import fitsio
import healpy as hp
import numpy as np
import os
from scipy.special import erf
import scipy.integrate

from .catalog import Catalog,Entry
from .utilities import TOTAL_SQDEG, SEC_PER_DEG, astro_to_sphere, calc_theta_i, apply_errormodels
from .utilities import make_lockfile, sample_from_pdf, chisq_pdf, schechter_pdf, nfw_pdf

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
        self.config = config

        # This will generate a maskgalfile if necessary
        self.read_maskgals(config.maskgalfile)

    def calc_radmask(self, *args, **kwargs): pass
    def read_maskgals(self, maskgalfile):

        make_maskgals = False
        lockfile = maskgalfile + '.lock'
        locktest = make_lockfile(lockfile, block=True, maxtry=300, waittime=2)
        if not locktest:
            raise RuntimeError("Could not get a lock to read/write maskgals!")

        if not os.path.isfile(maskgalfile):
            # We don't have the maskgals...generate them
            self._gen_maskgals(maskgalfile)

        # Read the maskgals
        self.maskgals = Catalog.from_fits_file(maskgalfile)

        # Clear lockfile
        os.remove(lockfile)

    def _gen_maskgals(self, maskgalfile):
        """
        """

        minrad = np.clip(np.floor(10.*self.config.percolation_r0 * (3./100.)**self.config.percolation_beta) / 10., None, 0.5)
        maxrad = np.ceil(10.*self.config.percolation_r0 * (300./100.)**self.config.percolation_beta) / 10.

        nradbins = np.ceil((maxrad - minrad) / self.config.maskgal_rad_stepsize).astype(np.int32) + 1
        radbins = np.arange(nradbins, dtype=np.float32) * self.config.maskgal_rad_stepsize + minrad

        nmag = self.config.nmag
        ncol = nmag - 1

        ngals = self.config.maskgal_ngals  # * nsamples

        maskgals = Catalog.zeros(ngals, dtype=[('r', 'f4'),
                                               ('phi', 'f4'),
                                               ('x', 'f4'),
                                               ('y', 'f4'),
                                               ('m', 'f4'),
                                               ('refmag', 'f4'),
                                               ('refmag_obs', 'f4'),
                                               ('refmag_obs_err', 'f4'),
                                               ('chisq', 'f4'),
                                               ('cwt', 'f4'),
                                               ('nfw', 'f4'),
                                               ('dzred', 'f4'),
                                               ('zwt', 'f4'),
                                               ('lumwt', 'f4'),
                                               ('limmag', 'f4'),
                                               ('limmag_dered', 'f4'),
                                               ('exptime', 'f4'),
                                               ('m50', 'f4'),
                                               ('eff', 'f4'),
                                               ('w', 'f4'),
                                               ('theta_r', 'f4', nradbins),
                                               ('mark', np.bool),
                                               ('radbins', 'f4', nradbins),
                                               ('nin', 'f4', nradbins),
                                               ('nin_orig', 'f4', nradbins),
                                               ('zp', 'f4'),
                                               ('ebv', 'f4'),
                                               ('extinction', 'f4'),
                                               ('nsig', 'f4')])

        maskgals['radbins'] = np.tile(radbins, maskgals.size).reshape(maskgals.size, nradbins)

        # Generate chisq
        maskgals.chisq = sample_from_pdf(chisq_pdf, [0.0, self.config.chisq_max],
                                         self.config.chisq_max / 10000.,
                                         maskgals.size, k=ncol)
        # Generate mstar
        maskgals.m = sample_from_pdf(schechter_pdf,
                                     [-2.5*np.log10(10.0),
                                       -2.5*np.log10(self.config.lval_reference) + self.config.maskgal_dmag_extra],
                                     0.002, maskgals.size,
                                     alpha=self.config.calib_lumfunc_alpha, mstar=0.0)
        # Generate nfw(r)
        maskgals.r = sample_from_pdf(nfw_pdf,
                                     [0.001, maxrad],
                                     0.001, maskgals.size, radfactor=True)

        # Generate phi
        maskgals.phi = 2. * np.pi * np.random.random(size=maskgals.size)

        # Precompute x/y
        maskgals.x = maskgals.r * np.cos(maskgals.phi)
        maskgals.y = maskgals.r * np.sin(maskgals.phi)

        # Compute weights to go with these values

        # Chisq weight
        maskgals.cwt = chisq_pdf(maskgals.chisq, ncol)

        # Nfw weight
        maskgals.nfw = nfw_pdf(maskgals.r, radfactor=True)

        # luminosity weight

        # We just choose a reference mstar for the normalization code
        mstar = 19.0
        normmag = mstar - 2.5 * np.log10(self.config.lval_reference)
        steps = np.arange(10.0, normmag, 0.01)
        f = schechter_pdf(steps, alpha=self.config.calib_lumfunc_alpha, mstar=mstar)
        n = scipy.integrate.simps(f, steps)
        maskgals.lumwt = schechter_pdf(maskgals.m + mstar, mstar=mstar, alpha=self.config.calib_lumfunc_alpha) / n

        # zred weight
        maskgals.dzred = np.random.normal(loc=0.0, scale=self.config.maskgal_zred_err, size=maskgals.size)
        maskgals.zwt = (1. / (np.sqrt(2.*np.pi) * self.config.maskgal_zred_err)) * np.exp(-(maskgals.dzred**2.) / (2.*self.config.maskgal_zred_err**2.))

        # Radial function
        for i, rad in enumerate(radbins):
            inside, = np.where((maskgals.r <= rad) & (maskgals.m < -2.5*np.log10(self.config.lval_reference)))
            maskgals.nin_orig[:, i] = inside.size

            if self.config.rsig <= 0.0:
                theta_r = np.ones(maskgals.size)
            else:
                theta_r = 0.5 + 0.5*erf((rad - maskgals.r) / (np.sqrt(2.)*self.config.rsig))
            maskgals.theta_r[:, i] = theta_r

            inside2, = np.where(maskgals.m < -2.5*np.log10(self.config.lval_reference))
            maskgals.nin[:, i] = np.sum(theta_r[inside2])

        # And save it

        hdr = fitsio.FITSHDR()
        hdr['version'] = 5
        hdr['r0'] = self.config.percolation_r0
        hdr['beta'] = self.config.percolation_beta
        hdr['stepsize'] = self.config.maskgal_rad_stepsize
        hdr['nmag'] = self.config.nmag
        hdr['ngals'] = self.config.maskgal_ngals
        hdr['chisqmax'] = self.config.chisq_max
        hdr['lvalref'] = self.config.lval_reference
        hdr['extra'] = self.config.maskgal_dmag_extra
        hdr['alpha'] = self.config.calib_lumfunc_alpha
        hdr['rsig'] = self.config.rsig
        hdr['zrederr'] = self.config.maskgal_zred_err

        maskgals.to_fits_file(maskgalfile, clobber=True, header=hdr)


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
        ras = cluster.ra + self.maskgals.x/(cluster.mpc_scale)/np.cos(np.radians(cluster.dec))
        decs = cluster.dec + self.maskgals.y/(cluster.mpc_scale)
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
        if config.d.hpix > 0:
            border = np.radians(config.border) + hp.nside2resol(nside_mask)
            theta, phi = hp.pix2ang(config.d.nside, config.d.hpix)
            radius = np.sqrt(2) * (hp.nside2resol(config.d.nside)/2. + border)
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

