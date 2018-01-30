import fitsio
import healpy as hp
import numpy as np

from utilities import astro_to_sphere
from catalog import Catalog, Entry
from healpy import pixelfunc


class DepthMap(object):
    """
    A class to use a healpix depth map

    parameters
    ----------
    config: Config object
        Configuration object with depthfile

    """
    def __init__(self, config):
        # record for posterity
        self.depthfile = config.depthfile

        depthinfo, hdr = fitsio.read(self.depthfile, ext=1, header=True)
        # convert into catalog for convenience...
        depthinfo = Catalog(depthinfo)

        nlim        = depthinfo.hpix.size
        nside       = hdr['NSIDE']
        nest        = hdr['NEST']
        self.nsig   = hdr['NSIG']
        self.zp     = hdr['ZP']
        self.nband  = hdr['NBAND']
        self.w      = hdr['W']
        self.eff    = hdr['EFF']

        if nest != 1:
            hpix_ring = depthinfo.hpix
        else:
            hpix_ring = hp.nest2ring(nside, depthinfo.hpix)

        muse = np.arange(nlim)

        # if we have a sub-region of the sky, cut down mask to save memory
        if config.hpix > 0:
            border = config.border + hp.nside2resol(nside)
            theta, phi = hp.pix2ang(config.nside, config.hpix)
            radius = np.sqrt(2) * (hp.nside2resol(config.nside)/2. + border)
            pixint = hp.query_disc(nside, hp.ang2vec(theta, phi), 
                                        np.radians(radius), inclusive=False)
            muse, = esutil.numpy_util.match(hpix_ring, pixint)

        offset = np.min(hpix_ring)-1
        ntot = np.max(hpix_ring) - np.min(hpix_ring) + 3
        self.nside = nside
        self.offset = offset
        self.ntot = ntot

        self.fracgood = np.zeros(ntot,dtype='f4')

        # check if we have a fracgood in the input maskinfo
        try:
            self.fracgood_float = 1
            self.fracgood[hpix_ring-offset] = depthinfo[muse].fracgood
        except AttributeError:
            self.fracgood_float = 0
            self.fracgood[hpix_ring-offset] = 1

        self.exptime = np.zeros(ntot,dtype='f4')
        self.exptime[hpix_ring-offset] = depthinfo[muse].exptime
        self.limmag = np.zeros(ntot,dtype='f4')
        self.limmag[hpix_ring-offset] = depthinfo[muse].limmag
        self.m50 = np.zeros(ntot,dtype='f4')
        self.m50[hpix_ring-offset] = depthinfo[muse].m50


    def get_depth(self, ra=None, dec=None, theta=None, phi=None, ipring=None):
        # require ra/dec or theta/phi and check

        # return depth info
        pass


    def calc_maskdepth(self, maskgals, ra, dec, mpc_scale):
        """
        set masgals parameters: limmag, exptime, m50
        parameters
        ----------
        maskgals: Object holding mask galaxy parameters
        ra: Right ascention of cluster
        dec: Declination of cluster
        mpc_scale: scaling to go from mpc to degrees (check units) at cluster redshift

        """

        unseen = hp.pixelfunc.UNSEEN

        # compute ra and dec based on maskgals
        ras = ra + (maskgals.x/(mpc_scale*3600.))/np.cos(dec*np.pi/180.)
        decs = dec + maskgals.y/(mpc_scale*3600.)
        theta = (90.0 - decs)*np.pi/180.
        phi = ras*np.pi/180.

        maskgals.w = self.w
        maskgals.eff = None
        maskgals.limmag = unseen
        maskgals.zp[0] = self.zp
        maskgals.nsig[0] = self.nsig

        theta, phi = astro_to_sphere(ras, decs)
        ipring = hp.ang2pix(self.nside, theta, phi)
        ipring_offset = np.clip(ipring - self.offset, 0, self.ntot-1)

        maskgals.limmag     = self.limmag[ipring_offset]
        maskgals.exptime    = self.exptime[ipring_offset]
        maskgals.m50        = self.m50[ipring_offset]

        bd, = np.where(maskgals.limmag < 0.0)
        ok = np.delete(np.copy(maskgals.limmag), bd)
        nok = ok.size

        if (bd.size > 0):
            if (nok >= 3):
                # fill them in
                maskgals.limmag[bd] = median(maskgals.limmag[ok])
                maskgals.exptime[bd] = median(maskgals.exptime[ok])
                maskgals.m50[bd] = median(maskgals.m50[ok])
            elif (nok > 0):
                # fill with mean
                maskgals.limmag[bd] = mean(maskgals.limmag[ok])
                maskgals.exptime[bd] = mean(maskgals.exptime[ok])
                maskgals.m50[bd] = mean(maskgals.m50[ok])
            else:
                # very bad
                ok = where(depthstr.limmag > 0.0)
                maskgals.limmag = depthstr.limmag_default
                maskgals.exptime = depthstr.exptime_default
                maskgals.m50 = depthstr.m50_default
