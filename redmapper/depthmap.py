from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import fitsio
import healpy as hp
import numpy as np
from healpy import pixelfunc
import esutil

from .utilities import astro_to_sphere
from .catalog import Catalog, Entry


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

        #self.npix = depthinfo.hpix.size
        nside_mask = hdr['NSIDE']
        nest = hdr['NEST']
        self.nsig = hdr['NSIG']
        self.zp = hdr['ZP']
        self.nband = hdr['NBAND']
        self.w = hdr['W']
        self.eff = hdr['EFF']

        self.config_area = config.area
        self.submask_hpix = config.hpix
        self.submask_nside = config.nside
        self.submask_border = config.border
        self.galfile_nside = config.galfile_nside

        if nest != 1:
            hpix_ring = depthinfo.hpix
        else:
            hpix_ring = hp.nest2ring(nside_mask, depthinfo.hpix)

        # if we have a sub-region of the sky, cut down mask to save memory
        if self.submask_hpix > 0:
            border = np.radians(self.submask_border) + hp.nside2resol(nside_mask)
            theta, phi = hp.pix2ang(self.submask_nside, self.submask_hpix)
            radius = np.sqrt(2) * (hp.nside2resol(self.submask_nside)/2. + border)
            pixint = hp.query_disc(nside_mask, hp.ang2vec(theta, phi),
                                   radius, inclusive=False)
            suba, subb = esutil.numpy_util.match(pixint, hpix_ring)
            hpix_ring = hpix_ring[subb]
            duse = subb
        else:
            duse = np.arange(hpix_ring.size, dtype='i4')

        self.nside = nside_mask
        self.offset = np.min(hpix_ring) - 1
        self.ntot = np.max(hpix_ring) - np.min(hpix_ring) + 3

        self.npix = hpix_ring.size

        self.fracgood = np.zeros(self.npix + 1, dtype='f4')
        try:
            self.fracgood_float = 1
            self.fracgood[0:self.npix] = depthinfo[duse].fracgood
        except AttributeError:
            self.fracgood_float = 0
            self.fracgood[0:self.npix] = 0

        self.exptime = np.zeros(self.npix + 1, dtype='f4')
        self.exptime[0:self.npix] = depthinfo[duse].exptime
        self.limmag = np.zeros(self.npix + 1, dtype='f4')
        self.limmag[0:self.npix] = depthinfo[duse].limmag
        self.m50 = np.zeros(self.npix + 1, dtype='f4')
        self.m50[0:self.npix] = depthinfo[duse].m50

        # And the overflow bins
        self.fracgood[self.npix] = hp.UNSEEN
        self.exptime[self.npix] = hp.UNSEEN
        self.limmag[self.npix] = hp.UNSEEN
        self.m50[self.npix] = hp.UNSEEN

        # The look-up table
        #  Set default to overflow bin
        self.hpix_to_index = np.zeros(self.ntot, dtype='i4') + self.npix
        self.hpix_to_index[hpix_ring - self.offset] = np.arange(self.npix)

    def get_depth_values(self, ras, decs):
        """
        """

        theta = (90.0 - decs) * np.pi / 180.
        phi = ras * np.pi / 180.

        ipring_offset = np.clip(hp.ang2pix(self.nside, theta, phi) - self.offset,
                                0, self.ntot - 1)

        return (self.limmag[self.hpix_to_index[ipring_offset]],
                self.exptime[self.hpix_to_index[ipring_offset]],
                self.m50[self.hpix_to_index[ipring_offset]])


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
        #theta = (90.0 - decs)*np.pi/180.
        #phi = ras*np.pi/180.

        maskgals.w[:] = self.w
        maskgals.eff = None
        maskgals.limmag[:] = unseen
        maskgals.zp[0] = self.zp
        maskgals.nsig[0] = self.nsig

        #theta, phi = astro_to_sphere(ras, decs)
        #ipring = hp.ang2pix(self.nside, theta, phi)
        #ipring_offset = np.clip(ipring - self.offset, 0, self.ntot-1)

        #maskgals.limmag = self.limmag[self.hpix_to_index[ipring_offset]]
        #maskgals.exptime = self.exptime[self.hpix_to_index[ipring_offset]]
        #maskgals.m50 = self.m50[self.hpix_to_index[ipring_offset]]

        maskgals.limmag, maskgals.exptime, maskgals.m50 = self.get_depth_values(ras, decs)

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
                raise RuntimeError("This shouldn't get here...")
                #ok, = np.where(self.limmag > 0.0)
                #maskgals.limmag = self.limmag_default
                #maskgals.exptime = depthstr.exptime_default
                #maskgals.m50 = depthstr.m50_default

    def calc_areas(self, mags):
        """
        """

        pixsize = hp.nside2pixarea(self.nside, degrees=True)

        if (self.w < 0.0):
            # This is just constant area
            areas = np.zeros(mags.size) + self.config_area
            return areas

        if self.submask_hpix > 0:
            # for the subregion, we need the area covered in the main pixel
            # I'm not sure what to do about border...but you shouldn't
            # be running this with a subregion with a border
            if self.submask_border > 0.0:
                raise ValueError("Cannot run calc_areas() with a subregion with a border")

            hpix = np.arange(self.ntot) + self.offset
            theta, phi = hp.pix2ang(self.nside, hpix)
            hpix_submask = hp.ang2pix(self.galfile_nside, theta, phi)

            use, = np.where(hpix_submask == self.submask_hpix)
        else:
            use = np.arange(self.ntot)

        areas = np.zeros(mags.size)

        gd, = np.where((self.m50[self.hpix_to_index[use]] >= 0.0))

        depths = self.m50[self.hpix_to_index[use[gd]]]
        st = np.argsort(depths)
        depths = depths[st]

        fracgoods = self.fracgood[self.hpix_to_index[use[gd[st]]]]

        inds = np.clip(np.searchsorted(depths, mags) - 1, 1, depths.size - 1)

        lo = (inds < 0)
        areas[lo] = np.sum(fracgoods) * pixsize
        carea = pixsize * np.cumsum(fracgoods)
        areas[~lo] = carea[carea.size - inds[~lo]]

        return areas

