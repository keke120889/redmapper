import fitsio
import healpy as hp
import numpy as np
from utilities import astro_to_sphere


class DepthMap(object):
    """
    A class to use a healpix depth map

    parameters
    ----------
    confstr: Config object
        Configuration object with depthfile

    """
    def __init__(self, confstr):
        # record for posterity
        self.depthfile = confstr.depthfile

        depthinfo, hdr = fitsio.read(filename, ext=1, header=True)
        # convert into catalog for convenience...
        depthinfo = Catalog(depthinfo)

        nlim, nside, nest = depthinfo.hpix.size, hdr['NSIDE'], hdr['NEST']
        if nest != 1:
            hpix_ring = depthinfo.hpix
        else:
            hpix_ring = hp.nest2ring(nside, depthinfo.hpix)

        muse = np.arange(nlim)

        # if we have a sub-region of the sky, cut down mask to save memory
        if confstr.hpix > 0:
            border = confstr.border + hp.nside2resol(nside)
            theta, phi = hp.pix2ang(confstr.nside, confstr.hpix)
            radius = np.sqrt(2) * (hp.nside2resol(confstr.nside)/2. + border)
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
        UNTESTED
        
        parameters
        ----------
        maskgals:
        ra:
        dec:
        mpc_scale:
        
        """
        #assume depthstr = depthinfo
        unseen = -1.63750e30
        
        # compute ra and dec based on maskgals
        ras = ra + (maskgals.x/(mpc_scale*3600.))/np.cos(dec*np.pi/180.)
        decs = dec + maskgals.y/(mpc_scale*3600.)
        theta = (90.0 - decs)*np.pi/180.
        phi = ras*np.pi/180.
        
        maskgals.w = depthstr.w                 #does this work?
        maskgals.eff = depthstr.eff
        maskgals.limmag = unseen
        maskgals[0].zp = depthstr.zp
        maskgals[0].nsig = depthstr.nsig
        
        theta, phi = astro_to_sphere(ras, decs)
        ipring = hp.ang2pix(maskgals.nside, theta, phi)
        ipring_offset = np.clip(ipring - maskgals.offset, 0, maskgals.npix-1)
        
        maskgals.limmag = depthstr.limmag[ipring_offset]
        maskgals.exptime = depthstr.exptime[ipring_offset]
        maskgals.m50 = depthstr.m50[ipring_offset]
        maskgals.ebv = depthstr.ebv[ipring_offset]
        maskgals.extinction = maskgals.ebv*a_lambda
        maskgals.limmag_dered = maskgals.limmag     #ignore extinction
        
        bd = where(maskgals.limmag < 0.0)
        ok = np.delete(np.copy(maskgals.limmag), bd)
        nok = ok.size
        
        if (bd.size > 0):
            if (nok ge 3) then begin
                # fill them in
                maskgals.limmag[bd] = median(maskgals.limmag[ok])
                maskgals.exptime[bd] = median(maskgals.exptime[ok])
                maskgals.m50[bd] = median(maskgals.m50[ok])
            elif(nok > 0):
                # fill with mean
                maskgals[bd].limmag = mean(maskgals[ok].limmag)
                maskgals[bd].exptime = mean(maskgals[ok].exptime)
                maskgals[bd].m50 = mean(maskgals[ok].m50)
            else:
                # very bad
                ok = where(depthstr.limmag > 0.0)
                maskgals.limmag = depthstr.limmag_default
                maskgals.exptime = depthstr.exptime_default
                maskgals.m50 = depthstr.m50_default
        
        # etc...
        # was thinking could call get_depth...
    

        # dtype = [('RA','f8'),
        #          ('DEC','f8'),
        #          ('TEST','i4')]

        # self.arr = np.zeros(100,dtype=dtype)

        # self.arr['RA'][:] = np.arange(100)
        # self.arr['RA'][:] = self.arr['RA'] + 1.0
