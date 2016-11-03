import fitsio
import healpy as hp
import numpy as np


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
        # compute ra and dec based on maskgals
        # ras = ra + (maskgals['X']/(mpc_scale*3600.))/np.cos(dec*np.pi/180.)
        # decs = dec + maskgals['Y']/(mpc_scale*3600.)
        # theta = (90.0 - decs)*np.pi/180.
        # phi = ras*np.pi/180.

        # etc...
        # was thinking could call get_depth...
    

        # dtype = [('RA','f8'),
        #          ('DEC','f8'),
        #          ('TEST','i4')]

        # self.arr = np.zeros(100,dtype=dtype)

        # self.arr['RA'][:] = np.arange(100)
        # self.arr['RA'][:] = self.arr['RA'] + 1.0
        pass
