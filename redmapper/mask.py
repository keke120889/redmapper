import esutil, fitsio
import healpy as hp
import numpy as np
from catalog import Catalog,Entry
from utilities import TOTAL_SQDEG, SEC_PER_DEG, astro_to_sphere


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

        theta, phi = astro_to_sphere(ra, dec)
        ipring = hp.ang2pix(self.nside, theta, phi)
        ipring_offset = np.clip(ipring - maskstr.offset, 0, maskstr.npix-1)
        ref = 0 if self.fracgood_float == 0 else np.random.rand(ras.size)
        radmask = np.zeros(ra.size, dtype=np.bool_)
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

