import esutil
import healpy as hp
import numpy as np
from catalog import Entry
from utilities import TOTAL_SQDEG, SEC_PER_DEG, astro_to_sphere

_PIXRES = 8

class Mask(object):
    """Docstring."""
    def __init__(self): pass
    def calc_radmask(self, *args, **kwargs): pass

    def read_maskgals(self, maskgalfile):
        # if maskgalfile doesn't exist
        # self.gen_maskgals(self, maskgalfile)
        pass


class HPMask(Mask):
    """Docstring."""

    def __init__(self, confstr):
        maskinfo, hdr = fitsio.read(confstr.maskfile, ext=1, header=True)
        maskinfo = Entry(maskinfo)
        nlim, nside, nest = maskinfo.hpix.size, hdr['NSIDE'], hdr['NEST']
        hpix_ring = maskinfo.hpix if nest != 1 else hp.nest2ring(nside, maskinfo.hpix)
        muse = np.arange(nlim)
        
        if confstr.hpix > 0:
            border = confstr.border + hp.nside2resol(nside)
            theta, phi = hp.pix2ang(confstr.nside, confstr.hpix)
            radius = np.sqrt(2) * (hp.nside2resol(confstr.nside)/2. + border)
            pixint = hp.query_disc(nside, hp.ang2vec(theta, phi), 
                                        np.radians(radius), inclusive=False)
            muse, = esutil.numpy_util.match(hpix_ring, pixint)
        
        offset, ntot = min(hpix_ring)-1, max(hpix_ring)-min(hpix_ring)+3
        self.nside, self.offset, self.npix = nside, offset, ntot
        
        try:
            self.fracgood_float = 1
            self.fracgood[hpix_ring-offset] = instr[muse].fracgood
        except ValueError:
            self.fracgood_float = 0
            self.fracgood[hpix_ring-offset] = 1


    def compute_radmask(self, ra, dec):
        theta, phi = astro_to_sphere(ra, dec)
        ipring = hp.ang2pix(self.nside, theta, phi)
        ipring = np.clip(ipring - maskstr.offset, 0, maskstr.npix-1)
        ref = 0 if self.fracgood_float == 0 else np.random.rand(ras.size)     
        radmask = np.zeros(ra.size, dtype=np.bool_)
        radmask[np.where(self.fracgood[ipring] > ref)] = True
        return radmask
            
    def set_radmask(self, clusters, mpcscale):
        ras = clusters.ra + self.maskgals.x/(mpcscale*SEC_PER_DEG)/np.cos(np.radians(clusters.dec))
        decs = clusters.dec + self.maskgals.y/(mpcscale*SEC_PER_DEG)
        self.maskgals['MASKED'] = self.compute_radmask(ras,decs)

