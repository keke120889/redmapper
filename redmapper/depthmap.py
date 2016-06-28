import fitsio
import healpy as hp
import numpy as np

class DepthMap():
    
    def __init__(self,filename):
        # call method to read
        self.filename = filename
        self.read_depthmap()

    def read_depthmap(self):
        # read in from filename
        # self.nside = nside
        pass

    def get_depth(self, ra=None, dec=None, theta=None, phi=None, ip_ring=None):
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
