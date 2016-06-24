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
        #self.nside = nside
        pass

    def get_depth(self,ra=None,dec=None,theta=None,phi=None,ipring=None):
        # require ra/dec or theta/phi and check

        # return depth info
        pass

    
    def calc_maskdepth(self,maskgals,ra,dec,mpcscale):
        # compute ra and dec based on maskgals
        ras=ra + (maskgals['X']/(mpcscale*3600.))/np.cos(dec*np.pi/180.)
        decs=dec + maskgals['Y']/(mpcscale*3600.)

        theta = (90.0 - decs)*np.pi/180.
        phi = ras*np.pi/180.

        # etc...
        # was thinking could call get_depth...
    

        
