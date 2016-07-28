# miscellaneous utilities and functions

import numpy as np
from scipy.linalg import solve_banded
from pkg_resources import resource_filename
import scipy.interpolate as interpolate
import fitsio

###################################
## Useful constants ##
###################################
TOTAL_SQDEG = 4 * 180**2 / np.pi


######################################
## mstar LUT code
######################################

class MStar(object):
    def __init__(self, survey, band):
        self.survey = survey.strip()
        self.band = band.strip()

        self.mstar_file = resource_filename(__name__,'data/mstar/mstar_%s_%s.fit' % (self.survey, self.band))

        try:
            self._mstar_arr = fitsio.read(self.mstar_file,ext=1)
        except:
            raise IOError("Could not find mstar file mstar_%s_%s.fit" % (self.survey, self.band))

        #self._spl = CubicSpline(self._mstar_arr['Z'],self._mstar_arr['MSTAR'])
        self._f = interpolate.interp1d(self._mstar_arr['Z'],self._mstar_arr['MSTAR'],kind='cubic')
    def __call__(self, z):
        # may want to check the type ... if it's a scalar, return scalar?  TBD
        
        return self._f(z)




#############################################################
## cubic spline interpolation, based on Eddie Schlafly's code, from NumRec
##   http://faun.rc.fas.harvard.edu/eschlafly/apored/cubicspline.py
#############################################################

class CubicSpline(object):
    def __init__(self, x, y, yp=None):
        npts = len(x)
        mat = np.zeros((3, npts))
        # enforce continuity of 1st derivatives
        mat[1,1:-1] = (x[2:  ]-x[0:-2])/3.
        mat[2,0:-2] = (x[1:-1]-x[0:-2])/6.
        mat[0,2:  ] = (x[2:  ]-x[1:-1])/6.
        bb = np.zeros(npts)
        bb[1:-1] = ((y[2:  ]-y[1:-1])/(x[2:  ]-x[1:-1]) -
                    (y[1:-1]-y[0:-2])/(x[1:-1]-x[0:-2]))
        if yp is None: # natural cubic spline
            mat[1,0] = 1.
            mat[1,-1] = 1.
            bb[0] = 0.
            bb[-1] = 0.
        elif yp == '3d=0':
            mat[1, 0] = -1./(x[1]-x[0])
            mat[0, 1] =  1./(x[1]-x[0])
            mat[1,-1] =  1./(x[-2]-x[-1])
            mat[2,-2] = -1./(x[-2]-x[-1])
            bb[ 0] = 0.
            bb[-1] = 0.
        else:
            mat[1, 0] = -1./3.*(x[1]-x[0])
            mat[0, 1] = -1./6.*(x[1]-x[0])
            mat[2,-2] =  1./6.*(x[-1]-x[-2])
            mat[1,-1] =  1./3.*(x[-1]-x[-2])
            bb[ 0] = yp[0]-1.*(y[ 1]-y[ 0])/(x[ 1]-x[ 0])
            bb[-1] = yp[1]-1.*(y[-1]-y[-2])/(x[-1]-x[-2])
        y2 = solve_banded((1,1), mat, bb)
        self.x, self.y, self.y2 = (x, y, y2)

    def splint(self,x):
        npts = len(self.x)
        lo = np.searchsorted(self.x, x)-1
        lo = np.clip(lo, 0, npts-2)
        hi = lo + 1
        dx = self.x[hi] - self.x[lo]
        a = (self.x[hi] - x)/dx
        b = (x-self.x[lo])/dx
        y = (a*self.y[lo]+b*self.y[hi]+
             ((a**3-a)*self.y2[lo]+(b**3-b)*self.y2[hi])*dx**2./6.)
        return y
        
    def __call__(self, x):
        return self.splint(x)
