# miscellaneous utilities and functions

import numpy as np
from scipy import special
from scipy.linalg import solve_banded
from pkg_resources import resource_filename
import scipy.interpolate as interpolate
import fitsio
from scipy.special import erf
from numpy import random

###################################
## Useful constants/conversions ##
###################################
TOTAL_SQDEG = 4 * 180**2 / np.pi
SEC_PER_DEG = 3600

def astro_to_sphere(ra, dec):
    return np.radians(90.0-dec), np.radians(ra)

#Equation 7 in Rykoff et al. 2014
def chisq_pdf(data, k):
    normalization = 1./(2**(k/2.) * special.gamma(k/2.))
    return normalization * data**((k/2.)-1) * np.exp(-data/2.)

def gaussFunction(x, *p):
   A, mu, sigma = p
   return A*np.exp(-(x-mu)**2./(2.*sigma**2))


######################################
## mstar LUT code
######################################
class MStar(object):
    def __init__(self, survey, band):
        self.survey = survey.strip()
        self.band = band.strip()

        try:
            self.mstar_file = resource_filename(__name__,'data/mstar/mstar_%s_%s.fit' % (self.survey, self.band))
        except:
            raise IOError("Could not find mstar resource mstar_%s_%s.fit" % (self.survey, self.band))
        try:
            self._mstar_arr = fitsio.read(self.mstar_file,ext=1)
        except:
            raise IOError("Could not find mstar file mstar_%s_%s.fit" % (self.survey, self.band))

        # Tom - why not use CubicSpline here? That's why it exists...
        self._f = CubicSpline(self._mstar_arr['Z'],self._mstar_arr['MSTAR'])
        #self._f = interpolate.interp1d(self._mstar_arr['Z'],self._mstar_arr['MSTAR'],kind='cubic')

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

def calc_theta_i(mag, mag_err, maxmag, limmag):
    """
    Calculate theta_i. This is reproduced from calclambda_chisq_theta_i.pro

    parameters
    ----------
    mag:
    mag_err:
    maxmag:
    limmag:

    returns
    -------
    theta_i:
    """

    theta_i = np.ones(mag.size)
    eff_lim = np.clip(maxmag, 0, limmag)
    dmag = eff_lim - mag
    calc, = np.where(dmag < 5.0 * mag_err)
    if calc.size > 0:
        theta_i[calc] = 0.5 + 0.5 * erf(dmag[calc] / (np.sqrt(2) * mag_err[calc]))
    hi, = np.where(mag > limmag)
    if hi.size > 0:
        theta_i[hi] = 0.0
    return theta_i

    #theta_i = np.ones((len(mag)))
    #eff_lim = np.clip(maxmag,0,limmag)
    #dmag = eff_lim - mag
    #calc = dmag < 5.0
    #N_calc = np.count_nonzero(calc==True)
    #if N_calc > 0: theta_i[calc] = 0.5 + 0.5*erf(dmag[calc]/(np.sqrt(2)*mag_err[calc]))
    #hi = mag > limmag
    #N_hi = np.count_nonzero(hi==True)
    #if N_hi > 0:
    #    theta_i[hi] = 0.0
    #return theta_i

def apply_errormodels(maskgals, mag_in, b = None, err_ratio=1.0, fluxmode=False, 
    nonoise=False, inlup=False):
    """
    Find magnitude and uncertainty.

    parameters
    ----------
    mag_in    :
    nonoise   : account for noise / no noise
    zp:       : Zero point magnitudes
    nsig:     :
    fluxmode  :
    lnscat    :
    b         : parameters for luptitude calculation
    inlup     :
    errtflux  :
    err_ratio : scaling factor

    returns
    -------
    mag
    mag_err

    """
    f1lim = 10.**((maskgals.limmag - maskgals.zp[0])/(-2.5))
    fsky1 = (((f1lim**2.) * maskgals.exptime)/(maskgals.nsig[0]**2.) - f1lim)
    fsky1 = np.clip(fsky1, 0.001, None)

    if inlup:
        bnmgy = b*1e9
        tflux = maskgals.exptime*2.0*bnmgy*np.sinh(-np.log(b)-0.4*np.log(10.0)*mag_in)
    else:
        tflux = maskgals.exptime*10.**((mag_in - maskgals.zp[0])/(-2.5))

    noise = err_ratio*np.sqrt(fsky1*maskgals.exptime + tflux)

    if nonoise:
        flux = tflux
    else:
        flux = tflux + noise*random.standard_normal(mag_in.size)

    if fluxmode:
        mag = flux/maskgals.exptime
        mag_err = noise/maskgals.exptime
    else:
        if b is not None:
            bnmgy = b*1e9

            flux_new = flux/maskgals.exptime
            noise_new = noise/maskgals.exptime

            mag = 2.5*np.log10(1.0/b) - np.arcsinh(0.5*flux_new/bnmgy)/(0.4*np.log(10.0))
            mag_err = 2.5*noise_new/(2.0*bnmgy*np.log(10.0)*np.sqrt(1.0+(0.5*flux_new/bnmgy)**2.0))
        else:
            mag = maskgals.zp[0]-2.5*np.log10(flux/maskgals.exptime)
            mag_err = (2.5/np.log(10.0))*(noise/flux)

            bad, = np.where(np.isfinite(mag) == False)
            mag[bad] = 99.0
            mag_err[bad] = 99.0

    return mag, mag_err

def interpol(v, x, xout):
    """
    """

    m = v.size
    nOut = m

    s = np.clip(np.searchsorted(x, xout) - 1, 0, m - 2)

    diff = v[s + 1] - v[s]

    return (xout - x[s]) * diff / (x[s + 1] - x[s]) + v[s]
