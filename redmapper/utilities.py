from __future__ import division, absolute_import, print_function
from past.builtins import xrange


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
            self._mstar_arr = fitsio.read(self.mstar_file, ext=1, upper=True)
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

#####################
## IDL interpol
#####################

def interpol(v, x, xout):
    """
    """

    m = v.size
    nOut = m

    s = np.clip(np.searchsorted(x, xout) - 1, 0, m - 2)

    diff = v[s + 1] - v[s]

    return (xout - x[s]) * diff / (x[s + 1] - x[s]) + v[s]

#####################
## IDL cic
#####################

def cic(value, posx=None, nx=None, posy=None, ny=None, posz=None, nz=None, average=False, isolated=True):
    """
    Port of idl astronomy utils cic.pro
    """

    # need type checks

    nrsamples = value.size
    dim = 0
    if (posx is not None and nx is not None):
        dim += 1
    if (posy is not None and ny is not None):
        dim += 1
    if (posz is not None and nz is not None):
        dim += 1

    if dim <= 2:
        nz = 1
        if dim == 1:
            ny = 1

    nxny = nx * ny

    # X-direction

    # coordinates of nearest grid point (ngp)
    ngx = posx.astype(np.int32) + 0.5

    # distance from sample to ngp
    dngx = ngx - posx

    # Index of ngp
    kx1 = ngx - 0.5
    # weight of ngp
    wx1 = 1.0 - np.abs(dngx)

    # other side
    left, = np.where(dngx < 0.0)
    # The following is only correct if x(ngp)>posx (ngp to the right).
    kx2 = kx1 - 1
    # Correct points where x(ngp)<posx (ngp to the left).
    if left.size > 0:
        kx2[left] += 2
    wx2 = np.abs(dngx)

    # free memory
    left = None

    bad, = np.where(kx2 == -1)
    if bad.size > 0:
        kx2[bad] = nx - 1
        if isolated:
            wx2[bad] = 0.0
    bad, = np.where(kx2 == nx)
    if bad.size > 0:
        kx2[bad] = 0
        if isolated:
            wx2[bad] = 0.0
    bad = None

    # Y-direction

    if dim >= 2:
        ngy = posy.astype(np.int32) + 0.5

        # distance from sample to ngp
        dngy = ngy - posy

        # Index of ngp
        ky1 = ngy - 0.5
        # weight of ngp
        wy1 = 1.0 - np.abs(dngy)

        # other side
        left, = np.where(dngy < 0.0)
        # The following is only correct if y(ngp)>posy (ngp to the right).
        ky2 = ky1 - 1
        # Correct points where x(ngp)<posx (ngp to the left).
        if left.size > 0:
            ky2[left] += 2
        wy2 = np.abs(dngy)

        # free memory
        left = None

        bad, = np.where(ky2 == -1)
        if bad.size > 0:
            ky2[bad] = ny - 1
            if isolated:
                wy2[bad] = 0.0
        bad, = np.where(ky2 == ny)
        if bad.size > 0:
            ky2[bad] = 0
            if isolated:
                wy2[bad] = 0.0
        bad = None
    else:
        ky1 = 0
        ky2 = 0
        wy1 = 1
        wy2 = 1

    # Z-direction

    if dim == 3:
        ngz = posz.astype(np.int32) + 0.5

        # distance from sample to ngp
        dngz = ngz - posz

        # Index of ngp
        kz1 = ngz - 0.5
        # weight of ngp
        wz1 = 1.0 - np.abs(dngz)

        # other side
        left, = np.where(dngz < 0.0)
        # The following is only correct if z(ngp)>posz (ngp to the right).
        kz2 = kz1 - 1
        # Correct points where z(ngp)<posz (ngp to the left).
        if left.size > 0:
            kz2[left] += 2
        wz2 = np.abs(dngz)

        # free memory
        left = None

        bad, = np.where(kz2 == -1)
        if bad.size > 0:
            kz2[bad] = nz - 1
            if isolated:
                wz2[bad] = 0.0
        bad, = np.where(kz2 == nz)
        if bad.size > 0:
            kz2[bad] = 0
            if isolated:
                wz2[bad] = 0.0
        bad = None
    else:
        kz1 = 0
        kz2 = 0
        wz1 = 1
        wz2 = 1

    # Interpolate samples to grid

    field = np.zeros(nx * ny * nz)

    if average:
        totcicweight = np.zeros_like(field)

    index = (kx1 + ky1*nx + kz1 * nxny).astype(np.int32)
    cicweight = wx1 * wy1 * wz1
    np.add.at(field, index, cicweight * value)
    if average:
        np.add.at(totcicweight, index, cicweight)

    index = (kx2 + ky1 * nx + kz1 * nxny).astype(np.int32)
    cicweight = wx2 * wy1 * wz1
    np.add.at(field, index, cicweight * value)
    if average:
        np.add.at(totcicweight, index, cicweight)

    if dim >= 2:
        index = (kx1 + ky2*nx + kz1*nxny).astype(np.int32)
        cicweight = wx1 * wy2 * wz1
        np.add.at(field, index, cicweight * value)
        if average:
            np.add.at(totcicweight, index, cicweight)
        index = (kx2 + ky2*nx + kz1*nxny).astype(np.int32)
        cicweight = wx2 * wy2 * wz1
        np.add.at(field, index, cicweight * value)
        if average:
            np.add.at(totcicweight, index, cicweight)
        if dim == 3:
            index = (kx1 + ky1 * nx + kz2 * nxny).astype(np.int32)
            cicweight = wx1 * wy1 * wz2
            np.add.at(field, index, cicweight * value)
            if average:
                np.add.at(totcicweight, index, cicweight)
            index = (kx2 + ky1 * nx + kz2 * nxny).astype(np.int32)
            cicweight = wx2 * wy1 * wz2
            np.add.at(field, index, cicweight * value)
            if average:
                np.add.at(totcicweight, index, cicweight)
            index = (kx1 + ky2 * nx + kz2 * nxny).astype(np.int32)
            cicweight = wx1 * wy2 * wz2
            np.add.at(field, index, cicweight * value)
            if average:
                np.add.at(totcicweight, index, cicweight)
            index = (kx2 + ky2 * nx + kz2 * nxny).astype(np.int32)
            cicweight = wx2 * wy2 * wz2
            np.add.at(field, index, cicweight * value)
            if average:
                np.add.at(totcicweight, index, cicweight)

    index = None

    if average:
        good, = np.where(totcicweight != 0)
        field[good] /= totcicweight[good]

    if dim == 1:
        return field
    elif dim == 2:
        return field.reshape((ny, nx))
    else:
        return field.reshape((nz, ny, nx))

#########################
# MakeNodes
#########################

def make_nodes(zrange, nodesize, maxnode=None):
    """
    """

    if maxnode is None or maxnode < 0.0:
        _maxnode = zrange[1]
    else:
        _maxnode = np.clip(maxnode, 0.0, zrange[1])

    # Start with a simple arange
    nodes = np.arange(zrange[0], _maxnode, nodesize)

    # Should we extend the last bin?
    if ((_maxnode - nodes.max()) > (nodesize / 2. + 0.01)):
        # one more node!
        nodes = np.append(nodes, _maxnode)
    elif ~np.allclose(np.max(nodes), _maxnode):
        nodes[-1] = _maxnode

    # and finally check if maxnode was lower
    if _maxnode < zrange[1]:
        nodes = np.append(nodes, zrange[1])

    return nodes

#######################
## Sample from a pdf
#######################

def sample_from_pdf(f, ran, step, nsamp, **kwargs):
    """
    """

    x = np.arange(ran[0], ran[1], step)
    pdf = f(x, **kwargs)
    pdf /= np.sum(pdf)
    cdf = np.cumsum(pdf)
    cdfi = (cdf * x.size).astype(np.int32)

    rand = (np.random.uniform(size=nsamp) * x.size).astype(np.int32)

    samples = np.zeros(nsamp)
    for i in xrange(nsamp):
        test, = np.where(cdfi >= rand[i])
        samples[i] = x[test[0]]

    return samples

# for multiprocessing classes
def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)
