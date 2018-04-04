from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import numpy as np
from scipy import special
import scipy.optimize

from .utilities import CubicSpline, interpol

class MedZFitter(object):
    """
    """
    def __init__(self, z_nodes, redshifts, values):
        self._z_nodes = z_nodes
        self._redshifts = redshifts
        self._values = values

    def fit(self, p0):
        """
        """
        # add bounds if we need it...

        pars = scipy.optimize.fmin(self, p0)

        return pars

    def __call__(self, pars):
        """
        """
        spl = redmapper.utilities.CubicSpline(self._z_nodes, pars)
        m = spl(self._redshifts)

        absdev = np.abs(self._values - m)
        t = np.sum(absdev.astype(np.float64))
        return t

class RedSequenceFitter(object):
    """
    """
    # NOTE: we will add in probabilities and backgrounds as well (!)
    def __init__(self, mean_nodes,
                 redshifts, colors, errs, mags=None, trunc=None,
                 slope_nodes=None, scatter_nodes=None):
        self._mean_nodes = np.atleast_1d(mean_nodes)
        if slope_nodes is None:
            self._slope_nodes = self._mean_nodes
        else:
            self._slope_nodes = np.atleast_1d(slope_nodes)
        if scatter_nodes is None:
            self._scatter_nodes = self._mean_nodes
        else:
            self._scatter_nodes = np.atleast_1d(scatter_nodes)

        self._redshifts = np.atleast_1d(redshifts)
        self._colors = np.atleast_1d(colors)
        self._mags = np.atleast_1d(mags)
        self._err2s = np.atleast_1d(errs)**2.

        self._n_mean_nodes = self._mean_nodes.size
        self._n_slope_nodes = self._slope_nodes.size
        self._n_scatter_nodes = self._scatter_nodes.size

        if self._redshifts.size != self._colors.size:
            raise ValueError("Number of redshifts must be equal to colors")
        if self._redshifts.size != self._errs.size:
            raise ValueError("Number of redshifts must be equal to errs")

        if trunc is not None:
            self._trunc = np.atleast_1d(trunc)
            if self._redshifts.size != self._trunc.size:
                raise ValueError("Number of redshifts must be equal to trunc")
        else:
            self._trunc = None

        if mags is not None:
            self._mags = np.atleast_1d(mags)
            if self._redshifts.size != self._mags.size:
                raise ValueError("Number of redshifts must be equal to mags")
            self._has_mags = True
        else:
            self._mags = np.zeros(self._redshifts.size)
            self._has_mags = False


    def fit(self, p0_mean, p0_slope, p0_scatter,
            fit_mean=False, fit_slope=False, fit_scatter=False):
        """
        """
        self._fit_mean = fit_mean
        self._fit_slope = fit_slope
        self._fit_scatter = fit_scatter

        if not self._has_mags and self._fit_scatter:
            raise ValueError("Can only do fit_scatter if mags were supplied")

        ctr = 0
        p0 = np.array([])
        if self._fit_mean:
            self._mean_index = 0
            ctr += self._n_mean_nodes
            p0 = p0.append(p0_mean)
        if self._fit_slope:
            self._slope_index = ctr
            ctr += self._n_slope_nodes
            p0 = p0.append(p0_slope)
        if self._fit_scatter:
            self._scatter_index = ctr
            ctr += self._n_scatter_nodes
            p0 = p0.append(p0_scatter)

        if ctr == 0:
            raise ValueError("Must select at least one of fit_mean, fit_slope, fit_scatter")

        # Precompute...
        if not self._fit_mean:
            spl = CubicSpline(self._mean_nodes, p0_mean)
            self._gmean = spl(self._redshifts)
        if not self._fit_slope:
            spl = CubicSpline(self._slope_nodes, p0_slope)
            self._gslope = spl(self._redshifts)
        if not self._fit_scatter:
            spl = CubicSpline(self._scatter_nodes, p0_scatter)
            self._gsig = np.sqrt(np.clip(spl(self._redshifts), 0.001, None)**2. + self._err2s)

        if not self._fit_mean and not self._fit_scatter and self._trunc is not None:
            self._phi_b = 0.5 * (1. + special.erf((self._trunc[1, :] - self._gmean) / self._gsig))
            self._phi_a = 0.5 * (1. + special.erf((self._trunc[0, :] - self._gmean) / self._gsig))

        pars = scipy.optimize.fmin(self, p0)

        retval = []
        if self._fit_mean:
            retval.append(pars[self._mean_index: self._mean_index + self._n_mean_nodes])
        if self._fit_slope:
            retval.append(pars[self._slope_index: self._slope_index + self._n_slope_nodes])
        if self._fit_scatter:
            retval.append(pars[self._scatter_index: self._scatter_index + self._n_scatter_nodes])

        return retval

    def __call__(self, pars):
        """
        """
        if self._fit_mean:
            # We are fitting the mean
            spl = CubicSpline(self._mean_nodes, pars[self._mean_index: self._mean_index + self._n_mean_nodes])
            gmean = spl(self._redshifts)
        else:
            gmean = self._gmean

        if self._fit_slope:
            # We are fitting the slope
            spl = CubicSpline(self._slope_nodes, pars[self._slope_index: self._slope_index + self._n_slope_nodes])
            gslope = spl(self._redshifts)
        else:
            gslope = self._gslope

        if self._fit_scatter:
            # We are fitting the scatter
            spl = CubicSpline(self._scatter_nodes, pars[self._scatter_index: self._scatter_index + self._n_slope_nodes])
            gsig = np.sqrt(np.clip(spl(self._redshifts), 0.001, None)**2. + self._err2s)

        if (self._fit_mean or self._fit_scatter) and self._trunc is not None:
            phi_b = 0.5 * (1. + special.erf((self._trunc[1, :] - gmean) / gsig))
            phi_a = 0.5 * (1. + special.erf((self._trunc[0, :] - gmean) / gsig))
        elif self._trunc is not None:
            phi_b = self._phi_b
            phi_a = self._phi_a

        xi = (self._values - gmean) / gsig

        phi = (1. / gsig) * (1. / np.sqrt(2. * np.pi)) * np.exp(-0.5 * xi**2.)

        if (self.trunc is not None):
            gci = phi / (phi_b - phi_a)
        else:
            gci = phi

        vals = np.log(gci)
        t = -np.sum(vals)

        return t

