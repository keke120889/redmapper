from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import os
import numpy as np
import fitsio
import time
from scipy.optimize import least_squares

from ..configuration import Configuration
from ..fitters import MedZFitter
from ..redsequence import RedSequenceColorPar
from ..galaxy import GalaxyCatalog
from ..catalog import Catalog, Entry
from ..utilities import make_nodes, CubicSpline, interpol

class RedmagicParameterFitter(object):
    """
    """
    def __init__(self, nodes, corrnodes, z, z_err,
                 chisq, mstar, zcal, zcal_err, refmag,
                 randomn, zmax, etamin, n0,
                 volume, zrange, zbinsize, zredstr, maxchi=20.0)
        """
        """
        self._nodes = np.atleast_1d(nodes)
        self._corrnodes = np.atleast_1d(corrnodes)

        self._n_nodes = self._nodes.size
        self._n_corrnodes = self._corrnodes.size

        self._z = np.atleast_1d(z)
        self._z_err = np.atleast_1d(z_err)
        self._chisq = np.atleast_1d(chisq)
        self._mstar = np.atleast_1d(mstar)
        self._zcal = np.atleast_1d(zcal)
        self._zcal_err = np.atleast_1d(zcal_err)
        self._refmag = np.atleast_1d(refmag)
        self._randomn = np.atleast_1d(randomn)
        self._zmax = zmax
        self._etamin = etamin
        self._n0 = n0
        self._volume = volume
        self._zrange = zrange
        self._zbinsize = zbinsize
        self._zredstr = zredstr
        #self.run_afterburner = run_afterburner

        self._maxchi = maxchi

        if self._z.size != self._z_err.size:
            raise ValueError("Number of z must be equal to z_err")
        if self._z.size != self._chisq.size:
            raise ValueError("Number of z must be equal to chisq")
        if self._z.size != self._mstar.size:
            raise ValueError("Number of z must be equal to mstar")
        if self._z.size != self._zcal.size:
            raise ValueError("Number of z must be equal to zcal")
        if self._z.size != self._zcal_err.size:
            raise ValueError("Number of z must be equal to zcal_err")
        if self._refmag.size != self._refmag.size:
            raise ValueError("Number of z must be equal to refmag")
        if self._randomn.size != self._randomn.size:
            raise ValueError("Number of z must be equal to randomn")
        if len(self._zrange) != 2:
            raise ValueError("zrange must have 2 elements")

    def fit(self, p0_cval, p0_bias, p0_eratio, afterburner=False):
        """
        """

        # always fit cval, that's what we're fitting.  The others are fit
        # inside the loop (oof)

        p0 = p0_cval

        self._afterburner = afterburner
        if afterburner:
            self._pars_bias = p0_bias
            self._pars_eratio p0_eratio

        pars = scipy.optimize.fmin(self, p0, disp=False, xtol=1e-6, ftol=1e-6)

        #self._fit_cval = True
        #if afterburner:
        #    self._fit_bias = True
        #    self._fit_eratio = True

        #ctr = 0
        #p0 = np.array([])
        #if self._fit_cval:
        #    self._cval_index = 0
        #    ctr += self._n_nodes
        #    p0 = np.append(p0, p0_cval)
        #if self._fit_bias:
        #    self._bias_index = ctr
        #    ctr += self._n_corr_nodes
        #    p0 = np.append(p0, p0_bias)
        #if self._fit_eratio:
        #    self._eratio_index = ctr
        #    ctr += self._n_corr_nodes
        #    p0 = np.append(p0, p0_eratio)

        #pars = scipy.optimize.fmin(self, p0, disp=False, xtol=1e-6, ftol=1e-6)

        retval = []
        retval = [pars]
        if afterburner:
            retval.append(self._pars_bias)
            retval.append(self._pars_eratio)

        return retval

    def __call__(self, pars):
        """
        """

        spl = CubicSpline(self._nodes, pars)
        chi2max = np.clip(spl(self._z), 0.1, self._maxchi)

        gd, = np.where(self._chisq < chi2max)
        if gd.size == 0:
            # This is bad.
            return 1e11

        if self._afterburner:
            # get good afterburner guys
            ab_gd, = np.where((self._chisq[self._ab_use] < chi2max[self._ab_use]) &
                              (self._refmag[self._ab_use] < (self._mstar[self._ab_use] - 2.5 * np.log10(self._etamin))))
            ab_gd = self._ab_use[ab_gd]

            # fit the bias
            mzfitter = MedZFitter(self._corrnodes, self._z[ab_gd], self._z[ab_gd] - self._zcal[ab_gd])
            self._pars_bias = mzfitter.fit(self._pars_bias)

            # apply the bias
            spl = CubicSpline(self._corrnodes, self._pars_bias)
            z_redmagic = self._z - spl(self._z)

            # update mstar
            self._mstar = self._zredstr.mstar(z_redmagic)

            # fit the error fix
            y = 1.4826 * np.abs(self._z[ab_gd] - self._zcal[ab_gd]) / self._z_err[ab_gd]
            efitter = MedZFitter(self._corrnodes, self._z[ab_gd], y)
            self._pars_eratio = efitter.fit(self._pars_eratio)

            # apply the error fix
            spl = CubicSpline(self._corrnodes, self._pars_eratio)
            z_redmagic_e = self._z_err * spl(self._z)
        else:
            z_redmagic = self._z
            z_redmagic_e = self._z_err

        zsamp = self._randomn * z_redmagic_e + z_redmagic

        gd, = np.where((self._chisq < chi2max) &
                       (self.refmag < (self._mstar - 2.5 * np.log10(self._etamin))) &
                       (z_redmagic < self._zmax))

        if gd.size == 0:
            # This is bad.
            return 1e11

        # Histogram the galaxies into bins
        h = esutil.stat.histogram(zsamp[gd],
                                  min=self._zrange[0], max=self._zrange[1] - 0.0001,
                                  self._zbinsize)

        # Compute density and error
        den = float(h) / self._volume
        den_err = np.sqrt(self._n0 * 1e-4 * self._volume) / self._volume

        # Compute cost function
        t = np.sum(((den - self._n0 * 1e-4) / den_err)**2.)

        return t


class RedmagicCalibrator(object):
    """
    """
    def __init__(self, conf):
        """
        """
        if not isinstance(conf, Configuration):
            self.config = Configuration(conf)
        else:
            self.config = conf

    def run(self):
        """
        """

        # make sure that we have pixelized file, zreds, etc.

        # check for vlim files

        # Create vlim if possible

        # make sure we compute area factors if less than full footprint.

        # Read in galaxies with zreds

        # Add redmagic fields

        # modify zrange for even bins, including cushion

        # Cut input galaxies

        # Run the galaxy cleaner

        # match to spectra???

        # Match to cluster catalog

        # loop over cuts...

        # make nodes

        # compute comoving volume
        vol[i] = (self.config.cosmo.V(zbins[i] - self.binsize/2.,
                                      zbins[i] + self.binsize/2.) *
                  (areastr.area[indices[i]] / 41252.961))


        # using vlim of course...
        # set zmax

        # Do initial binning, with warning

        # get starting values

        # minimize chisquared parameters (need that function)

        # run with afterburner

        # make pretty plots

        pass
