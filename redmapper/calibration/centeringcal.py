from __future__ import division, absolute_import, print_function

import os
import numpy as np
import fitsio
import time
import scipy.optimize

from ..configuration import Configuration
from ..utilities import sample_from_pdf, histoGauss, chisq_pdf
from ..redsequence import RedSequenceColorPar
from ..background import Background
from ..cluster import ClusterCatalog
from ..galaxy import GalaxyCatalog

class WcenFgFitter(object):
    """
    """
    def __init__(self, w, lscale):
        self._w = w
        self._lscale = lscale

    def fit(self, p0):
        """
        """

        pars = scipy.optimize.fmin(self, p0, disp=False)

        return pars

    def __call__(self, pars):
        """
        """
        sig = pars[1] * self._lscale

        f = (1./(np.sqrt(2. * np.pi) * sig)) * np.exp(-0.5 * (np.log(self._w) - pars[0])**2. / (sig**2.))

        t = -np.sum(np.log(f))
        if pars[1] < 0.0:
            t += 1000.0

        return t

class WcenCFitter(object):
    """
    """
    def __init__(self, pcen, psat, mstar, lamscale, refmag, cwt, phi1, bcounts):
        self._pcen = pcen
        self._psat = psat
        self._mstar = mstar
        self._lamscale = lamscale
        self._refmag = refmag
        self._cwt = cwt
        self._phi1 = phi1
        self._bcounts = bcounts

    def fit(self, p0):
        """
        """

        pars = scipy.optimize.fmin(self, p0, disp=False)

        return pars

    def __call__(self, pars):
        """
        """
        mbar = self._mstar + pars[0] + pars[1] * self._lamscale
        phicen = (1./(np.sqrt(2.*np.pi) * pars[2])) * np.exp(-0.5*(self._refmag - mbar)**2. / (pars[2]**2.))
        rho = self._pcen * phicen * self._cwt + self._psat * self._phi1 * self._cwt + (1. - self._pcen - self._psat) * self._bcounts

        bad, = np.where((rho < 1e-5) | (~np.isfinite(rho)))
        rho[bad] = 1e-5

        t = -np.sum(np.log(rho))
        if pars[2] < 0.0: t += 1000

        return t

class WcenCwFitter(object):
    """
    """
    def __init__(self, pcen, psat, wcen, ffg, fsat, lscale):
        self._pcen = pcen
        self._psat = psat
        self._wcen = wcen
        self._ffg = ffg
        self._fsat = fsat
        self._lscale = lscale

    def fit(self, p0):
        """
        """

        pars = scipy.optimize.fmin(self, p0, disp=False)

        return pars

    def __call__(self, pars):
        """
        """
        sig = pars[1] * self._lscale

        fcen = (1. / (np.sqrt(2.*np.pi)*sig)) * np.exp(-0.5*(np.log(self._wcen) - pars[0])**2. / (sig**2.))

        f = self._pcen * fcen + self._psat * self._fsat + (1. - self._pcen - self._psat) * self._ffg

        t = -np.sum(np.log(f))
        if pars[1] < 0.0: t+=1000.0

        return t

class WcenCalibrator(object):
    """
    """

    def __init__(self, config, iteration, randcatfile=None, randsatcatfile=None):
        self.config = config

        if iteration == 1:
            if randcatfile is None:
                if self.config.lnw_fg_sigma < 0:
                    raise RuntimeError("randcatfile must be set on iteration 1, or lnw_fg_mean and lnw_fg_sigma must be set in configuration")
            if randsatcatfile is None:
                if self.config.lnw_sat_sigma < 0:
                    raise RuntimeError("randsatcatfile must be set on iteration 1, or lnw_sat_mean and lnw_sat_sigma must be set in configuration")

        self.randcatfile = randcatfile
        self.randsatcatfile = randsatcatfile
        self.iteration = iteration

    def run(self):
        """
        """

        # Calibrate the brightest galaxy from the schechter function
        self._schechter_montecarlo_calib()

        # Read in the parameters (fine steps)
        zredstr = RedSequenceColorPar(self.config.parfile, fine=True)

        # Read in the background
        bkg = Background(self.config.bkgfile)

        # Read in the catalog
        cat = ClusterCatalog.from_catfile(self.config.catfile,
                                          cosmo=self.config.cosmo)

        # We set the redshift according to the initial spec redshift for training
        cat.z = cat.z_spec_init

        # Select clusters for wcen training
        use, = np.where((cat.Lambda/cat.scaleval > self.config.wcen_minlambda) &
                        (cat.Lambda/cat.scaleval < self.config.wcen_maxlambda) &
                        (cat.w > 0.0) &
                        (cat.z > self.config.wcen_cal_zrange[0]) &
                        (cat.z < self.config.wcen_cal_zrange[1]) &
                        (cat.maskfrac < self.config.max_maskfrac))

        cat = cat[use]

        randfiles = [self.randcatfile, self.randsatcatfile]
        # note that the config variables might already be set...
        for randfile in randfiles:
            if randfile is None:
                continue

            rcat = ClusterCatalog.from_catfile(randfile, cosmo=self.config.cosmo)
            rcat.z = rcat.z_spec_init

            use, = np.where((rcat.Lambda/rcat.scaleval > self.config.wcen_minlambda) &
                            (rcat.Lambda/rcat.scaleval < self.config.wcen_maxlambda) &
                            (rcat.w > 0.0) &
                            (rcat.z > self.config.wcen_cal_zrange[0]) &
                            (rcat.z < self.config.wcen_cal_zrange[1]) &
                            (rcat.maskfrac < self.config.max_maskfrac))
            rcat = rcat[use]

            lscalefg = 1./np.sqrt((rcat.Lambda / rcat.scaleval) / self.config.wcen_pivot)

            fgfitter = WcenFgFitter(rcat.w, lscalefg)
            p0 = np.array([np.mean(np.log(rcat.w)), np.std(np.log(rcat.w))])
            p = fgfitter.fit(p0)

            if randfile == self.randcatfile:
                self.config.lnw_fg_mean = p[0]
                self.config.lnw_fg_sigma = p[1]
            else:
                self.config.lnw_sat_mean = p[0]
                self.config.lnw_sat_sigma = p[1]

        # Prepare the model fits
        mstars = zredstr.mstar(cat.z)

        # Get the starting values...
        def _linfunc(p, x, y):
            return (p[1] + p[0] * x) - y

        fit = scipy.optimize.least_squares(_linfunc, [0.0, 0.0], loss='soft_l1',
                                           args=(np.log(cat.Lambda / self.config.wcen_pivot),
                                                 cat.refmag - mstars))
        Delta0 = fit.x[1]
        Delta1 = fit.x[0]

        resid = (cat.refmag - mstars) - (Delta0 + Delta1*np.log(cat.Lambda / self.config.wcen_pivot))
        sigma_m = np.std(resid)

        # This needs to be made more elegant if this is a common use case.
        chisqs = zredstr.calculate_chisq(GalaxyCatalog(cat._ndarray), cat.z)

        cwt = chisq_pdf(chisqs, zredstr.ncol)

        mmstar1 = mstars + self.phi1_mmstar_m + self.phi1_mmstar_slope * np.log(cat.Lambda / self.config.wcen_pivot)
        phisig1 = self.phi1_msig_m + self.phi1_msig_slope * np.log(cat.Lambda / self.config.wcen_pivot)
        phi1 = (1./(np.sqrt(2.*np.pi)*phisig1)) * np.exp(-0.5*(cat.refmag - mmstar1)**2. / (phisig1**2.))

        sigma_g = bkg.sigma_g_lookup(cat.z, chisqs, cat.refmag)
        mpc_scale = np.radians(1.) * self.config.cosmo.Da(0, cat.z)
        bcounts = (sigma_g / mpc_scale**2.) * np.pi * cat.r_lambda**2.

        # pcen = cat.p_cen[0]
        # psat = cat.p_sat[0]

        lscale = np.log((cat.Lambda / cat.scaleval) / self.config.wcen_pivot)

        wfitter = WcenCFitter(cat.p_cen[:, 0], cat.p_sat[:, 0], mstars, lscale, cat.refmag,
                              cwt, phi1, bcounts)
        p0 = np.array([Delta0, Delta1, sigma_m])
        p = wfitter.fit(p0)

        fgsig = self.config.lnw_fg_sigma / np.sqrt((cat.Lambda / cat.scaleval) / self.config.wcen_pivot)
        ffg = (1./(np.sqrt(2.*np.pi) * fgsig)) * np.exp(-0.5*(np.log(cat.w) - self.config.lnw_fg_mean)**2. / (fgsig**2.))
        satsig = self.config.lnw_sat_sigma / np.sqrt((cat.Lambda / cat.scaleval) / self.config.wcen_pivot)
        fsat = (1./(np.sqrt(2.*np.pi) * satsig)) * np.exp(-0.5*(np.log(cat.w) - self.config.lnw_sat_mean)**2. / (satsig**2.))
        lscalefg = 1./np.sqrt((cat.Lambda / cat.scaleval) / self.config.wcen_pivot)

        cwfitter = WcenCwFitter(cat.p_cen[:, 0], cat.p_sat[:, 0], cat.w, ffg, fsat, lscalefg)
        p0 = np.array([np.mean(np.log(cat.w)), np.std(np.log(cat.w))])
        wp = cwfitter.fit(p0)

        # and save this...
        wcenstr = np.zeros(1, dtype=[('DELTA0', 'f8'),
                                     ('DELTA1', 'f8'),
                                     ('SIGMA_M', 'f8'),
                                     ('PIVOT', 'f8'),
                                     ('LNW_FG_MEAN', 'f8'),
                                     ('LNW_FG_SIGMA', 'f8'),
                                     ('LNW_SAT_MEAN', 'f8'),
                                     ('LNW_SAT_SIGMA', 'f8'),
                                     ('LNW_CEN_MEAN', 'f8'),
                                     ('LNW_CEN_SIGMA', 'f8')])
        wcenstr['DELTA0'] = p[0]
        wcenstr['DELTA1'] = p[1]
        wcenstr['SIGMA_M'] = p[2]
        wcenstr['PIVOT'] = self.config.wcen_pivot
        wcenstr['LNW_FG_MEAN'] = self.config.lnw_fg_mean
        wcenstr['LNW_FG_SIGMA'] = self.config.lnw_fg_sigma
        wcenstr['LNW_SAT_MEAN'] = self.config.lnw_sat_mean
        wcenstr['LNW_SAT_SIGMA'] = self.config.lnw_sat_sigma
        wcenstr['LNW_CEN_MEAN'] = wp[0]
        wcenstr['LNW_CEN_SIGMA'] = wp[1]

        fitsio.write(self.config.wcenfile, wcenstr, clobber=True)




    def _schechter_montecarlo_calib(self):
        """
        Calibrate the brightest galaxy sampled from a schechter function with a simple
        monte carlo
        """

        nmag = 100000
        mag = np.zeros(nmag)

        mstar = 0.0
        mrange = -2.5 * np.log10(np.array([10.0, 0.2]))
        step = 0.002

        def schechter(x, alpha=-1.0, mstar=0.0):
            return 10.**(0.4*(alpha + 1.0)*(mstar - x)) * np.exp(-10.**(0.4*(mstar - x)))

        mag = sample_from_pdf(schechter, mrange, step, nmag, alpha=self.config.calib_lumfunc_alpha, mstar=mstar)

        # We want to sample lambda galaxies from a schechter function...
        # And figure out the 3 brightest galaxies (m1, m2, m3)

        ntrial = 5000
        nlambdas = 9
        lambdas = np.linspace(20, 100, num=nlambdas, dtype=np.int32)

        m1 = np.zeros((nlambdas, ntrial))
        m2 = np.zeros_like(m1)
        m3 = np.zeros_like(m1)

        for i in xrange(ntrial):
            r = np.random.rand(nmag)
            st = np.argsort(r)

            for j in xrange(nlambdas):
                u = st[0:lambdas[j] - 1]
                st2 = np.argsort(mag[u])

                m1[j, i] = mag[u[st2[0]]]
                m2[j, i] = mag[u[st2[1]]]
                m3[j, i] = mag[u[st2[2]]]

        mmstar1_mean = np.zeros(nlambdas)
        mmstar1_sigma = np.zeros(nlambdas)
        mmstar2_mean = np.zeros(nlambdas)
        mmstar2_sigma = np.zeros(nlambdas)
        mmstar3_mean = np.zeros(nlambdas)
        mmstar3_sigma = np.zeros(nlambdas)

        for i in xrange(nlambdas):
            coeff = histoGauss(None, m1[i, :])
            mmstar1_mean[i] = coeff[1]
            mmstar1_sigma[i] = coeff[2]

            coeff = histoGauss(None, m2[i, :])
            mmstar2_mean[i] = coeff[1]
            mmstar2_sigma[i] = coeff[2]

            coeff = histoGauss(None, m3[i, :])
            mmstar3_mean[i] = coeff[1]
            mmstar3_sigma[i] = coeff[2]

        fit = np.polyfit(np.log(lambdas / self.config.wcen_pivot), mmstar1_mean, 1)

        self.phi1_mmstar_m = fit[1]
        self.phi1_mmstar_slope = fit[0]

        fit = np.polyfit(np.log(lambdas / self.config.wcen_pivot), mmstar1_sigma, 1)
        self.phi1_msig_m = fit[1]
        self.phi1_msig_slope = fit[0]

