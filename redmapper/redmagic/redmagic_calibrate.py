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
from ..utilities import make_nodes, CubicSpline, interpol, read_members

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
        if not self.config.galfile_pixelized:
            raise RuntimeError("Code only runs with pixelized galfile.")

        if self.config.zredfile is None or not os.path.isfile(self.config.zredfile):
            raise RuntimeError("Must have zreds available.")

        if self.config.catfile is None or not os.path.isfile(self.config.catfile):
            raise RuntimeError("Must have a cluster catalog available.")

        # this is the number of calibration runs we have
        nruns = len(self.config.redmagic_etas)

        # check for vlim files
        vlim_masks = []
        vlim_areas = []
        if not os.path.isfile(self.config.depthfile):
            # If there is no depthfile, there are no proper vlim files...
            # So we're going to make temporary masks
            for vlim_lstar in self.config.redmagic_etas:
                vlim_masks.append(VolumeLimitMaskFixed(self.config))
                vlim_areas.append(vlim_masks[-1].get_areas())
        else:
            # There is a depth file so we can create/read vlim masks.
            for vlim_lstar in self.config.redmagic_etas:
                vlim_masks.append(VolumeLimitMask(self.config, vlim_lstar))
                vlim_areas.append(vlim_masks[-1].get_areas())

        # make sure we compute area factors if less than full footprint.
        area_factors = np.ones(nruns)
        if self.config.d.hpix > 0:
            for i in xrange(nruns):
                astr = vlim_areas[i]
                total_area = astr.area[0]

                hpix, = np.where(vlim_masks[i].fracgood > 0.0)
                hpix += vlim_masks[i].offset

                theta, phi = hp.pix2ang(vlim_masks[i].nside, hpix)
                ipring = hp.ang2pix(self.config.d.nside, theta, phi)

                inpix, = np.where(ipring == self.config.d.hpix)

                pix_area = np.sum(vlim_masks[i].fracgood[hpix - vlim_masks[i].offset]) * hp.nside2pixarea(vlim_masks[i].nside, degrees=True)

                area_factors[i] = pix_area / total_area
                self.config.log.info("Computed area_factor: %.2f" % (area_factors[i]))

        # Read in galaxies with zreds
        gals = GalaxyCatalog.from_galfile(self.config.galfile,
                                          nside=self.config.d.nside,
                                          hpix=self.config.d.hpix,
                                          border=self.config.border,
                                          zredfile=self.config.zredfile,
                                          truth=self.config.redmagic_mock_truthspec)

        # Add redmagic fields
        gals.add_fields([('zuse', 'f4'),
                         ('zuse_e', 'f4'),
                         ('zspec', 'f4'),
                         ('zcal', 'f4'),
                         ('zcal_e', 'f4'),
                         ('zredmagic', 'f4'),
                         ('zredmagic_e', 'f4')])

        gals.zuse = gals.zred_uncorr
        gals.zuse_e = gals.zred_uncorr_e

        zredstr = RedSequenceColorPar(self.config.parfile, fine=True)

        mstar_init = zredstr.mstar(gals.zuse)

        # modify zrange for even bins, including cushion
        corr_zrange = self.config.redmagic_zrange
        nbin = np.ceil((corr_zrange[1] - corr_zrange[0]) / self.config.redmagic_calib_zbinsize).astype(np.int32)
        corr_zrange[1] = nbin * self.config.redmagic_calib_zbinsize + corr_zrange[0]

        if self.config.redmagic_run_afterburner:
            lstar_cushion = 0.05
            z_cushion = 0.05
        else:
            lstar_cushion = 0.0
            z_cushion = 0.0

        # Cut input galaxies
        cut_zrange = [corr_zrange[0] - z_cushion, corr_zrange[1] + z_cushion]
        minlstar = np.clip(np.min(self.config.redmagic_etas) - lstar_cushion, 0.1, None)

        use, = np.where((gals.zuse > cut_zrange[0]) & (gals.zuse < cut_zrange[1]) &
                        (gals.chisq < self.config.redmagic_calib_chisqcut) &
                        (gals.refmag < (mstar_init - 2.5*np.log10(minlstar))))

        if use.size == 0:
            raise RuntimeError("No galaxies in redshift range/chisq range/eta range.")

        gals = gals[use]

        # Run the galaxy cleaner
        # FIXME: implement cleaner

        # match to spectra
        if not self.config.redmagic_mock_truthspec:
            self.config.log.info("Reading and matching spectra...")

            spec = Catalog.from_fits_file(self.config.specfile)
            use, = np.where(spec.z_err < 0.001)
            spec = spec[use]

            i0, i1, dists = gals.match_many(spec.ra, spec.dec, 3./3600., maxmatch=1)
            gals.zspec[i1] = spec.z[i0]
        else:
            self.config.log.info("Using truth spectra for reference...")
            gals.zspec = gals.ztrue

        # Match to cluster catalog

        cat = Catalog.from_fits_file(self.config.catfile)
        mem = read_members(self.config.catfile)

        mem.add_fields([('z_err', 'f4')])

        a, b = esutil.numpy_util.match(cat.mem_match_id, mem.mem_match_id)
        mem.z_err[b] = cat.z_lambda_e[a]

        use, = np.where(mem.p > self.config.calib_corr_pcut)
        mem = mem[use]

        gals.zcal[:] = -1.0
        gals.zcal_e[:] = -1.0

        a, b = esutil.numpy_util.match(gals.id, mem.id)
        gals.zcal[a] = mem.z[b]
        gals.zcal_e[a] = mem.z_err[b]

        # Clear out members
        mem = None

        self.config.redmagicfile = self.config.redmapper_filename('redmagic_calib')

        # loop over cuts...

        for i in range(nruns):
            self.config.log("Working on %s: etamin = %.3f, n0 = %.3f" % (self.config.redmagic_names[i], self.config.redmagic_eta[i], self.config.redmagic_n0[i]))

            redmagic_zrange = [self.config.redmagic_zrange[0],
                               self.config.redmagic_zmaxes[i]]

            corr_zrange = redmagic_zrange
            nbin = np.ceil((corr_zrange[1] - corr_zrange[0]) / self.config.redmagic_calib_zbinsize).astype(np.int32)
            corr_zrange[1] = nbin * self.config.redmagic_calib_zbinsize + corr_zrange[0]

            # Compute the nodes
            nodes = make_nodes(redmagic_zrange, self.config.redmagic_calib_nodesize)
            corrnodes = make_nodes(redmagic_zrange, self.config.calib_corr_nodesize)

            # Initial histogram
            h = esutil.stat.histogram(gals.zuse,
                                      min=corr_zrange[0], max=corr_zrange[1] - 0.0001,
                                      binsize=self.config.redmagic_calib_zbinsize)
            zbins = np.arange(h.size, dtype=np.float64) * self.config.redmagic_calib_zbinsize + corr_zrange[0] + self.config.redmagic_calib_zbinsize / 2.

            # compute comoving volume
            vol_default = np.zeros(zbins.size)
            for j in xrange(zbins.size):
                vol_default[j] = (self.config.cosmo.V(zbins[j] - self.redmagic_calib_zbinsize/2.,
                                                      zbins[j] + self.redmagic_calib_zbinsize/2.) *
                                  (self.config.area * area_factors[i] / 41252.961))

            dens = h.astype(np.float64) / vol_default

            # FIXME
            #bad, = np.where(dens < n0s[i])
            #if bad.size > 0:
            #    self.config.log.info("Problem: not enough galaxies at z= %s" % (zbins[bad].__str__()))
            #    continue

            etamin_ref = np.clip(self.config.redmagic_etas[i] - lstar_cushion, 0.1, None)
            use, = np.where(gals.refmag < (mstar_init - 2.5*np.log10(etamin_ref)))

            # Determine which of the galaxies to use in the afterburner
            gd, = np.where(gals.zcal[use] > 0.0)
            ntrain = int(self.config.redmagic_calib_fractrain * gd.size)
            r = np.random.random(gd.size)
            st = np.argsort(r)
            afterburner_use = gd[st[0: ntrain]]

            # Compute the volume
            vmask = vlim_masks[i]
            astr = vlim_areas[i]

            aind = np.searchsorted(astr.z, zbins)
            z_areas = astr.area[aind] * area_factors[i]

            volume = np.zeros(zbins.size)
            for j in xrange(zbins.size):
                volume[j] = (self.config.cosmo.V(zbins[j] - self.redmagic_calib_zbinsize/2.,
                                                 zbins[j] + self.redmagic_calib_zbinsize/2.) *
                                  (z_areas[j] / 41252.961))

            zmax = vmask.calc_zmax(gals.ra[use], gals.dec[use])

            randomn = np.random.normal(use.size)
            zsamp = randomn * gals.zuse_e[use] + gals.zuse[use]

            h = esutil.stat.histogram(zsamp,
                                      min=corr_zrange[0], max=corr_zrange[1] - 0.0001,
                                      binsize=self.config.redmagic_calib_zbinsize)
            dens = h.astype(np.float64) / volume

            bad, = np.where(dens < self.config.redmagic_n0[i] * 1e-4)
            if nbad > 0:
                self.config.log.info("Warning: not enough galaxies at z=%s" % (zbins[bad].__str__()))

            # get starting values
            cmaxvals = np.zeros(nodes.size)

            aind = np.searchsorted(astr.z, nodes)
            test_areas = astr.area[aind] * area_factors[i]

            test_vol = np.zeros(nodes.size)
            for j in xrange(nodes.size):
                volume[j] = (self.config.cosmo.V(nodes[j] - self.redmagic_calib_zbinsize/2.,
                                                 nodes[j] + self.redmagic_calib_zbinsize/2.) *
                             (test_areas[j] / 41252.961))

            for j in xrange(nodes.size):
                zrange = [nodes[j] - self.config.redmagic_calib_zbinsize/2.,
                          nodes[j] + self.config.redmagic_calib_zbinsize/2.]
                if j == 0:
                    zrange = [nodes[j],
                              nodes[j] + self.config.redmagic_calib_zbinsize]
                elif j == nodes.size - 1:
                    zrange = [nodes[j] - self.config.redmagic_calib_zbinsize,
                              nodes[j]]

                u, = np.where((zsamp > zrange[0]) & (zsamp < zrange[1]))

                st = np.argsort(gals.chisq[use[u]])
                test_den = np.arange(u.size) / test_vol[j]
                ind = np.searchsorted(test_den, self.config.redmagic_n0s[i] * 1e-4)
                cmaxvals[k] = gals.chisq[use[u[st[ind]]]]

            # minimize chisquared parameters (need that function)
            rmfitter = RedmagicParameterFitter(nodes, corrnodes,
                                               zsamp, gals.zuse_e[use],
                                               gals.chisq[use], mstar_init[use],
                                               gals.zcal[use], gals.zcal_e[use],
                                               gals.refmag[use], randomn,
                                               self.config.redmagic_zmaxes[i],
                                               self.config.redmagic_eta[i],
                                               self.config.redmagic_n0[i],
                                               volume, corr_zrange,
                                               self.config.redmagic_calib_zbinsize,
                                               zredstr, maxchi=self.config.redmagic_calib_chisqcut,
                                               ab_use=blah)
            cmaxvals, blah, blah = rmfitter.fit(cmaxvals, blah, blah, afterburner=False)

            # run with afterburner
            if self.config.redmagic_run_afterburner:
                cmaxvals, blah, blah = rmfitter.fit(cmaxvals, blah, blah, afterburner=True)

            # Save the calibrations

            # make pretty plots



