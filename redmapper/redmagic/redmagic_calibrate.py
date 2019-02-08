from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import os
import numpy as np
import fitsio
import time
import scipy.optimize
import esutil
import healpy as hp

from ..configuration import Configuration
from ..fitters import MedZFitter
from ..redsequence import RedSequenceColorPar
from ..galaxy import GalaxyCatalog
from ..catalog import Catalog, Entry
from ..utilities import make_nodes, CubicSpline, interpol, read_members
from ..plotting import SpecPlot, NzPlot
from ..volumelimit import VolumeLimitMask, VolumeLimitMaskFixed


class RedmagicParameterFitter(object):
    """
    """
    def __init__(self, nodes, corrnodes, z, z_err,
                 chisq, mstar, zcal, zcal_err, refmag,
                 randomn, zmax, etamin, n0,
                 volume, zrange, zbinsize, zredstr, maxchi=20.0, ab_use=None):
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
        self._ab_use = ab_use

        self._maxchi = maxchi
        self._afterburner = False

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
        if self._z.size != self._refmag.size:
            raise ValueError("Number of z must be equal to refmag")
        if self._z.size != self._randomn.size:
            raise ValueError("Number of z must be equal to randomn")
        if len(self._zrange) != 2:
            raise ValueError("zrange must have 2 elements")

    def fit(self, p0_cval, p0_bias=None, p0_eratio=None, afterburner=False):
        """
        """

        if self._ab_use is None and afterburner:
            raise RuntimeError("Must set afterburner_use if using the afterburner")

        # always fit cval, that's what we're fitting.  The others are fit
        # inside the loop (oof)

        p0 = p0_cval

        self._afterburner = afterburner
        if afterburner:
            if p0_bias is None or p0_eratio is None:
                raise RuntimeError("Must set p0_bias, p0_ratio if using the afterburner")
            self._pars_bias = p0_bias
            self._pars_eratio = p0_eratio

        pars = scipy.optimize.fmin(self, p0, disp=False, xtol=1e-8, ftol=1e-8)

        retval = [pars]
        if afterburner:
            retval.append(self._pars_bias)
            retval.append(self._pars_eratio)

        return retval

    def __call__(self, pars):
        """
        """

        # chi2max is computed at the raw redshift
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
            self._pars_bias = mzfitter.fit(self._pars_bias, min_val=-0.1, max_val=0.1)

            # apply the bias
            spl = CubicSpline(self._corrnodes, self._pars_bias)
            z_redmagic = self._z - spl(self._z)

            # update mstar
            self._mstar = self._zredstr.mstar(z_redmagic)

            # fit the error fix
            y = 1.4826 * np.abs(self._z[ab_gd] - self._zcal[ab_gd]) / self._z_err[ab_gd]
            efitter = MedZFitter(self._corrnodes, self._z[ab_gd], y)
            self._pars_eratio = efitter.fit(self._pars_eratio, min_val=0.5, max_val=1.5)

            # apply the error fix
            spl = CubicSpline(self._corrnodes, self._pars_eratio)
            z_redmagic_e = self._z_err * spl(self._z)

        else:
            z_redmagic = self._z
            z_redmagic_e = self._z_err

        zsamp = self._randomn * z_redmagic_e + z_redmagic

        # Note that self._mstar is computed at z_redmagic

        gd, = np.where((self._chisq < chi2max) &
                       (self._refmag < (self._mstar - 2.5 * np.log10(self._etamin))) &
                       (z_redmagic < self._zmax))

        if gd.size == 0:
            # This is bad.
            return 1e11

        # Histogram the galaxies into bins
        h = esutil.stat.histogram(zsamp[gd],
                                  min=self._zrange[0], max=self._zrange[1] - 0.0001,
                                  binsize=self._zbinsize)

        # Compute density and error
        den = h.astype(np.float64) / self._volume
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

    def run(self, gals=None):
        """

        """
        import matplotlib.pyplot as plt

        # set gals for testing purposes...

        if gals is None:
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
        if self.config.depthfile is None or not os.path.isfile(self.config.depthfile):
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

        # Note that the area is already scaled properly!

        # make sure we compute area factors if less than full footprint.
        #area_factors = np.ones(nruns)
        #if self.config.d.hpix > 0:
        #    for i in xrange(nruns):
        #        astr = vlim_areas[i]
        #        total_area = astr.area[0]

        #        hpix, = np.where(vlim_masks[i].fracgood > 0.0)
        #        hpix += vlim_masks[i].offset

        #        theta, phi = hp.pix2ang(vlim_masks[i].nside, hpix)
        #        ipring = hp.ang2pix(self.config.d.nside, theta, phi)

        #        inpix, = np.where(ipring == self.config.d.hpix)

        #        pix_area = np.sum(vlim_masks[i].fracgood[hpix - vlim_masks[i].offset]) * hp.nside2pixarea(vlim_masks[i].nside, degrees=True)

        #        area_factors[i] = pix_area / total_area
        #        self.config.logger.info("Computed area_factor: %.2f" % (area_factors[i]))

        if gals is None:
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

        # This selects all *possible* redmagic galaxies
        gals = gals[use]
        mstar_init = mstar_init[use]

        # Run the galaxy cleaner
        # FIXME: implement cleaner

        # match to spectra
        if not self.config.redmagic_mock_truthspec:
            self.config.logger.info("Reading and matching spectra...")

            spec = Catalog.from_fits_file(self.config.specfile)
            use, = np.where(spec.z_err < 0.001)
            spec = spec[use]

            i0, i1, dists = gals.match_many(spec.ra, spec.dec, 3./3600., maxmatch=1)
            gals.zspec[i1] = spec.z[i0]
        else:
            self.config.logger.info("Using truth spectra for reference...")
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

        #a, b = esutil.numpy_util.match(gals.id, mem.id)
        #gals.zcal[a] = mem.z[b]
        #gals.zcal_e[a] = mem.z_err[b]

        # Hack this...
        i0, i1, dist = gals.match_many(mem.ra, mem.dec, 1./3600., maxmatch=1)
        gals.zcal[i1] = mem.z[i0]
        gals.zcal_e[i1] = mem.z_err[i0]

        # Clear out members
        mem = None

        self.config.redmagicfile = self.config.redmapper_filename('redmagic_calib')

        # loop over cuts...

        for i in range(nruns):
            self.config.logger.info("Working on %s: etamin = %.3f, n0 = %.3f" % (self.config.redmagic_names[i], self.config.redmagic_etas[i], self.config.redmagic_n0s[i]))

            redmagic_zrange = [self.config.redmagic_zrange[0],
                               self.config.redmagic_zmaxes[i]]

            corr_zrange = redmagic_zrange
            nbin = np.ceil((corr_zrange[1] - corr_zrange[0]) / self.config.redmagic_calib_zbinsize).astype(np.int32)
            corr_zrange[1] = nbin * self.config.redmagic_calib_zbinsize + corr_zrange[0]

            # Compute the nodes
            nodes = make_nodes(redmagic_zrange, self.config.redmagic_calib_nodesize)
            corrnodes = make_nodes(redmagic_zrange, self.config.redmagic_calib_corr_nodesize)

            # Prepare calibration structure
            vmaskfile = ''
            if isinstance(vlim_masks[i], VolumeLimitMask):
                vmaskfile = vlim_masks[i].vlimfile

            calstr = Entry(np.zeros(1, dtype=[('zrange', 'f4', 2),
                                              ('name', 'a%d' % (len(self.config.redmagic_names[i]) + 1)),
                                              ('maxchi', 'f4'),
                                              ('nodes', 'f8', nodes.size),
                                              ('etamin', 'f8'),
                                              ('n0', 'f8'),
                                              ('cmax', 'f8', nodes.size),
                                              ('corrnodes', 'f8', corrnodes.size),
                                              ('bias', 'f8', corrnodes.size),
                                              ('eratio', 'f8', corrnodes.size),
                                              ('vmaskfile', 'a%d' % (len(vmaskfile) + 1))]))

            calstr.zrange[:] = redmagic_zrange
            calstr.name = self.config.redmagic_names[i]
            calstr.maxchi = self.config.redmagic_calib_chisqcut
            calstr.nodes[:] = nodes
            calstr.corrnodes[:] = corrnodes
            calstr.etamin = self.config.redmagic_etas[i]
            calstr.n0 = self.config.redmagic_n0s[i]
            calstr.vmaskfile = vmaskfile

            # Initial histogram
            h = esutil.stat.histogram(gals.zuse,
                                      min=corr_zrange[0], max=corr_zrange[1] - 0.0001,
                                      binsize=self.config.redmagic_calib_zbinsize)
            zbins = np.arange(h.size, dtype=np.float64) * self.config.redmagic_calib_zbinsize + corr_zrange[0] + self.config.redmagic_calib_zbinsize / 2.

            # compute comoving volume
            #vol_default = np.zeros(zbins.size)
            #for j in xrange(zbins.size):
            #    vol_default[j] = (self.config.cosmo.V(zbins[j] - self.config.redmagic_calib_zbinsize/2.,
            #                                          zbins[j] + self.config.redmagic_calib_zbinsize/2.) *
            #                      (self.config.area * area_factors[i] / 41252.961))

            #dens = h.astype(np.float64) / vol_default

            etamin_ref = np.clip(self.config.redmagic_etas[i] - lstar_cushion, 0.1, None)

            # These are possible redmagic galaxies for this selection
            red_poss, = np.where(gals.refmag < (mstar_init - 2.5*np.log10(etamin_ref)))

            # Determine which of the galaxies to use in the afterburner
            gd, = np.where(gals.zcal[red_poss] > 0.0)
            ntrain = int(self.config.redmagic_calib_fractrain * gd.size)
            r = np.random.random(gd.size)
            st = np.argsort(r)
            afterburner_use = gd[st[0: ntrain]]

            # Compute the volume
            vmask = vlim_masks[i]
            astr = vlim_areas[i]

            aind = np.searchsorted(astr.z, zbins)
            z_areas = astr.area[aind] #* area_factors[i]

            volume = np.zeros(zbins.size)
            for j in xrange(zbins.size):
                volume[j] = (self.config.cosmo.V(zbins[j] - self.config.redmagic_calib_zbinsize/2.,
                                                 zbins[j] + self.config.redmagic_calib_zbinsize/2.) *
                                  (z_areas[j] / 41252.961))

            zmax = vmask.calc_zmax(gals.ra[red_poss], gals.dec[red_poss])

            if self.config.redmagic_calib_pz_integrate:
                randomn = np.random.normal(size=red_poss.size)
            else:
                randomn = np.zeros(red_poss.size)

            zsamp = randomn * gals.zuse_e[red_poss] + gals.zuse[red_poss]

            h = esutil.stat.histogram(zsamp,
                                      min=corr_zrange[0], max=corr_zrange[1] - 0.0001,
                                      binsize=self.config.redmagic_calib_zbinsize)
            dens = h.astype(np.float64) / volume

            bad, = np.where(dens < self.config.redmagic_n0s[i] * 1e-4)
            if bad.size > 0:
                self.config.logger.info("Warning: not enough galaxies at z=%s" % (zbins[bad].__str__()))

            # get starting values
            cmaxvals = np.zeros(nodes.size)

            aind = np.searchsorted(astr.z, nodes)
            test_areas = astr.area[aind] #* area_factors[i]

            test_vol = np.zeros(nodes.size)
            for j in xrange(nodes.size):
                test_vol[j] = (self.config.cosmo.V(nodes[j] - self.config.redmagic_calib_zbinsize/2.,
                                                   nodes[j] + self.config.redmagic_calib_zbinsize/2.) *
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

                st = np.argsort(gals.chisq[red_poss[u]])
                test_den = np.arange(u.size) / test_vol[j]
                ind = np.searchsorted(test_den, self.config.redmagic_n0s[i] * 1e-4)
                cmaxvals[j] = gals.chisq[red_poss[u[st[ind]]]]


            print(cmaxvals)

            # minimize chisquared parameters (need that function)
            rmfitter = RedmagicParameterFitter(nodes, corrnodes,
                                               gals.zuse[red_poss], gals.zuse_e[red_poss],
                                               gals.chisq[red_poss], mstar_init[red_poss],
                                               gals.zcal[red_poss], gals.zcal_e[red_poss],
                                               gals.refmag[red_poss], randomn,
                                               #self.config.redmagic_zmaxes[i],
                                               zmax,
                                               self.config.redmagic_etas[i],
                                               self.config.redmagic_n0s[i],
                                               volume, corr_zrange,
                                               self.config.redmagic_calib_zbinsize,
                                               zredstr,
                                               maxchi=self.config.redmagic_calib_chisqcut,
                                               ab_use=afterburner_use)

            self.config.logger.info("Fitting first pass...")
            cmaxvals, = rmfitter.fit(cmaxvals, afterburner=False)

            print(cmaxvals)

            # default is no bias or eratio
            biasvals = np.zeros(corrnodes.size)
            eratiovals = np.ones(corrnodes.size)

            # run with afterburner
            if self.config.redmagic_run_afterburner:
                self.config.logger.info("Fitting with afterburner...")
                cmaxvals, biasvals, eratiovals = rmfitter.fit(cmaxvals, biasvals, eratiovals, afterburner=True)

            print(cmaxvals)
            print(biasvals)
            print(eratiovals)

            # Record the calibrations
            calstr.cmax[:] = cmaxvals
            calstr.bias[:] = biasvals
            calstr.eratio[:] = eratiovals

            calstr.to_fits_file(self.config.redmagicfile, clobber=False, extname=calstr.name)

            # Compute the redmagic selection...
            # We're going to go back to all the galaxies here for simplicity
            # though this is a bit inefficient.

            gals.zredmagic = gals.zuse
            gals.zredmagic_e = gals.zuse_e

            # chi2max is computed at the raw, uncorrected redshift
            spl = CubicSpline(calstr.nodes, calstr.cmax)
            chi2max = np.clip(spl(gals.zuse), 0.1, self.config.redmagic_calib_chisqcut)

            if self.config.redmagic_run_afterburner:
                # Corrections are also computed at raw, uncorrected redshift
                spl = CubicSpline(calstr.corrnodes, calstr.bias)
                gals.zredmagic -= spl(gals.zuse)

                spl = CubicSpline(calstr.corrnodes, calstr.eratio)
                gals.zredmagic_e *= spl(gals.zuse)

            # Recompute mstar for the redmagic de-biased redshift
            mstar = zredstr.mstar(gals.zredmagic)

            # And zmax is for the redmagic de-biased redshift
            zmax = vmask.calc_zmax(gals.ra, gals.dec)

            gd, = np.where((gals.chisq < chi2max) &
                           (gals.refmag < (mstar - 2.5*np.log10(calstr.etamin))) &
                           (gals.zredmagic < zmax))

            # make pretty plots
            nzplot = NzPlot(self.config, binsize=self.config.redmagic_calib_zbinsize)
            nzplot.plot_redmagic_catalog(gals[gd], calstr.name, calstr.etamin, calstr.n0,
                                         vlim_areas[i], zrange=corr_zrange,
                                         sample=self.config.redmagic_calib_pz_integrate)

            # This stringification can be streamlined, I think.

            specplot = SpecPlot(self.config)

            # spectroscopic comparison
            okspec, = np.where(gals.zspec[gd] > 0.0)
            if okspec.size > 10:
                fig = specplot.plot_values(gals.zspec[gd[okspec]], gals.zredmagic[gd[okspec]],
                                           gals.zredmagic_e[gd[okspec]],
                                           name='z_{\mathrm{redmagic}}',
                                           title='%s: %3.1f-%02d' %
                                           (calstr.name, calstr.etamin, int(calstr.n0)),
                                           figure_return=True)
                fig.savefig(self.config.redmapper_filename('redmagic_calib_zspec_%s_%3.1f-%02d' %
                                                           (calstr.name, calstr.etamin,
                                                            int(calstr.n0)),
                                                           paths=(self.config.plotpath,),
                                                           filetype='png'))
                plt.close(fig)

            okcal, = np.where(gals.zcal[gd] > 0.0)
            if okcal.size > 10:
                fig = specplot.plot_values(gals.zcal[gd[okcal]], gals.zredmagic[gd[okcal]],
                                           gals.zredmagic_e[gd[okcal]],
                                           name='z_{\mathrm{redmagic}}',
                                           specname='z_{\mathrm{cal}}',
                                           title='%s: %3.1f-%02d' %
                                           (calstr.name, calstr.etamin, int(calstr.n0)),
                                           figure_return=True)
                fig.savefig(self.config.redmapper_filename('redmagic_calib_zcal_%s_%3.1f-%02d' %
                                                           (calstr.name, calstr.etamin,
                                                            int(calstr.n0)),
                                                           paths=(self.config.plotpath,),
                                                           filetype='png'))
                plt.close(fig)
