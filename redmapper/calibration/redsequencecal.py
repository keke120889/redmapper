from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import os
import numpy as np
import fitsio
import time
from scipy.optimize import least_squares

from ..configuration import Configuration
from ..fitters import MedZFitter, RedSequenceFitter, RedSequenceOffDiagonalFitter, CorrectionFitter
from ..redsequence import RedSequenceColorPar
from ..color_background import ColorBackground
from ..galaxy import GalaxyCatalog
from ..catalog import Catalog, Entry
from ..zred_color import ZredColor
from ..utilities import make_nodes, CubicSpline, interpol

class RedSequenceCalibrator(object):
    """
    """

    def __init__(self, conf, galfile):
        if not isinstance(conf, Configuration):
            self.config = Configuration(conf)
        else:
            self.config = conf

        self._galfile = galfile

    def run(self, doRaise=True):
        """
        """

        gals = GalaxyCatalog.from_galfile(self._galfile)

        if self.config.calib_use_pcol:
            use, = np.where((gals.z > self.config.zrange[0]) &
                            (gals.z < self.config.zrange[1]) &
                            (gals.pcol > self.config.calib_pcut))
        else:
            use, = np.where((gals.z > self.config.zrange[0]) &
                            (gals.z < self.config.zrange[1]) &
                            (gals.p > self.config.calib_pcut))

        if use.size == 0:
            raise RuntimeError("No good galaxies in %s!" % (self._galfile))

        gals = gals[use]

        nmag = self.config.nmag
        ncol = nmag - 1

        # Reference mag nodes for pivot
        pivotnodes = make_nodes(self.config.zrange, self.config.calib_pivotmag_nodesize)

        # Covmat nodes
        covmatnodes = make_nodes(self.config.zrange, self.config.calib_covmat_nodesize)

        # correction nodes
        corrnodes = make_nodes(self.config.zrange, self.config.calib_corr_nodesize)

        # correction slope nodes
        corrslopenodes = make_nodes(self.config.zrange, self.config.calib_corr_slope_nodesize)

        # volume factor (hard coded)
        volnodes = make_nodes(self.config.zrange, 0.01)

        # Start building the par dtype
        dtype = [('pivotmag_z', 'f4', pivotnodes.size),
                 ('pivotmag', 'f4', pivotnodes.size),
                 ('minrefmag', 'f4', pivotnodes.size),
                 ('maxrefmag', 'f4', pivotnodes.size),
                 ('medcol', 'f4', (pivotnodes.size, ncol)),
                 ('medcol_width', 'f4', (pivotnodes.size, ncol)),
                 ('covmat_z', 'f4', covmatnodes.size),
                 ('sigma', 'f4', (ncol, ncol, covmatnodes.size)),
                 ('covmat_amp', 'f4', (ncol, ncol, covmatnodes.size)),
                 ('covmat_slope', 'f4', (ncol, ncol, covmatnodes.size)),
                 ('corr_z', 'f4', corrnodes.size),
                 ('corr', 'f4', corrnodes.size),
                 ('corr_slope_z', 'f4', corrslopenodes.size),
                 ('corr_slope', 'f4', corrslopenodes.size),
                 ('corr_r', 'f4', corrslopenodes.size),
                 ('corr2', 'f4', corrnodes.size),
                 ('corr2_slope', 'f4', corrslopenodes.size),
                 ('corr2_r', 'f4', corrslopenodes.size),
                 ('volume_factor_z', 'f4', volnodes.size),
                 ('volume_factor', 'f4', volnodes.size)]

        # And for each color, make the nodes
        node_dict = {}
        self.ztag = [None] * ncol
        self.ctag = [None] * ncol
        self.zstag = [None] * ncol
        self.stag = [None] * ncol
        for j in xrange(ncol):
            self.ztag[j] = 'z%02d' % (j)
            self.ctag[j] = 'c%02d' % (j)
            self.zstag[j] = 'zs%02d' % (j)
            self.stag[j] = 'slope%02d' % (j)

            node_dict[self.ztag[j]] = make_nodes(self.config.zrange, self.config.calib_color_nodesizes[j],
                                                 maxnode=self.config.calib_color_maxnodes[j])
            node_dict[self.zstag[j]] = make_nodes(self.config.zrange, self.config.calib_slope_nodesizes[j],
                                                  maxnode=self.config.calib_color_maxnodes[j])

            dtype.extend([(self.ztag[j], 'f4', node_dict[self.ztag[j]].size),
                          (self.ctag[j], 'f4', node_dict[self.ztag[j]].size),
                          (self.zstag[j], 'f4', node_dict[self.zstag[j]].size),
                          (self.stag[j], 'f4', node_dict[self.zstag[j]].size)])

        # Make the pars ... and fill them with the defaults
        self.pars = Entry(np.zeros(1, dtype=dtype))

        self.pars.pivotmag_z = pivotnodes
        self.pars.covmat_z = covmatnodes
        self.pars.corr_z = corrnodes
        self.pars.corr_slope_z = corrslopenodes
        self.pars.volume_factor_z = volnodes

        for j in xrange(ncol):
            self.pars._ndarray[self.ztag[j]] = node_dict[self.ztag[j]]
            self.pars._ndarray[self.zstag[j]] = node_dict[self.zstag[j]]

        # And a special subset of color galaxies
        if self.config.calib_use_pcol:
            coluse, = np.where(gals.pcol > self.config.calib_color_pcut)
        else:
            coluse, = np.where(gals.p > self.config.calib_color_pcut)

        colgals = gals[coluse]

        # And a placeholder zredstr which allows us to do stuff
        self.zredstr = RedSequenceColorPar(None, config=self.config)

        # And read the color background
        self.bkg = ColorBackground(self.config.bkgfile_color)

        # And prepare for luptitude corrections
        if self.config.b[0] == 0.0:
            self.do_lupcorr = False
        else:
            self.do_lupcorr = True
            self.bnmgy = self.config.b * 1e9
            self.lupzp = 22.5

        # Compute pivotmags
        self._calc_pivotmags(colgals)

        # Compute median colors
        self._calc_medcols(colgals)

        # Compute diagonal parameters
        self._calc_diagonal_pars(gals, doRaise=doRaise)

        # Compute off-diagonal parameters
        self._calc_offdiagonal_pars(gals, doRaise=doRaise)

        # Compute volume factor
        self._calc_volume_factor(self.config.zrange[1])

        # Write out the parameter file
        self.save_pars(self.config.parfile, clobber=False)

        # Compute zreds without corrections
        # Later will want this parallelized, I think
        self._calc_zreds(gals, do_correction=False)

        # Compute correction (mode1)
        self._calc_corrections(gals)

        # Compute correction (mode2)
        self._calc_corrections(gals, mode2=True)

        # And re-save the parameter file
        self.save_pars(self.config.parfile, clobber=True)

        # Recompute zreds with corrections
        # Later will want this parallelized, I think
        self._calc_zreds(gals, do_correction=True)

        # And want to save galaxies and zreds
        zredfile = self._galfile.rstrip('.fit') + '_zreds.fit'
        gals.to_fits_file(zredfile)

        # Make diagnostic plots
        self._make_diagnostic_plots(gals)

    def _compute_startvals(self, nodes, z, val, xval=None, err=None, median=False, fit=False, mincomp=3):
        """
        """

        def _linfunc(p, x, y):
            return (p[1] + p[0] * x) - y

        if (not median and not fit) or (median and fit):
            raise RuntimeError("Must select one and only one of median and fit")

        if median:
            mvals = np.zeros(nodes.size)
            scvals = np.zeros(nodes.size)
        else:
            cvals = np.zeros(nodes.size)
            svals = np.zeros(nodes.size)

        if err is not None:
            if err.size != val.size:
                raise ValueError("val and err must be the same length")

            # default all to 0.1
            evals = np.zeros(nodes.size) + 0.1
        else:
            evals = None

        for i in xrange(nodes.size):
            if i == 0:
                zlo = nodes[0]
            else:
                zlo = (nodes[i - 1] + nodes[i]) / 2.
            if i == nodes.size - 1:
                zhi = nodes[i]
            else:
                zhi = (nodes[i] + nodes[i + 1]) / 2.

            u, = np.where((z > zlo) & (z < zhi))

            if u.size < mincomp:
                if i > 0:
                    if median:
                        mvals[i] = mvals[i - 1]
                        scvals[i] = scvals[i - 1]
                    else:
                        cvals[i] = cvals[i - 1]
                        svals[i] = svals[i - 1]

                    if err is not None:
                        evals[i] = evals[i - 1]
            else:
                if median:
                    mvals[i] = np.median(val[u])
                    scvals[i] = np.median(np.abs(val[u] - mvals[i]))
                else:
                    fit = least_squares(_linfunc, [0.0, 0.0], loss='soft_l1', args=(xval[u], val[u]))
                    cvals[i] = fit.x[1]
                    svals[i] = np.clip(fit.x[0], None, 0.0)

                if err is not None:
                    evals[i] = np.median(err[u])

        if median:
            return mvals, scvals
        else:
            return cvals, svals, evals

    def _compute_single_lupcorr(self, j, cvals, svals, gals, dmags, mags, lups, mind, sign):
        spl = CubicSpline(self.pars._ndarray[self.ztag[j]], cvals)
        cv = spl(gals.z)
        spl = CubicSpline(self.pars._ndarray[self.zstag[j]], svals)
        sv = spl(gals.z)

        mags[:, mind] = mags[:, mind + sign] + sign * (cv + sv * dmags)

        flux = 10.**((mags[:, mind] - self.lupzp) / (-2.5))
        lups[:, mind] = 2.5 * np.log10(1.0 / self.config.b[mind]) - np.arcsinh(0.5 * flux / self.bnmgy[mind]) / (0.4 * np.log(10.0))

        magcol = mags[:, j] - mags[:, j + 1]
        lupcol = lups[:, j] - lups[:, j + 1]

        lupcorr = lupcol - magcol

        return lupcorr

    def _calc_pivotmags(self, gals):
        """
        """

        print("Calculating pivot magnitudes...")

        # With binning, approximate the positions for starting the fit
        pivmags = np.zeros_like(self.pars.pivotmag_z)

        for i in xrange(pivmags.size):
            pivmags, _ = self._compute_startvals(self.pars.pivotmag_z, gals.z, gals.refmag, median=True)

        medfitter = MedZFitter(self.pars.pivotmag_z, gals.z, gals.refmag)
        pivmags = medfitter.fit(pivmags)

        self.pars.pivotmag = pivmags

        # and min and max...
        self.pars.minrefmag = self.zredstr.mstar(self.pars.pivotmag_z) - 2.5 * np.log10(30.0)
        lval_min = np.clip(self.config.lval_reference - 0.1, 0.001, None)
        self.pars.maxrefmag = self.zredstr.mstar(self.pars.pivotmag_z) - 2.5 * np.log10(lval_min)

    def _calc_medcols(self, gals):
        """
        """

        print("Calculating median colors...")

        ncol = self.config.nmag - 1

        galcolor = gals.galcol

        for j in xrange(ncol):
            col = galcolor[:, j]

            # get the start values
            mvals, scvals = self._compute_startvals(self.pars.pivotmag_z, gals.z, col, median=True)

            # compute the median
            medfitter = MedZFitter(self.pars.pivotmag_z, gals.z, col)
            mvals = medfitter.fit(mvals)

            # and the scatter
            spl = CubicSpline(self.pars.pivotmag_z, mvals)
            med = spl(gals.z)
            medfitter = MedZFitter(self.pars.pivotmag_z, gals.z, np.abs(col - med))
            scvals = medfitter.fit(scvals)

            self.pars.medcol[:, j] = mvals
            self.pars.medcol_width[:, j] = 1.4826 * scvals

    def _calc_diagonal_pars(self, gals, doRaise=True):
        """
        """

        # The main routine to compute the red sequence on the diagonal

        ncol = self.config.nmag - 1

        galcolor = gals.galcol
        galcolor_err = gals.galcol_err

        # compute the pivot mags
        spl = CubicSpline(self.pars.pivotmag_z, self.pars.pivotmag)
        pivotmags = spl(gals.z)

        # And set the right probabilities
        if self.config.calib_use_pcol:
            probs = gals.pcol
        else:
            probs = gals.p

        # Figure out the order of the colors for luptitude corrections
        mags = np.zeros((gals.size, self.config.nmag))

        if self.do_lupcorr:
            col_indices = np.zeros(ncol, dtype=np.int32)
            sign_indices = np.zeros(ncol, dtype=np.int32)
            mind_indices = np.zeros(ncol, dtype=np.int32)

            c=0
            for j in xrange(self.config.ref_ind, self.config.nmag):
                col_indices[c] = j - 1
                sign_indices[c] = -1
                mind_indices[c] = j
                c += 1
            for j in xrange(self.config.ref_ind - 2, -1, -1):
                col_indices[c] = j
                sign_indices[c] = 1
                mind_indices[c] = j
                c += 1

            lups = np.zeros_like(mags)

            mags[:, self.config.ref_ind] = gals.mag[:, self.config.ref_ind]
            flux = 10.**((mags[:, self.config.ref_ind] - self.lupzp) / (-2.5))
            lups[:, self.config.ref_ind] = 2.5 * np.log10(1.0 / self.config.b[self.config.ref_ind]) - np.arcsinh(0.5 * flux / self.bnmgy[self.config.ref_ind]) / (0.4 * np.log(10.0))
        else:
            col_indices = np.arange(ncol)
            sign_indices = np.ones(ncol, dtype=np.int32)
            mind_indices = col_indices

        # One color at a time along the diagonal
        for c in xrange(ncol):
            starttime = time.time()

            # The order is given by col_indices, which ensures that we work from the
            # reference mag outward
            j = col_indices[c]
            sign = sign_indices[c]
            mind = mind_indices[c]

            print("Working on diagonal for color %d" % (j))

            col = galcolor[:, j]
            col_err = galcolor_err[:, j]

            # Need to go through the _ndarray because ztag and zstag are strings
            cvals = np.zeros(self.pars._ndarray[self.ztag[j]].size)
            svals = np.zeros(self.pars._ndarray[self.zstag[j]].size)
            scvals = np.zeros(self.pars.covmat_z.size) + 0.05
            photo_err = np.zeros_like(cvals)

            # Calculate median truncation
            spl = CubicSpline(self.pars.pivotmag_z, self.pars.medcol[:, j])
            med = spl(gals.z)
            spl = CubicSpline(self.pars.pivotmag_z, self.pars.medcol_width[:, j])
            sc = spl(gals.z)

            # What is the maximum scatter in each node?
            # This is based on the median fit, which does not include photometric
            # error, and should always be larger.  This helps regularize the edges
            # where things otherwise can run away.
            scatter_max = spl(self.pars.covmat_z)

            u, = np.where((galcolor[:, j] > (med - self.config.calib_color_nsig * sc)) &
                          (galcolor[:, j] < (med + self.config.calib_color_nsig * sc)))
            trunc = self.config.calib_color_nsig * sc[u]

            dmags = gals.refmag - pivotmags

            # And the starting values...
            # Note that this returns the slope values (svals) at the nodes from the cvals
            # but these might not be the same nodes, so we have to approximate
            cvals_temp, svals_temp, _ = self._compute_startvals(self.pars._ndarray[self.ztag[j]],
                                                                gals.z[u], col[u],
                                                                xval=dmags[u],
                                                                fit=True, mincomp=5)
            cvals[:] = cvals_temp
            inds = np.searchsorted(self.pars._ndarray[self.ztag[j]],
                                   self.pars._ndarray[self.zstag[j]])
            svals[:] = svals_temp[inds]


            # And do the luptitude correction if necessary.
            if self.do_lupcorr:
                lupcorr = self._compute_single_lupcorr(j, cvals, svals, gals, dmags, mags, lups, mind, sign)
            else:
                lupcorr = np.zeros(gals.size)

            # We fit in stages: first the mean, then the slope, then the scatter,
            # and finally all three
            rsfitter = RedSequenceFitter(self.pars._ndarray[self.ztag[j]],
                                         gals.z[u], col[u], col_err[u],
                                         dmags=dmags[u],
                                         trunc=trunc,
                                         slope_nodes=self.pars._ndarray[self.zstag[j]],
                                         scatter_nodes=self.pars.covmat_z,
                                         lupcorrs=lupcorr[u],
                                         probs=probs[u],
                                         bkgs=self.bkg.lookup_diagonal(j, col[u], gals.refmag[u], doRaise=doRaise),
                                         scatter_max=scatter_max, use_scatter_prior=True)

            # fit the mean
            cvals, = rsfitter.fit(cvals, svals, scvals, fit_mean=True)
            # Update the lupcorr...
            if self.do_lupcorr:
                rsfitter._lupcorrs[:] = self._compute_single_lupcorr(j, cvals, svals, gals, dmags, mags, lups, mind, sign)[u]
            # fit the slope
            svals, = rsfitter.fit(cvals, svals, scvals, fit_slope=True)
            # fit the scatter
            scvals, = rsfitter.fit(cvals, svals, scvals, fit_scatter=True)
            # fit combined
            cvals, svals, scvals = rsfitter.fit(cvals, svals, scvals,
                                                fit_mean=True, fit_slope=True, fit_scatter=True)
            # Re-fit...
            #cvals, svals, scvals = rsfitter.fit(cvals, svals, scvals,
            #                                    fit_mean=True, fit_slope=True, fit_scatter=True)


            # And record in the parameters
            self.pars._ndarray[self.ctag[j]] = cvals
            self.pars._ndarray[self.stag[j]] = svals
            self.pars.sigma[j, j, :] = scvals
            self.pars.covmat_amp[j, j, :] = scvals ** 2.

            # And print the time taken
            print('Done in %.2f seconds.' % (time.time() - starttime))

    def _calc_offdiagonal_pars(self, gals, doRaise=True):
        """
        """
        # The routine to compute the off-diagonal elements

        ncol = self.config.nmag - 1

        galcolor = gals.galcol
        galcolor_err = gals.galcol_err

        # compute the pivot mags
        spl = CubicSpline(self.pars.pivotmag_z, self.pars.pivotmag)
        pivotmags = spl(gals.z)

        # And set the right probabilities
        if self.config.calib_use_pcol:
            probs = gals.pcol
        else:
            probs = gals.p

        # Compute c, slope, and median and width for all galaxies/colors
        ci = np.zeros((gals.size, ncol))
        si = np.zeros_like(ci)
        medci = np.zeros_like(ci)
        medwidthi = np.zeros_like(ci)
        gsig = np.zeros_like(ci)

        for j in xrange(ncol):
            spl = CubicSpline(self.pars._ndarray[self.ztag[j]],
                              self.pars._ndarray[self.ctag[j]])
            ci[:, j] = spl(gals.z)
            spl = CubicSpline(self.pars._ndarray[self.zstag[j]],
                              self.pars._ndarray[self.stag[j]])
            si[:, j] = spl(gals.z)
            spl = CubicSpline(self.pars.pivotmag_z, self.pars.medcol[:, j])
            medci[:, j] = spl(gals.z)
            spl = CubicSpline(self.pars.pivotmag_z, self.pars.medcol_width[:, j])
            medwidthi[:, j] = spl(gals.z)
            spl = CubicSpline(self.pars.covmat_z, self.pars.sigma[j, j, :])
            gsig[:, j] = spl(gals.z)

        if self.do_lupcorr:
            mags = np.zeros((gals.size, self.config.nmag))
            lups = np.zeros_like(mags)

            mags[:, self.config.ref_ind] = gals.refmag

            for j in xrange(self.config.ref_ind + 1, self.config.nmag):
                mags[:, j] = mags[:, j - 1] - (ci[:, j - 1] + si[:, j - 1] * (gals.refmag - pivotmags))
            for j in xrange(self.config.ref_ind - 1, -1, -1):
                mags[:, j] = mags[:, j + 1] + (ci[:, j] + si[:, j] * (gals.refmag - pivotmags))

            for j in xrange(self.config.nmag):
                flux = 10.**((mags[:, j] - self.lupzp) / (-2.5))
                lups[:, j] = 2.5 * np.log10(1.0 / self.config.b[j]) - np.arcsinh(0.5 * flux / self.bnmgy[j]) / (0.5 * np.log(10.0))

            magcol = mags[:, :-1] - mags[:, 1:]
            lupcol = lups[:, :-1] - lups[:, 1:]

            lupcorr = lupcol - magcol
        else:
            lupcorr = np.zeros((gals.size, ncol))

        template_col = np.zeros((gals.size, ncol))
        for j in xrange(ncol):
            template_col[:, j] = ci[:, j] + si[:, j] * (gals.refmag - pivotmags) + lupcorr[:, j]

        res = galcolor - template_col

        # figure out order with a ranking based on the configured order
        bits = 2**np.arange(ncol, dtype=np.int32)
        covmat_rank = np.zeros((ncol * ncol - ncol) // 2, dtype=np.int32)
        covmat_order = self.config.calib_color_order
        ctr = 0
        for j in xrange(ncol):
            for k in xrange(j + 1, ncol):
                covmat_rank[ctr] = bits[covmat_order[j]] + bits[covmat_order[k]]
                ctr += 1

        covmat_rank = np.sort(covmat_rank)

        full_covmats = self.pars.covmat_amp.copy()

        for ctr in xrange(covmat_rank.size):
            starttime = time.time()

            j = -1
            k = -1
            for tctr in xrange(ncol):
                if ((covmat_rank[ctr] & bits[tctr]) > 0):
                    if j < 0:
                        j = covmat_order[tctr]
                    else:
                        k = covmat_order[tctr]

            # swap if necessary
            if k < j:
                temp = k
                k = j
                j = temp

            print("Working on %d, %d" % (j, k))

            u, = np.where((galcolor[:, j] > medci[:, j] - self.config.calib_color_nsig * medwidthi[:, j]) &
                          (galcolor[:, j] < medci[:, j] + self.config.calib_color_nsig * medwidthi[:, j]) &
                          (galcolor[:, k] > medci[:, k] - self.config.calib_color_nsig * medwidthi[:, k]) &
                          (galcolor[:, k] < medci[:, k] + self.config.calib_color_nsig * medwidthi[:, k]))

            bvals = self.bkg.lookup_offdiag(j, k, galcolor[:, j], galcolor[:, k], gals.refmag, doRaise=doRaise)

            odfitter = RedSequenceOffDiagonalFitter(self.pars.covmat_z,
                                                    gals.z[u],
                                                    res[u, j], res[u, k],
                                                    gsig[u, j], gsig[u, k],
                                                    gals.mag_err[u, :],
                                                    j, k,
                                                    probs[u],
                                                    bvals[u],
                                                    self.config.calib_covmat_prior,
                                                    min_eigenvalue=self.config.calib_covmat_min_eigenvalue)

            rvals = odfitter.fit(np.zeros(self.pars.covmat_z.size), full_covmats=full_covmats)

            self.pars.sigma[j, k, :] = rvals
            self.pars.sigma[k, j, :] = rvals

            self.pars.covmat_amp[j, k, :] = rvals * self.pars.sigma[j, j, :] * self.pars.sigma[k, k, :]
            self.pars.covmat_amp[k, j, :] = self.pars.covmat_amp[j, k, :]

            full_covmats[j, k, :] = self.pars.covmat_amp[j, k, :]
            full_covmats[k, j, :] = self.pars.covmat_amp[k, j, :]

            print("Done in %.2f seconds." % (time.time() - starttime))

    def _calc_volume_factor(self, zref):
        """
        """

        dz = 0.01

        self.pars.volume_factor = ((self.config.cosmo.Dl(0.0, zref + dz) / (1. + (zref + dz)) -
                                   self.config.cosmo.Dl(0.0, zref) / (1. + zref)) /
                                   (self.config.cosmo.Dl(0.0, self.pars.volume_factor_z + dz) / (1. + (self.pars.volume_factor_z + dz)) -
                                    self.config.cosmo.Dl(0.0, self.pars.volume_factor_z) / (1. + self.pars.volume_factor_z)))


    def save_pars(self, filename, clobber=False):
        """
        """

        hdr = fitsio.FITSHDR()
        hdr['NCOL'] = self.config.nmag - 1
        hdr['MSTARSUR'] = self.config.mstar_survey
        hdr['MSTARBAN'] = self.config.mstar_band
        hdr['LIMMAG'] = self.config.limmag_catalog
        # Saved with arbitrary cushion that seems to work well
        hdr['ZRANGE0'] = np.clip(self.config.zrange[0] - 0.07, 0.01, None)
        hdr['ZRANGE1'] = self.config.zrange[1] + 0.07
        hdr['ALPHA'] = self.config.calib_lumfunc_alpha
        hdr['ZBINFINE'] = self.config.zredc_binsize_fine
        hdr['ZBINCOAR'] = self.config.zredc_binsize_coarse
        hdr['LOWZMODE'] = 0
        hdr['REF_IND'] = self.config.ref_ind
        for j, b in enumerate(self.config.b):
            hdr['BVALUE%d' % (j + 1)] = b

        self.pars.to_fits_file(filename, header=hdr, clobber=clobber)

    def _calc_zreds(self, gals, do_correction=True):
        """
        """

        # This is temporary

        zredstr = RedSequenceColorPar(self.config.parfile)

        zredc = ZredColor(zredstr, adaptive=True, do_correction=do_correction)

        gals.add_zred_fields()

        starttime = time.time()
        for i, g in enumerate(gals):
            try:
                zredc.compute_zred(g)
            except:
                pass

        print('Computed zreds in %.2f seconds.' % (time.time() - starttime))

    def _calc_corrections(self, gals, mode2=False):
        """
        """

        # p or pcol
        if self.config.calib_use_pcol:
            probs = gals.pcol
        else:
            probs = gals.p

        # Set a threshold removing 5% worst lkhd outliers
        st = np.argsort(gals.lkhd)
        thresh = gals.lkhd[st[int(0.05 * gals.size)]]

        # This is an arbitrary 2sigma cut...
        guse, = np.where((gals.lkhd > thresh) &
                         (np.abs(gals.z - gals.zred) < 2. * gals.zred_e))

        spl = CubicSpline(self.pars.pivotmag_z, self.pars.pivotmag)
        pivotmags = spl(gals.z)

        w = 1. / (np.exp((thresh - gals.lkhd[guse]) / 0.2) + 1.0)

        # The offset cvals
        cvals = np.zeros(self.pars.corr_z.size)
        # The slope svals
        svals = np.zeros(self.pars.corr_slope_z.size)
        # And the r value to be multiplied by error
        rvals = np.ones(self.pars.corr_slope_z.size)
        # And the background vals
        bkg_cvals = np.zeros(self.pars.corr_slope_z.size)

        cvals[:], _ = self._compute_startvals(self.pars.corr_z, gals.z, gals.z - gals.zred, median=True)

        # Initial guess for bkg_cvals is trickier and not generalizable (sadly)
        for i in xrange(self.pars.corr_slope_z.size):
            if i == 0:
                zlo = self.pars.corr_slope_z[0]
            else:
                zlo = (self.pars.corr_slope_z[i - 1] + self.pars.corr_slope_z[i]) / 2.
            if i == (self.pars.corr_slope_z.size - 1):
                zhi = self.pars.corr_slope_z[i]
            else:
                zhi = (self.pars.corr_slope_z[i] + self.pars.corr_slope_z[i + 1]) / 2.

            if mode2:
                u, = np.where((gals.zred[guse] > zlo) & (gals.zred[guse] < zhi))
            else:
                u, = np.where((gals.z[guse] > zlo) & (gals.z[guse] < zhi))

            if u.size < 3:
                if i > 0:
                    bkg_cvals[i] = bkg_cvals[i - 1]
            else:
                st = np.argsort(probs[guse[u]])
                uu = u[st[0:u.size // 2]]
                bkg_cvals[i] = np.std(gals.z[guse[uu]] - gals.zred[guse[uu]])**2.

        if mode2:
            print("Fitting zred2 corrections...")
            z = gals.zred
        else:
            print("Fitting zred corrections...")
            z = gals.z

        corrfitter = CorrectionFitter(self.pars.corr_z,
                                      z[guse],
                                      gals.z[guse] - gals.zred[guse],
                                      gals.zred_e[guse],
                                      slope_nodes=self.pars.corr_slope_z,
                                      probs=np.clip(probs[guse], None, 0.99),
                                      dmags=gals.refmag[guse] - pivotmags[guse],
                                      ws=w)

        # self.config.calib_corr_nocorrslope
        # first fit the mean
        cvals, = corrfitter.fit(cvals, svals, rvals, bkg_cvals, fit_mean=True)
        # fit the slope (if desired)
        if not self.config.calib_corr_nocorrslope:
            svals, = corrfitter.fit(cvals, svals, rvals, bkg_cvals, fit_slope=True)
        # Fit r
        rvals, = corrfitter.fit(cvals, svals, rvals, bkg_cvals, fit_r=True)
        # Fit bkg
        bkg_cvals, = corrfitter.fit(cvals, svals, rvals, bkg_cvals, fit_bkg=True)

        # Combined fit
        if not self.config.calib_corr_nocorrslope:
            cvals, svals, rvals, bkg_cvals = corrfitter.fit(cvals, svals, rvals, bkg_cvals, fit_mean=True, fit_slope=True, fit_r=True, fit_bkg=True)
        else:
            cvals, rvals, bkg_cvals = corrfitter.fit(cvals, svals, rvals, bkg_cvals, fit_mean=True, fit_r=True, fit_bkg=True)

        # And record the values
        if mode2:
            self.pars.corr2 = cvals
            self.pars.corr2_slope = svals
            self.pars.corr2_r = rvals
        else:
            self.pars.corr = cvals
            self.pars.corr_slope = svals
            self.pars.corr_r = rvals

    def _make_diagnostic_plots(self, gals):
        """
        """

        import matplotlib.pyplot as plt

        # what plots do we want?
        # We want to split this out into different modules?

        # For each color, plot
        #  Color(z)
        #  Slope(z)
        #  scatter(z)
        # And a combined
        #  All off-diagonal r value plots

        zredstr = RedSequenceColorPar(self.config.parfile, zbinsize=0.005)

        for j in xrange(self.config.nmag - 1):
            fig = plt.figure(figsize=(10, 6))
            fig.clf()

            zredstr.plot_redsequence_diag(fig, j, self.config.bands)
            fig.savefig(os.path.join(self.config.outpath, self.config.plotpath,
                                     '%s_%s-%s.png' % (self.config.d.outbase,
                                                       self.config.bands[j], self.config.bands[j + 1])))
            plt.close(fig)

        fig = plt.figure(figsize=(10, 6))
        fig.clf()
        zredstr.plot_redsequence_offdiags(fig, self.config.bands)
        fig.savefig(os.path.join(self.config.outpath, self.config.plotpath,
                                 '%s_offdiags.png' % (self.config.d.outbase)))

        # And two panel plot with
        #  left panel is offset, scatter, outliers as f(z)
        #  Right panel is zred vs z (whichever)
        # We need to do this for both zred and zred2.

        zbinsize = 0.02
        pcut = 0.9
        ntrial = 1000

        mlim = zredstr.mstar(gals.zred) - 2.5 * np.log10(0.2)

        use, = np.where((gals.p > pcut) &
                        (gals.refmag < mlim) &
                        (gals.zred < 2.0))

        ugals = gals[use]

        zbins = np.arange(self.config.zrange[0], self.config.zrange[1], zbinsize)

        dtype = [('ztrue', 'f4'),
                 ('zuse', 'f4'),
                 ('delta', 'f4'),
                 ('delta_err', 'f4'),
                 ('delta_std', 'f4'),
                 ('zuse_e', 'f4'),
                 ('f_out', 'f4')]


        # There are two modes to plot
        for mode in xrange(2):
            if mode == 0:
                zuse = ugals.z
                dzuse = ugals.zred - ugals.z
                zuse_e = ugals.zred_e
                xlabel = r'$z_{\mathrm{true}}$'
                ylabel_left = r'$z_{\mathrm{red}} - z_{\mathrm{true}}$'
                ylabel_right = r'$z_{\mathrm{red}}$'
                xcol = 'ztrue'
                modename = 'zred'
            else:
                zuse = ugals.zred2
                dzuse = ugals.z - ugals.zred2
                zuse_e = ugals.zred2_e
                xlabel = r'$z_{\mathrm{red2}}$'
                ylabel_left = r'$z_{\mathrm{true}} - z_{\mathrm{red2}}$'
                ylabel_right = r'$z_{\mathrm{true}}$'
                xcol = 'zuse'
                modename = 'zred2'

            zstr = np.zeros(zbins.size, dtype=dtype)

            for i, z in enumerate(zbins):
                # Get all the galaxies in the bin
                u1, = np.where((zuse >= z) & (zuse < (z + zbinsize)))

                if u1.size < 3:
                    print('Warning: not enough galaxies in zbin: %.2f, %.2f' % (z, z + zbinsize))
                    continue

                med = np.median(dzuse[u1])
                gsigma = 1.4826 * np.abs(dzuse[u1] - med) / zuse_e[u1]

                u2, = np.where(np.abs(gsigma) < 3.0)
                if u2.size < 3:
                    print('Warning: not enough galaxies in zbin: %.2f, %.2f' % (z, z + zbinsize))

                use = u1[u2]

                zstr['ztrue'][i] = np.median(ugals.z[use])
                zstr['zuse'][i] = np.median(zuse[use])
                zstr['delta'][i] = np.median(dzuse[use])

                barr = np.zeros(ntrial)
                for t in xrange(ntrial):
                    r = np.random.choice(dzuse[use], size=use.size, replace=True)
                    barr[t] = np.median(r)

                # Error on median as determined from bootstrap resampling
                zstr['delta_err'][i] = np.std(barr)

                # The typical error
                zstr['delta_std'][i] = 1.4826 * np.median(np.abs(dzuse[use] - zstr['delta'][i]))

                # And outliers ...
                u4, = np.where(np.abs(dzuse[u1] - zstr['delta'][i]) > 4.0 * zstr['delta_std'][i])
                zstr['f_out'][i] = float(u4.size) / float(u1.size)

                zstr['zuse_e'][i] = np.median(zuse_e[use])

            # Cut out bins that didn't get a fit
            cut, = np.where(zstr['ztrue'] > 0.0)
            zstr = zstr[cut]

            # Now we can make the plots
            fig = plt.figure(figsize=(10, 6))
            fig.clf()

            # Left panel is offset, scatter, etc.
            ax = fig.add_subplot(121)
            ax.errorbar(zstr[xcol], zstr['delta'], yerr=zstr['delta_err'], fmt='k^')
            ax.plot(self.config.zrange, [0.0, 0.0], 'k:')
            ax.plot(zstr[xcol], zstr['delta_std'], 'r-')
            ax.plot(zstr[xcol], zstr['zuse_e'], 'b-')
            ax.plot(zstr[xcol], zstr['f_out'], 'm-')
            ax.set_xlim(self.config.zrange)
            ax.set_ylim(-0.05, 0.05)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel_left)

            ax = fig.add_subplot(122)
            if mode == 0:
                ax.hexbin(ugals.z, ugals.zred, bins='log', extent=[self.config.zrange[0], self.config.zrange[1], self.config.zrange[0], self.config.zrange[1]])
            else:
                ax.hexbin(ugals.zred2, ugals.z, bins='log', extent=[self.config.zrange[0], self.config.zrange[1], self.config.zrange[0], self.config.zrange[1]])
            ax.plot(self.config.zrange, self.config.zrange, 'r--')
            ax.set_xlim(self.config.zrange)
            ax.set_ylim(self.config.zrange)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel_right)

            fig.tight_layout()
            fig.savefig(os.path.join(self.config.outpath, self.config.plotpath,
                                     '%s_%s_plots.png' % (self.config.d.outbase, modename)))

            plt.close(fig)

