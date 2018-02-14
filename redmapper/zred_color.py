from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import numpy as np
import esutil
import scipy.integrate
import scipy.interpolate
import copy

from .galaxy import GalaxyCatalog
from .utilities import interpol

class ZredColor(object):
    """
    """

    def __init__(self, zredstr, sigint=0.001, do_correction=True, adaptive=True,
                 use_chisq=True, use_photoerr=True, zrange=None):
        self.zredstr = zredstr

        self.sigint = sigint
        self.do_correction = do_correction
        self.adaptive = adaptive
        self.use_chisq = use_chisq
        self.use_photoerr = use_photoerr
        self.zrange = zrange

        if (zredstr.z[1] - zredstr.z[0]) >= 0.01:
            # Must turn off adaptive if the stepsize is too large
            self.adaptive = False

    def compute_zred(self, galaxy):
        """
        """

        nz = self.zredstr.z.size
        refmagindex = self.zredstr.refmagindex(galaxy.refmag)

        if self.adaptive:
            step = 2
        else:
            step = 1

        notextrap, = np.where(~self.zredstr.extrapolated)

        if self.zrange is None:
            zbinstart = 0
            zbinstop = nz - 1
        else:
            u, = np.where((self.zredstr.z > self.zrange[0]) &
                          (self.zredstr.z < self.zrange[1]))
            zbinstart = u[0]
            zbinstop = u[-1]

        lndist = np.zeros(nz) - 1e12

        zbins = np.arange(zbinstart, zbinstop + step, step)

        # Mark the bins that are completely out of range
        # This last check makes sure we don't hit the overflow bin
        good = ((galaxy.refmag < self.zredstr.maxrefmag[zbins]) &
                (galaxy.refmag > self.zredstr.minrefmag[zbins]) &
                (self.zredstr.z[zbins] < 100.0))

        lndist[zbins[~good]] = -1e11

        if np.nonzero(good)[0].size > 0:
            # we have at least one good bin
            zbins = zbins[good]
            lndist[zbins] = self._calculate_lndist(galaxy, zbins)
        else:
            self._reset_bad_values(galaxy)
            return

        if self.adaptive:
            # only consider a maximum in the non-extrapolated region
            ind_temp = np.argmax(lndist[notextrap])
            ind = notextrap[ind_temp]

            # go over the nearest neighbors
            neighbors = 5

            minindex = ind - neighbors if ind - neighbors >= 0 else 0
            maxindex = ind + neighbors if ind + neighbors <= nz else nz

            if minindex == 0:
                maxindex = 1 + 2*neighbors
            if maxindex == (nz - 1):
                minindex = nz - 2 - 2 * neighbors

            zbins = np.arange(minindex, maxindex + 1)
            # select out the values that have not been run yet
            #  (these are very negative)
            to_run, = np.where(lndist[zbins] < -1e10)

            if to_run.size > 0:
                zbins = zbins[to_run]
                lndist[zbins] = self._calculate_lndist(galaxy, zbins)

        # move from log space to regular space
        maxlndist = np.max(lndist)
        with np.errstate(invalid='ignore'):
            dist = np.exp(lndist - maxlndist)

        # fix infinities and NaNs
        bad, = np.where(~np.isfinite(dist))
        dist[bad] = 0.0

        # did we hit a boundary?
        good, = np.where(dist > 0.0)
        if good.size >= 2:
            if (dist[good[0]] > 1e-5) or (dist[good[-1]] > 1e-5):
                # we did hit a boundary since dist didn't go to zero
                neighbors = 10

                if dist[good[0]] > 1e-5:
                    minindex = good[0] - neighbors if good[0] - neighbors >= 0 else 0
                    maxindex = good[0] - 1 if good[0] - 1 >= 0 else 0
                else:
                    maxindex = good[-1] + neighbors if good[-1] + neighbors <= nz else nz
                    minindex = good[-1] + 1 if good[-1] + 1 <= nz else nz

                zbins = np.arange(minindex, maxindex + 1)
                to_run, = np.where(lndist[zbins] < -1e10)
                if to_run.size > 0:
                    zbins = zbins[to_run]
                    lndist[zbins] = self._calculate_lndist(galaxy, zbins)

                    with np.errstate(invalid='ignore'):
                        dist[zbins] = np.exp(lndist[zbins] - maxlndist)

                    bad, = np.where(~np.isfinite(dist))
                    dist[bad] = 0.0

        # Now estimate zred and zred_e
        good, = np.where(dist > 0.0)
        if good.size < 2:
            self._reset_bad_values(galaxy)
            return

        # take the maximum where not extrapolated
        ind_temp = np.argmax(dist[notextrap])
        ind = notextrap[ind_temp]

        calcinds_base = np.arange(0, self.zredstr.z.size, step)

        # Go from the peak and include all (every other point) that is > 1e-5.
        l, = np.where((calcinds_base <= ind) & (dist[calcinds_base] > 1e-5))
        h, = np.where((calcinds_base > ind) & (dist[calcinds_base] > 1e-5))

        # if this is a catastrophic failure, kick out and don't crash
        if (l.size == 0 and h.size == 0):
            self._reset_bad_values(galaxy)
            return

        if l.size > 0:
            lcut, = np.where((l - np.roll(l, 1)) > 1)
            if lcut.size > 0:
                l = l[lcut[0]:]
        if h.size > 0:
            hcut, = np.where((h - np.roll(h, 1)) > 1)
            if hcut.size > 0:
                h = h[0:hcut[0]]

        if l.size == 0:
            calcinds = calcinds_base[h]
        elif h.size == 0:
            calcinds = calcinds_base[l]
        else:
            calcinds = np.concatenate((calcinds_base[l], calcinds_base[h]))

        if calcinds.size >= 3:
            tdist = scipy.integrate.trapz(dist[calcinds], self.zredstr.z[calcinds])
            zred_temp = scipy.integrate.trapz(dist[calcinds] * self.zredstr.z[calcinds],
                                              self.zredstr.z[calcinds]) / tdist
            zred_e = scipy.integrate.trapz(dist[calcinds] * self.zredstr.z[calcinds]**2.,
                                           self.zredstr.z[calcinds]) / tdist - zred_temp**2.
        else:
            tdist = np.sum(dist[calcinds])
            zred_temp = np.sum(dist[calcinds] * self.zredstr.z[calcinds]) / tdist
            zred_e = np.sum(dist[calcinds] * self.zredstr.z[calcinds]**2.) / tdist - zred_temp**2.

        if zred_e < 0.0:
            zred_e = 1.0
        else:
            zred_e = np.sqrt(zred_e)

        zred_e = zred_e if zred_e > 0.005 else 0.005

        # Now fit a parabola to get the perfect zred
        ind_temp = np.argmax(dist[notextrap])
        ind = notextrap[ind_temp]

        zred = zred_temp.copy()

        neighbors = 2
        use, = np.where(lndist > -1e10)
        if use.size >= neighbors * 2 + 1:
            minuse = use.min()
            maxuse = use.max()

            # Will need to check all this...
            minindex = minuse if minuse > ind - neighbors else ind - neighbors
            maxindex = maxuse if maxuse < ind + neighbors else ind + neighbors
            if minindex == minuse:
                maxindex = np.clip(minuse + 2 * neighbors, None, maxuse)
            elif maxindex == maxuse:
                minindex = np.clip(maxuse - 1 - 2*neighbors, minuse, None)

            if ((maxindex - minindex + 1) >= 5):
                fit = np.polyfit(self.zredstr.z[minindex:maxindex + 1],
                                 lndist[minindex:maxindex + 1], 2)
                if fit[0] < 0.0:
                    ztry = -fit[1] / (2.0 * fit[0])
                    # Don't let it move to far, or it's a bad fit
                    if (np.abs(ztry - zred) < 2.0*zred_e):
                        zred = ztry

        # And compute values at the real zred peak
        x = (self.zredstr.z - zred) / zred_e
        newdist = np.exp(-0.5 * x * x)

        bad, = np.where((lndist < -1e10) | (~np.isfinite(lndist)))
        newdist[bad] = 0.0
        lndist[bad] = -1e11

        if calcinds.size >= 3:
            # Note there maybe should be a distcorr here, but this is not
            #  actually computed in the IDL code (bug?)
            lkhd = scipy.integrate.trapz(newdist[calcinds] * (lndist[calcinds]), self.zredstr.z[calcinds]) / scipy.integrate.trapz(newdist[calcinds], self.zredstr.z[calcinds])
        else:
            lkhd = np.sum(newdist[calcinds] * lndist[calcinds]) / np.sum(newdist[calcinds])

        # Compute chisq at the closest bin position
        zbin = np.argmin(np.abs(zred - self.zredstr.z))
        chisq = self.zredstr.calculate_chisq(galaxy, np.array([zbin, zbin]), z_is_index=True, calc_lkhd=(not self.use_chisq))[0]

        if not np.isfinite(lkhd):
            self._reset_bad_values(galaxy)
            return

        # And apply the corrections
        zred2 = np.zeros(1) + zred
        zred2_e = np.zeros(1) + zred_e
        zred_uncorr = np.zeros(1) + zred
        zred_uncorr_e = np.zeros(1) + zred_e

        if self.do_correction:
            olddz = -1.0
            dz = 0.0
            iteration = 0

            #pivotmag = self.zredstr.pivot_func(zred)
            pivotmag = interpol(self.zredstr.pivotmag, self.zredstr.z, zred)

            while np.abs(olddz - dz) > 1e-3 and iteration < 10:
                olddz = copy.copy(dz)
                #dz = self.zredstr.corr_func(zred + olddz) + (galaxy.refmag - pivotmag) * self.zredstr.corr_slope_func(zred + olddz)
                dz = interpol(self.zredstr.corr, self.zredstr.z, zred + olddz) + (galaxy.refmag - pivotmag) * interpol(self.zredstr.corr_slope, self.zredstr.z, zred + olddz)
                iteration += 1

            zred = zred + dz

            # evaluate error correction at "z_true"
            #zred_e *= self.zredstr.corr_r_func(zred)
            zred_e *= interpol(self.zredstr.corr_r, self.zredstr.z, zred)

            # And the zred2 correction
            #dz = self.zredstr.corr2_func(zred2) + (galaxy.refmag - pivotmag) * self.zredstr.corr2_slope_func(zred2)
            dz = interpol(self.zredstr.corr2, self.zredstr.z, zred2) + (galaxy.refmag - pivotmag) * (interpol(self.zredstr.corr2_slope, self.zredstr.z, zred2))
            # this is evaluated at zred0
            #r2 = self.zredstr.corr2_r_func(zred2)
            r2 = interpol(self.zredstr.corr2_r, self.zredstr.z, zred2)

            zred2 += dz
            zred2_e *= r2

        # Finally store the values

        galaxy.zred = zred
        galaxy.zred_e = zred_e
        galaxy.zred2 = zred2
        galaxy.zred2_e = zred2_e
        galaxy.zred_uncorr = zred_uncorr
        galaxy.zred_uncorr_e = zred_uncorr_e
        galaxy.chisq = chisq
        galaxy.lkhd = lkhd

        # and we're done

    def _calculate_lndist(self, galaxy, zbins):
        """
        """

        # Note we need to deal with photoerr...
        if zbins.size > 1:
            # we have many bins...
            lndist = self.zredstr.calculate_chisq(galaxy, zbins, z_is_index=True, calc_lkhd=(not self.use_chisq))
        else:
            # we have a single bin... hack this
            lndist = self.zredstr.calculate_chisq(galaxy, np.array([zbins, zbins]), z_is_index=True, calc_lkhd=(not self.use_chisq))[0]

        if self.use_chisq:
            lndist *= -0.5

        #with np.errstate(invalid='ignore'):
        lndistcorr = np.log((10.**(0.4 * (self.zredstr.alpha + 1.0) *
                                   (self.zredstr._mstar[zbins] - galaxy.refmag)) *
                             np.exp(-10.**(0.4 * (self.zredstr._mstar[zbins] - galaxy.refmag)))) *
                            self.zredstr.volume_factor[zbins])

        lndist += lndistcorr

        bad, = np.where(~np.isfinite(lndist))
        lndist[bad] = -1e11

        return lndist

    def _reset_bad_values(self, galaxy):
        """
        """

        galaxy.lkhd = -1000.0
        galaxy.zred = -1.0
        galaxy.zred_e = -1.0
        galaxy.zred2 = -1.0
        galaxy.zred2_e = -1.0
        galaxy.zred_uncorr = -1.0
        galaxy.zred_uncorr_e = -1.0
        galaxy.chisq = -1.0

