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
                 use_photoerr=True, zrange=None):
        self.zredstr = zredstr

        self.sigint = sigint
        self.do_correction = do_correction
        self.adaptive = adaptive
        self.use_photoerr = use_photoerr
        self.zrange = zrange

        self.nz = self.zredstr.z.size - 1
        self.notextrap, = np.where(~self.zredstr.extrapolated)

        if self.zrange is None:
            self.zbinstart = 0
            self.zbinstop = self.nz - 1
        else:
            u, = np.where((self.zredstr.z > self.zrange[0]) &
                          (self.zredstr.z < self.zrange[1]))
            self.zbinstart = u[0]
            self.zbinstop = u[-1]

        if (zredstr.z[1] - zredstr.z[0]) >= 0.01:
            # Must turn off adaptive if the stepsize is too large
            self.adaptive = False

    def compute_zreds(self, galaxies):
        """
        """

        for galaxy in galaxies:
            self.compute_zred(galaxy, no_corrections=True)

        if self.do_correction:
            # Bulk processing
            olddzs = np.zeros(galaxies.size)
            dzs = np.zeros_like(olddzs)
            iteration = 0

            pivotmags = interpol(self.zredstr.pivotmag, self.zredstr.z, galaxies.zred_uncorr)

            while (iteration < 5):
                olddzs[:] = dzs
                dzs[:] = interpol(self.zredstr.corr, self.zredstr.z, galaxies.zred_uncorr + olddzs) + (galaxies.refmag - pivotmags) * interpol(self.zredstr.corr_slope, self.zredstr.z, galaxies.zred_uncorr + olddzs)
                iteration += 1

            galaxies.zred = galaxies.zred_uncorr + dzs
            galaxies.zred_e = galaxies.zred_uncorr_e * interpol(self.zredstr.corr_r, self.zredstr.z, galaxies.zred)

            dz2s = interpol(self.zredstr.corr2, self.zredstr.z, galaxies.zred_uncorr)
            r2s = interpol(self.zredstr.corr2_r, self.zredstr.z, galaxies.zred_uncorr)

            galaxies.zred2 = galaxies.zred_uncorr + dz2s
            galaxies.zred2_e = galaxies.zred_uncorr_e * r2s


    def compute_zred(self, galaxy, no_corrections=False):
        """
        """

        if self.adaptive:
            step = 2
        else:
            step = 1

        lndist = np.zeros(self.nz) - 1e12
        chisq = np.zeros(self.nz) + 1e12

        zbins = np.arange(self.zbinstart, self.zbinstop, step)

        # Mark the bins that are completely out of range
        # This last check makes sure we don't hit the overflow bin
        good = ((galaxy.refmag < self.zredstr.maxrefmag[zbins]) &
                (galaxy.refmag > self.zredstr.minrefmag[zbins]) &
                (self.zredstr.z[zbins] < 100.0))

        lndist[zbins[~good]] = -1e11

        if np.nonzero(good)[0].size > 0:
            # we have at least one good bin
            zbins = zbins[good]
            lndist[zbins], chisq[zbins] = self._calculate_lndist(galaxy, zbins)
        else:
            self._reset_bad_values(galaxy)
            return

        if self.adaptive:
            # only consider a maximum in the non-extrapolated region
            ind_temp = np.argmax(lndist[self.notextrap])
            ind = self.notextrap[ind_temp]

            # go over the nearest neighbors
            neighbors = 5

            minindex = ind - neighbors if ind - neighbors >= 0 else 0
            maxindex = ind + neighbors if ind + neighbors <= self.nz else self.nz

            if minindex == 0:
                maxindex = 1 + 2*neighbors
            if maxindex == (self.nz - 1):
                minindex = self.nz - 2 - 2 * neighbors

            zbins = np.arange(minindex, maxindex + 1)
            # select out the values that have not been run yet
            #  (these are very negative)
            to_run, = np.where(lndist[zbins] < -1e10)

            if to_run.size > 0:
                zbins = zbins[to_run]
                lndist[zbins], chisq[zbins] = self._calculate_lndist(galaxy, zbins)

        # move from log space to regular space
        maxlndist = np.max(lndist)
        with np.errstate(invalid='ignore', over='ignore'):
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
                    maxindex = good[-1] + neighbors if good[-1] + neighbors < self.nz else self.nz-1
                    minindex = good[-1] + 1 if good[-1] + 1 < self.nz else self.nz-1

                zbins = np.arange(minindex, maxindex + 1)
                to_run, = np.where(lndist[zbins] < -1e10)
                if to_run.size > 0:
                    zbins = zbins[to_run]
                    lndist[zbins], chisq[zbins] = self._calculate_lndist(galaxy, zbins)

                    with np.errstate(invalid='ignore', over='ignore'):
                        dist[zbins] = np.exp(lndist[zbins] - maxlndist)

                    bad, = np.where(~np.isfinite(dist))
                    dist[bad] = 0.0

        # Now estimate zred and zred_e
        good, = np.where(dist > 0.0)
        if good.size < 2:
            self._reset_bad_values(galaxy)
            return

        # take the maximum where not extrapolated
        ind_temp = np.argmax(dist[self.notextrap])
        ind = self.notextrap[ind_temp]

        # This needs a -1 because the top redshift in zredstr is a high-end filler
        calcinds_base = np.arange(0, self.nz, step)

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
        ind_temp = np.argmax(dist[self.notextrap])
        ind = self.notextrap[ind_temp]

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
                X = np.zeros((maxindex - minindex + 1, 3))
                X[:, 1] = self.zredstr.z[minindex:maxindex + 1]
                X[:, 0] = X[:, 1] * X[:, 1]
                X[:, 2] = 1
                y = lndist[minindex: maxindex + 1]

                fit = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)

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

        # Get chisq at the closest bin position
        zbin = np.argmin(np.abs(zred - self.zredstr.z))
        #chisq = self.zredstr.calculate_chisq(galaxy, np.array([zbin, zbin]), z_is_index=True, calc_lkhd=(not self.use_chisq))[0]
        chisq = chisq[zbin]

        if not np.isfinite(lkhd):
            self._reset_bad_values(galaxy)
            return

        # And apply the corrections
        zred2 = np.zeros(1) + zred
        zred2_e = np.zeros(1) + zred_e
        zred_uncorr = np.zeros(1) + zred
        zred_uncorr_e = np.zeros(1) + zred_e

        if self.do_correction and not no_corrections:
            olddz = -1.0
            dz = 0.0
            iteration = 0

            pivotmag = interpol(self.zredstr.pivotmag, self.zredstr.z, zred)

            while np.abs(olddz - dz) > 1e-3 and iteration < 10:
                olddz = copy.copy(dz)
                dz = interpol(self.zredstr.corr, self.zredstr.z, zred + olddz) + (galaxy.refmag - pivotmag) * interpol(self.zredstr.corr_slope, self.zredstr.z, zred + olddz)
                iteration += 1

            zred = zred + dz

            # evaluate error correction at "z_true"
            zred_e *= interpol(self.zredstr.corr_r, self.zredstr.z, zred)

            # And the zred2 correction
            dz = interpol(self.zredstr.corr2, self.zredstr.z, zred2) + (galaxy.refmag - pivotmag) * (interpol(self.zredstr.corr2_slope, self.zredstr.z, zred2))
            # this is evaluated at zred0
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
            chisq = self.zredstr.calculate_chisq_redshifts(galaxy, zbins, z_is_index=True, calc_lkhd=False)
        else:
            # we have a single bin... hack this
            chisq = self.zredstr.calculate_chisq(galaxy, np.array([zbins[0], zbins[0]]), z_is_index=True, calc_lkhd=False)[0]

        lndist = -0.5 * chisq

        #with np.errstate(invalid='ignore'):
        lndistcorr = np.log((10.**(0.4 * (self.zredstr.alpha + 1.0) *
                                   (self.zredstr._mstar[zbins] - galaxy.refmag)) *
                             np.exp(-10.**(0.4 * (self.zredstr._mstar[zbins] - galaxy.refmag)))) *
                            self.zredstr.volume_factor[zbins])

        lndist += lndistcorr

        bad, = np.where(~np.isfinite(lndist))
        lndist[bad] = -1e11

        return (lndist, chisq)

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

