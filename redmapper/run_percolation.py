from __future__ import print_function

import fitsio
import numpy as np
import esutil
import copy

from cluster import ClusterCatalog
from background import Background
from mask import HPMask
from galaxy import GalaxyCatalog
from cluster import Cluster
from cluster import ClusterCatalog
from depthmap import DepthMap
from zlambda import Zlambda
from zlambda import ZlambdaCorrectionPar
from cluster_runner import ClusterRunner

class RunPercolation(ClusterRunner):
    """
    """

    def _additional_intialization(self, **kwargs):
        self.runmode = 'percolation'
        self.read_zreds = True
        self.zreds_required = True
        self.filetype = 'final'

    def _more_setup(self, *args, **kwargs):

        # read in the catalog...
        self.cat = ClusterCatalog.from_catfile(self.config.catfile,
                                               zredstr=self.zredstr,
                                               config=self.config,
                                               bkg=self.bkg,
                                               cosmo=self.cosmo)

        keepz = kwargs.pop('keepz', False)
        keepid = kwargs.pop('keepid', False)
        specseed = kwargs.pop('specseed', False)

        zrange=copy.copy(self.config.zrange)
        if keepz:
            zrange[0] -= self.config.calib_zrange_cushion
            zrange[0] = zrange[0] if zrange[0] > 0.05 else 0.05
            zrange[1] += self.config.calib_zrange_cushion

        use, = np.where((self.cat.z > zrange[0]) &
                        (self.cat.z < zrange[1]) &
                        (np.isfinite(self.cat.lnlike)) &
                        (self.cat.Lambda > self.config.percolation_minlambda))

        # How to bail if use.size == 0?  Need a framework for fail...

        if keepid:
            st = np.argsort(self.cat.mem_match_id[use])
        else:
            # Reverse sort by total likelihood
            st = np.argsort(self.cat.lnlike[use])[::-1]

        self.cat = self.cat[use[st]]

        self.cat.ra_orig = self.cat.ra
        self.cat.dec_orig = self.cat.dec

        self.do_percolation_masking = True
        self.do_lam_plusminus = True
        self.record_members = True

        self.use_memradius = False
        if self.config.percolation_memradius > 1.0:
            self.use_memradius = True

        self.use_memlum = False
        #self.limlum = np.clip(self.config.lval_reference - 0.1, 0.01, None)
        if (self.config.lval_reference > 0.1):
            self.limlum = self.limlum if self.limlum > 0.1 else 0.1

        if (self.config.percolation_memlum > 0.0 and
            self.config.percolation_memlum < self.config.lval_reference):
            if self.config.percolation_memlum < self.limlum:
                self.limlum = self.config.percolation_memlum

        if self.config.percolation_lmask > 0.0:
            if self.config.percolation_lmask < self.limlum:
                self.limlum = self.config.percolation_lmask

    def _process_cluster(self, cluster):
        bad = False
        iteration = 0
        done = False

        # check if the central galaxy has already been masked.
        minind = np.argmin(cluster.neighbors.r)
        if self.specseed:
            specind = minind

        if cluster.neighbors.pfree[minind] < self.config.percolation_pbcg_cut:
            bad = True
            self._reset_bad_values()
            return

        # calculate lambda (percolated) and update redshift.
        # we don't need error yet or radial masking (will need to modify runner code)

        # in order to do centering, we need to go to 2*rlambda + cushion.
        lc, = np.where(cluster.neighbors.r < 2.05 * self.r0 * (self.Lambda/100.)**self.beta)
        if lc.size < 2:
            bad = True
            self._reset_bad_values()
            return

        lam = cluster.calc_richness(self.mask, index=lc, calc_err=False)

        incut, = np.where((cluster.neighbors.pmem > 0.0) &
                          (cluster.neighbors.r > np.min(cluster.neighbors.r)))

        # check if we're all bad
        if ((cluster.Lambda/cluster.scaleval < self.config.percolation_minlambda) or
            (incut.size < 3)):
            bad = True
            self._reset_bad_values()
            return

        if not self.keepz:
            zlam = Zlambda(cluster)
            z_lambda, z_lambda_e = zlam.calc_zlambda(cluster.redshift, self.mask, calc_err=False, calcpz=False)

            cluster.redshift = z_lambda

        if cluster.redshift < 0.0:
            bad = True
            self._reset_bad_values()
            return

        # GET CENTERING STRUCTURE HERE
        # AND THE CENTERING STRUCTURE WILL SET THE CLUSTER PROPERTIES

        if cent.ncent == 0:
            bad = True
            self._reset_bad_values()
            return

        #cluster.ncent = cent.ncent
        #cluster.ncent_good = cent.ncent_good

        #cluster.ra = cent.ra[0]
        #cluster.dec = cent.dec[0]

        #if cent.maxind >= 0:
            # we have centered on a galaxy, and need to update stats.
        #    cluster.mag[:] = cluster.neighbors.mag[cent.maxind, :]
        #    cluster.mag_err[:] = cluster.neighbors.mag[cent.maxind, :]
        #    cluster.refmag = cluster.neighbors.refmag[cent.maxind]
        #    cluster.refmag_err = cluster.neighbors.refmag_err[cent.maxind]
        #    cluster.zred = cluster.neighbors.zred[cent.maxind]
        #    cluster.zred_e = cluster.neighbors.zred[cent.maxind]
        #    cluster.zred_chisq = cluster.neighbors.zred_chisq[cent.maxind]

        # Since we changed the center, we need new "dist" values!
        #  (this uses the new ra/dec)
        cluster.update_neighbors_dist()

        for i in xrange(self.config.percolation_niter):
            if cluster.redshift < 0.0:
                bad = True
            if bad:
                self._reset_bad_values()
                return

            if i == 0:
                # update the mask info...
                self.mask.set_radmask(cluster, cluster.mpc_scale)

                self.depthstr.calc_maskdepth(self.mask.maskgals,
                                             cluster.ra, cluster.dec, cluster.mpc_scale)

            rmask = self.rmask_0 * (self.Lambda/100.)**self.rmask_beta * ((1. + cluster.redshift) / (1. + self.rmask_zpivot))**self.rmask_gamma

            if rmask < cluster.r_lambda:
                rmask = cluster.r_lambda

            if i == (self.config.percolation_niter - 1):
                # this is the last iteration, make sure we go out to the mask radius
                lc, = np.where(cluster.r < 1.1 * rmask)
            else:
                # previous iteration -- leave cushion for getting bigger
                lc, = np.where(cluster.r < 2.0 * rlambda)

            if lc.size < 2:
                bad = True
                continue

            lam = cluster.calc_richness(self.mask)

            if (((cluster.Lambda/cluster.scaleval) < self.config.percolation_minlambda) or
                (cluster.neighbors.pfree[cent.maxind] < self.config.percolation_pbcg_cut)):
                bad = True
                continue

            if i == 0:
                # Only on the first iteration -- with new center -- is this necessary
                pass
