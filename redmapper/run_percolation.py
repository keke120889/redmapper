from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import fitsio
import numpy as np
import esutil
import copy
import sys

from .cluster import ClusterCatalog
from .background import Background
from .mask import HPMask
from .galaxy import GalaxyCatalog
from .cluster import Cluster
from .cluster import ClusterCatalog
from .depthmap import DepthMap
from .zlambda import Zlambda
from .zlambda import ZlambdaCorrectionPar
from .cluster_runner import ClusterRunner
from .centering import CenteringBCG, CenteringWcenZred

###################################################
# Order of operations:
#  __init__()
#    _additional_initialization() [override this]
#  run()
#    _setup()
#    _more_setup() [override this]
#    _process_cluster() [override this]
#    _postprocess() [override this]
#  output()
###################################################

class RunPercolation(ClusterRunner):
    """
    """

    def _additional_initialization(self, **kwargs):
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
                                               cosmo=self.cosmo,
                                               r0=self.r0,
                                               beta=self.beta)

        self.keepz = kwargs.pop('keepz', False)
        self.keepid = kwargs.pop('keepid', False)
        self.specseed = kwargs.pop('specseed', False)

        zrange=copy.copy(self.config.zrange)
        if self.keepz:
            zrange[0] -= self.config.calib_zrange_cushion
            zrange[0] = zrange[0] if zrange[0] > 0.05 else 0.05
            zrange[1] += self.config.calib_zrange_cushion

        use, = np.where((self.cat.z > zrange[0]) &
                        (self.cat.z < zrange[1]) &
                        (np.isfinite(self.cat.lnlike)) &
                        (self.cat.Lambda > self.config.percolation_minlambda))

        # How to bail if use.size == 0?  Need a framework for fail...

        if self.keepid:
            st = np.argsort(self.cat.mem_match_id[use])
        else:
            # Reverse sort by total likelihood
            st = np.argsort(self.cat.lnlike[use])[::-1]
            self.cat.mem_match_id[:] = 0

        self.cat = self.cat[use[st]]

        # This preserves previously set ids
        self._generate_mem_match_ids()

        self.cat.ra_orig = self.cat.ra
        self.cat.dec_orig = self.cat.dec

        self.do_percolation_masking = True
        self.do_lam_plusminus = True
        self.record_members = True
        self.do_correct_zlambda = True
        self.do_pz = True

        self.use_memradius = False
        if self.config.percolation_memradius > 1.0:
            self.use_memradius = True

        self.use_memlum = False
        if (self.config.lval_reference > 0.1):
            self.limlum = self.limlum if self.limlum > 0.1 else 0.1

        if (self.config.percolation_memlum > 0.0 and
            self.config.percolation_memlum < self.config.lval_reference):
            if self.config.percolation_memlum < self.limlum:
                self.limlum = self.config.percolation_memlum

        if self.config.percolation_lmask > 0.0:
            if self.config.percolation_lmask < self.limlum:
                self.limlum = self.config.percolation_lmask

        self.maxiter = self.config.percolation_niter
        self.min_lambda = self.config.percolation_minlambda

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
            self._reset_bad_values(cluster)
            return bad

        # calculate lambda (percolated) and update redshift.
        # we don't need error yet or radial masking (will need to modify runner code)

        # in order to do centering, we need to go to 2*rlambda + cushion.
        lc, = np.where(cluster.neighbors.r < 2.05 * self.r0 * (cluster.Lambda/100.)**self.beta)
        if lc.size < 2:
            bad = True
            self._reset_bad_values(cluster)
            return bad

        lam = cluster.calc_richness(self.mask, index=lc, calc_err=False)

        incut, = np.where((cluster.neighbors.pmem > 0.0) &
                          (cluster.neighbors.r > np.min(cluster.neighbors.r)))

        # check if we're all bad
        if ((cluster.Lambda/cluster.scaleval < self.config.percolation_minlambda) or
            (incut.size < 3)):
            bad = True
            self._reset_bad_values(cluster)
            return bad

        if not self.keepz:
            zlam = Zlambda(cluster)
            z_lambda, z_lambda_e = zlam.calc_zlambda(cluster.redshift, self.mask, calc_err=False, calcpz=False)

            cluster.redshift = z_lambda

        if cluster.redshift < 0.0:
            bad = True
            self._reset_bad_values(cluster)
            return bad

        # Grab the correct centering class here
        cent = reduce(getattr, self.config.centerclass.split('.'), sys.modules[__name__])(cluster)
        if not cent.find_center() or cent.ngood==0:
            bad = True
            self._reset_bad_values(cluster)
            return bad

        # Record the centering values
        # update the cluster center!
        cluster.ra = cent.ra[0]
        cluster.dec = cent.dec[0]
        cluster.update_neighbors_dist()

        # only update central galaxy values if we centered on a galaxy
        #  (this is typical, but not required for a centering module)
        if cent.index[0] >= 0:
            # check order of index
            cluster.mag[:] = cluster.neighbors.mag[cent.index[0], :]
            cluster.mag_err[:] = cluster.neighbors.mag[cent.index[0], :]
            cluster.refmag = cluster.neighbors.refmag[cent.index[0]]
            cluster.refmag_err = cluster.neighbors.refmag_err[cent.index[0]]
            cluster.ebv_mean = cluster.neighbors.ebv[cent.index[0]]
            if self.did_read_zreds:
                cluster.zred = cluster.neighbors.zred[cent.index[0]]
                cluster.zred_e = cluster.neighbors.zred_e[cent.index[0]]
                cluster.zred_chisq = cluster.neighbors.chisq[cent.index[0]]

            cluster.id_cent[:] = cluster.neighbors.id[cent.index]

        # And update the center info...
        cluster.ncent_good = cent.ngood
        cluster.ra_cent[:] = cent.ra
        cluster.dec_cent[:] = cent.dec
        cluster.p_cen[:] = cent.p_cen
        cluster.q_cen[:] = cent.q_cen
        cluster.p_fg[:] = cent.p_fg
        cluster.q_miss = cent.q_miss
        cluster.p_sat[:] = cent.p_sat
        cluster.p_c[:] = cent.p_c

        # now we iterate over the new center to get the redshift

        for i in xrange(self.maxiter):
            if cluster.redshift < 0.0:
                bad = True
            if bad:
                self._reset_bad_values(cluster)
                return bad

            if i == 0:
                # update the mask info...
                self.mask.set_radmask(cluster)

                self.depthstr.calc_maskdepth(self.mask.maskgals,
                                             cluster.ra, cluster.dec, cluster.mpc_scale)

            rmask = self.rmask_0 * (cluster.Lambda/100.)**self.rmask_beta * ((1. + cluster.redshift) / (1. + self.rmask_zpivot))**self.rmask_gamma

            if rmask < cluster.r_lambda:
                rmask = cluster.r_lambda

            if i == (self.config.percolation_niter - 1):
                # this is the last iteration, make sure we go out to the mask radius
                lc, = np.where(cluster.neighbors.r < 1.1 * rmask)
            else:
                # previous iteration -- leave cushion for getting bigger
                lc, = np.where(cluster.neighbors.r < 2.0 * cluster.r_lambda)

            if lc.size < 2:
                bad = True
                continue

            lam = cluster.calc_richness(self.mask)

            if (((cluster.Lambda/cluster.scaleval) < self.config.percolation_minlambda) or
                (cluster.neighbors.pfree[cent.maxind] < self.config.percolation_pbcg_cut)):
                bad = True
                continue

            if i == 0:
                # Maybe this is i less than maxiter??
                # Only on the first iteration -- with new center -- is this necessary
                zlam = Zlambda(cluster)
                z_lambda, z_lambda_e = zlam.calc_zlambda(cluster.redshift, self.mask,
                                                         calc_err=True, calcpz=True)
                cluster.z_lambda = z_lambda
                cluster.z_lambda_e = z_lambda_e
                cluster.pzbins[:] = zlam.pzbins
                cluster.pz[:] = zlam.pz

            if not self.keepz:
                cluster.redshift = z_lambda

            if cluster.redshift < 0.0:
                bad = True
                continue

        if bad:
            # Kick out here, just in case it went bad on last iter...
            self._reset_bad_values(cluster)
            return bad

        # and we're done with the iteration loop

        # Compute connectivity factor w
        minind = np.argmin(cluster.neighbors.r)
        u, = np.where((cluster.neighbors.r > cluster.neighbors.r[minind]) &
                      (cluster.neighbors.r < cluster.r_lambda) &
                      (cluster.neighbors.p > 0.0))
        if u.size == 0:
            # This is another way to get a bad cluster
            self._reset_bad_values(cluster)
            return bad

        lum = 10.**((cluster.mstar - cluster.neighbors.refmag[u]) / 2.5)
        if self.config.wcen_uselum:
            cluster.w = np.log(np.sum(cluster.neighbors.p[u] * lum / np.sqrt(cluster.neighbors.r[u]**2. + self.config.wcen_rsoft**2.)) / ((1./cluster.r_lambda) * np.sum(cluster.neighbors.p[u] * lum)))
        else:
            cluster.w = np.log(np.sum(cluster.neighbors.p[u] / np.sqrt(cluster.neighbors.r[u]**2. + self.config.wcen_rsoft**2.)) / ((1./cluster.r_lambda) * np.sum(cluster.neighbors.p[u])))

        # We need to compute richness for other possible centers!
        cluster.lambda_cent[0] = cluster.Lambda
        cluster.zlambda_cent[0] = cluster.z_lambda
        if cluster.ncent_good > 1:
            for ce in xrange(1,cluster.ncent_good):
                cluster_temp = cluster.copy()
                cluster_temp.ra = cluster.ra_cent[ce]
                cluster_temp.dec = cluster.dec_cent[ce]
                cluster_temp.update_neighbors_dist()

                clc, = np.where(cluster_temp.neighbors.r < 1.5*cluster.r_lambda)
                lam = cluster_temp.calc_richness(self.mask, calc_err=False, idx=clc)
                cluster.lambda_cent[ce] = lam

                if ce == 1:
                    # For just the first alternate center compute z_lambda (for speed)
                    zlam = Zlambda(cluster_temp)
                    z_lambda, _ = zlam.calc_zlambda(cluster.redshift, self.mask, calc_err=False, calpz=False)
                    cluster.zlambda_cent[ce] = z_lambda

            # And the overall average over the centers...
            cluster.lambda_c = np.sum(cluster.p_cen * cluster.lambda_cent)
            cluster.lambda_ce = np.sqrt(np.sum(cluster.p_cen * cluster.lambda_cent**2.) - cluster.lambda_c**2.)
        else:
            # There's just one possible center...
            cluster.lambda_c = cluster.Lambda
            cluster.lambda_ce = 0.0


        # Everything else should be taken care of by cluster_runner
        return bad
