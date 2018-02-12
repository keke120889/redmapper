from __future__ import print_function

import fitsio
import numpy as np
import esutil
import copy

from cluster import ClusterCatalog
from catalog import Catalog
from background import Background
from mask import HPMask
from galaxy import GalaxyCatalog
from cluster import Cluster
from cluster import ClusterCatalog
from depthmap import DepthMap
from zlambda import Zlambda
from zlambda import ZlambdaCorrectionPar
from cluster_runner import ClusterRunner
from utilities import chisq_pdf

class RunLikelihoods(ClusterRunner):
    """
    """

    def _additional_initializations(self):
        self.runmode = 'likelihoods'

        self.read_zreds = False
        self.zreds_required = False
        self.filetype = 'like'

    def _more_setup(self, *args, **kwargs):

        self.cat = Catalog.from_fits_file(self.config.catfile)

        keepz = kwargs.pop('keepz', False)

        zrange = copy.copy(self.config.zrange)
        if keepz:
            zrange[0] -= self.config.calib_zrange_cushion
            zrange[0] = zrange[0] if zrange[0] > 0.05 else 0.05
            zrange[1] += self.config.calib_zrange_cushion

        use, = np.where((self.cat.z > zrange[0]) &
                        (self.cat.z < zrange[1]) &
                        (self.cat.Lambda > 0.0))

        self.cat = self.cat[use]

        self.do_lam_plusminux = False
        self.match_centers_to_galaxies = False
        self.record_members = False

        #self.limlum = np.clip(self.config.lval_reference - 0.1, 0.01, None)

    def _reset_bad_values(self, cluster):
        cluster.lnlamlike = -1e11
        cluster.lnbcglike = -1e11
        cluster.lnlike = -1e11

        super(ClusterRunner, self)._reset_bad_values(cluster)

    def _process_cluster(self, cluster):

        maxmag = cluster.mstar - 2.5*np.log10(self.limlum)

        lam = cluster.calc_richness(self.mask)

        minrind = np.argmin(cluster.neighbors.r)
        incut, = np.where((cluster.neighbors.pmem > 0.0) &
                          (cluster.neighbors.r > cluster.neighbors.r[minrind]))

        if cluster.Lambda < self.config.likelihoods_minlambda or incut.size < 3:
            # this is a bad cluster
            self._reset_bad_values(cluster)
            return

        # compute the cluster member likelihood
        cluster.lnlamlike = (-cluster.Lambda / cluster.scaleval -
                              np.sum(np.log(1.0 - cluster.neighbors.pmem[incut])))

        # And the central likelihood
        if self.config.lnw_cen_sigma <= 0.0:
            # We do not have a calibration yet
            cluster.lnbcglike = 0.0
        else:
            # First phi_cen
            mbar = (cluster.mstar + self.config.wcen_delta0 +
                    self.config.wcen_delta1 * np.log(cluster.Lambda / self.config.wcen_pivot))
            phi_cen = ((1. / (np.sqrt(2. * np.pi) * self.config.wcen_sigma_m)) *
                       np.exp(-0.5 * (cluster.neighbors.refmag[minrind] - mbar)**2. / self.config.wcen_sigma_m**2.))

            # And the zred or chisq filter
            if self.config.likelihoods_use_zred:
                if self.zlambda_corr is not None:
                    zrmod = interpol(self.zlambda_corr.zred_uncorr, self.zlambda_corr.z, cluster.get_z())
                else:
                    zrmod = cluster.get_z()

                g = ((1./(np.sqrt(2. * np.pi) * cluster.neighbors.zred_e[minrind])) *
                     np.exp(-0.5 * (cluster.neighbors.zred[minrind] - zrmod)**2. / cluster.neighbors.zred_e[minrind]**2.))
            else:
                # chisq filter
                g = chisq_pdf(cluster.neighbors.chisq[minrind], self.zredstr.dof)

            # and the w filter
            lum = 10.**((cluster.mstar - cluster.neighbors.refmag) / 2.5)
            u, = np.where((cluster.neighbors.r > 1e-5) & (cluster.neighbors.pmem > 0.0))
            w = np.alog(np.sum(cluster.neighbors.pmem[u] * lum[u] /
                               np.sqrt(cluster.neighbors.r[u]**2. + self.config.wcen_rsoft**2.)) / ((1. / cluster.r_lambda) * np.sum(cluster.neighbors.pmem[u] * lum[u])))
            sig = self.config.lnw_cen_sigma / np.sqrt(((np.clip(cluster.Lambda, None, self.config.wcen_maxlambda)) / cluster.scaleval) / self.config.wcen_pivot)
            fw = (1. / (np.sqrt(2. * np.pi) * sig)) * np.exp(-0.5 * (np.log(w) - self.config.lnw_cen_mean)**2. / (sig**2.))

            cluster.lnbcglike = np.log(phi_cen * g * fw)

            cluster.lnlike = cluster.lnbcglike + clsuter.lnlamlike


    def _postprocess(self):
        # For this catalog we're cutting on failed likelihood

        use, = np.where(self.cat.lnlamlike > -1e11)
        self.cat = self.cat[use]

        # should I delete unused columns here before saving?  
