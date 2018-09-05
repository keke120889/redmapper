from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import fitsio
import numpy as np
import esutil

from .cluster import ClusterCatalog
from .catalog import Catalog
from .background import Background
from .mask import HPMask
from .galaxy import GalaxyCatalog
from .cluster import Cluster
from .cluster import ClusterCatalog
from .depthmap import DepthMap
from .zlambda import Zlambda
from .zlambda import ZlambdaCorrectionPar
from .cluster_runner import ClusterRunner

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

class RunFirstPass(ClusterRunner):
    """
    """

    def _additional_initialization(self, specmode=False):
        # This is the runmode and where we get the mask/radius config vars from
        self.runmode = 'firstpass'

        self.read_zreds = True
        self.zreds_required = False
        self.specmode = specmode
        if specmode:
            self.filetype = 'firstpass_spec'
        else:
            self.filetype = 'firstpass'

    # FIXME: This needs to specify args here?
    def _more_setup(self, *args, **kwargs):

        incat = None

        # Check if there's a seedfile
        try:
            incat = Catalog.from_fits_file(self.config.seedfile)
        except:
            # If there's no seedfile, it's okay if it's not specmode and we have zreds
            if self.specmode:
                raise RuntimeError("Must have config.seedfile for run_firstpass in specmode.")
            elif not self.did_read_zreds:
                raise RuntimeError("Must have config.seedfile for run_firstpass with no zreds.")
        if incat is not None:
            # generate a cluster catalog from incat
            self.cat = ClusterCatalog.zeros(incat.size,
                                            zredstr=self.zredstr,
                                            config=self.config,
                                            bkg=self.bkg,
                                            cosmo=self.cosmo,
                                            r0=self.r0,
                                            beta=self.beta)

            self.cat.ra = incat.ra
            self.cat.dec = incat.dec
            self.cat.mag = incat.mag
            self.cat.mag_err = incat.mag_err
            self.cat.refmag = incat.refmag
            self.cat.refmag_err = incat.refmag_err
            self.cat.zred = incat.zred
            self.cat.zred_e = incat.zred_e
            self.cat.chisq = incat.zred_chisq
            self.cat.ebv_mean = incat.ebv
            self.cat.z_spec_init = incat.zspec

            if self.specmode:
                self.cat.z_init = incat.zspec
                self.cat.z = incat.zspec
            else:
                self.cat.z_init = incat.zred
                self.cat.z = incat.zred
        else:
            # we must not have a seedfile, and did_read_zreds
            # so generate a cluster catalog from self.gals

            use,=np.where((self.gals.zred >= self.config.zrange[0]) &
                          (self.gals.zred <= self.config.zrange[1]))

            if use.size == 0:
                self.fail = True
                return

            mstar = self.zredstr.mstar(self.gals.zred[use])
            mlim = mstars - 2.5 * np.log10(self.config.lval_reference)

            good, = np.where(self.gals.refmag[use] < mlim)

            if good.size == 0:
                self.fail = True
                return

            self.cat = ClusterCatalog.zeros(good.size,
                                            zredstr=self.zredstr,
                                            config=self.config,
                                            bkg=self.bkg,
                                            cosmo=self.cosmo)

            self.cat.ra = self.gals.ra[use[good]]
            self.cat.dec = self.gals.dec[use[good]]
            self.cat.mag = self.gals.mag[use[good]]
            self.cat.mag_err = self.gals.mag_err[use[good]]
            self.cat.refmag = self.gals.refmag[use[good]]
            self.cat.refmag_err = self.gals.refmag_err[use[good]]
            self.cat.zred = self.gals.zred[use[good]]
            self.cat.zred_e = self.gals.zred_e[use[good]]
            self.cat.chisq = self.gals.chisq[use[good]]

            self.cat.z_init = self.gals.zred
            self.cat.z = self.gals.zred

            # self.match_cat_to_spectra()

        self.do_lam_plusminus = False
        self.match_centers_to_galaxies = False
        self.do_percolation_masking = False
        self.do_correct_zlambda = False
        self.do_pz = False

        #self.limlum = np.clip(self.config.lval_reference - 0.1, 0.01, None)

        self.maxiter = self.config.firstpass_niter
        if self.specmode:
            self.maxiter = 1

        self.min_lambda = self.config.firstpass_minlambda

    def _process_cluster(self, cluster):
        bad = False
        iteration = 0
        done = False

        zuse = cluster.z_init.copy()

        for i in xrange(self.maxiter):
            if bad:
                done = True
                continue

            lam = cluster.calc_richness(self.mask, calc_err=False)

            if (lam < np.abs(self.config.firstpass_minlambda / cluster.scaleval)):
                bad = True
                done = True
                self._reset_bad_values(cluster)
                continue

            if i < self.maxiter:
                # only on first iteration, compute z_lambda
                # Really, this should be on at most n-1th iteration
                zlam = Zlambda(cluster)
                z_lambda, z_lambda_e = zlam.calc_zlambda(cluster.redshift, self.mask,
                                                         calc_err=True, calcpz=False)

                if z_lambda < self.config.zrange[0] or z_lambda > self.config.zrange[1]:
                    bad = True
                    done = True
                    self._reset_bad_values(cluster)
                    continue

                cluster.redshift = z_lambda

        if bad:
            cluster.z_lambda = -1.0
            cluster.z_lambda_e = -1.0
            cluster.z_lambda_niter = 0
        else:
            cluster.z_lambda = z_lambda
            cluster.z_lambda_e = z_lambda_e
            cluster.z_lambda_niter = zlam.niter

        cind = np.argmin(cluster.neighbors.r)
        cluster.chisq = cluster.neighbors.chisq[cind]

        # and record the .z for the next round
        if (self.specmode):
            cluster.z = cluster.z_spec_init
        else:
            cluster.z = cluster.z_lambda

        # All done
        return bad
