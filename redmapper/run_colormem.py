from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import fitsio
import numpy as np
import esutil
import copy
import sys

from .cluster import ClusterCatalog
from .color_background import ColorBackground
from .mask import HPMask
from .galaxy import GalaxyCatalog
from .cluster import Cluster
from .cluster import ClusterCatalog
from .depthmap import DepthMap
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

class RunColormem(ClusterRunner):
    """
    """

    def _additional_initialization(self, **kwargs):
        self.runmode = "calib_colormem"
        self.filetype = "colormem"
        self.use_colorbkg = True
        self.use_parfile = False

    def _more_setup(self, *args, **kwargs):
        self.rmask_0 = self.config.colormem_r0
        self.rmask_beta = self.config.colormem_beta

        self.cat = ClusterCatalog.from_catfile(self.config.redgalfile,
                                               zredstr=self.zredstr,
                                               config=self.config,
                                               cbkg=self.cbkg,
                                               cosmo=self.cosmo)

        use, = np.where((self.cat.z > self.config.zrange[0]) &
                        (self.cat.z < self.config.zrange[1]))
        self.cat = self.cat[use]

        # need to insert red model and get colormodes...
        self.cat.add_fields([('redmodel', 'f4', self.config.nmag - 1)])

        self.zbounds = np.concatenate([self.config.zrange[0] - 0.011,
                                       self.config.calib_colormem_zbounds,
                                       self.config.zrange[1] + 0.011])

        self.do_percolation_masking = False
        self.do_lam_plusminus = False
        self.record_members = False

        self.doublerun = True

    def _doublerun_sort(self):
        st = np.argsort(self.cat.Lambda)[::-1]
        self.cat = self.cat[st]

    def _process_cluster(self, cluster):

        bad = False

        m = 0
        found = False
        while ((m < self.zbounds.size - 1) and (not Found)):
            if (cluster.z > self.zbounds[m]) and (cluster.z <= self.zbounds[m + 1]):
                found = True
                mode = self.colormodes[m]
            else:
                m += 1
        if (not found):
            raise RuntimeError("Programmer error with illegal mode")

        lam = cluster.calc_richness_fit(self.mask, mode, calc_err=False,
                                        centcolor_in=cluster.redcolor[mode])

        ind = np.argmin(cluster.neighbors.r)
        cluster.p_bcg = cluster.neighbors.pmem[ind]

        if (lam / cluster.scaleval < self.config.calib_colormem_minlambda):
            bad = True
            self._reset_bad_values(cluster)
            return bad

        if cluster.p_bcg < self.config.calib_pcut:
            bad = True
            self._reset_bad_values(cluster)
            return bad

        return bad

