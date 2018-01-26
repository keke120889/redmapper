import fitsio
import esutil

import config as rmconfig
from cluster import ClusterCatalog
from background import Background
from mask import HPMask
from galaxy import GalaxyCatalog
from cluster import Cluster
from cluster import ClusterCatalog
from depthmap import DepthMap
from zlambda import Zlambda
from zlambda import ZlambdaCorrectionPar

class ClusterRunner(object):
    """
    """

    def __init__(self, conf):
        if not isinstance(conf, dict):
            # this needs to be read
            self.config = rmconfig.read_config(conf)
        else:
            self.config = conf

        # Will want to add stuff to check that everything needed is present?

    def _setup(self, mode):
        """
        """

        self.r0 = self.config[mode + '_r0']
        self.beta = self.config[mode + '_beta']

        # This always uses the "percolation" settings, maybe?
        self.rmask_0 = self.config['percolation_rmask_0']
        self.rmask_beta = self.config['percolation_rmask_beta']
        self.rmask_gamma = self.config['percolation_rmask_gamma']
        self.zpivot = self.config['percolation_rmask_zpivot']

        if self.config['percolation_lmask'] < 0.0:
            self.percolation_lmask = self.config['lval_reference']
        else:
            self.percolation_lmask = self.config['percolation_lmask']

        # maxdist is the maximum dist to match neighbors.  maxdist2 is for masking
        if self.beta == 0.0:
            # add a small cushion even if we have constant radius
            self.maxdist = 1.2 * self.r0
            self.maxdist2 = 1.2 * self.rmask_0
        else:
            # maximum possible richness is 300
            self.maxdist = self.r0 * (300./100.)**self.beta
            self.maxdist2 = self.rmask_0 * (300./100.)**self.rmask_beta

        if self.maxdist2 > self.maxdist:
            self.maxdist = self.maxdist2

        # read in background
        self.bkg = Background(self.config['bkgfile'])

        # read in parameters
        self.zredstr = RedSequenceColorPar(self.config['parfile'], fine=True)

        # read in mask
        self.mask = HPMask(self.config)

        # read in the depth structure
        try:
            self.depthstr = DepthMap(self.config)
        except:
            self.depthstr = None

        # read in the zlambda correction
        try:
            self.zlambda_corr = ZlambdaCorrectionPar(self.config['zlambdafile'])
        except:
            self.zlambda_corr = None

        # read in the galaxies

    def _more_setup(self):
        # This is to be overridden if necessary
        pass

    def _process_cluster(self, *args):
        # This must be overridden
        pass

    def run(self, *args, **kwargs):
        """
        """

        self._setup(kwargs['run_mode'])
        self._more_setup()

        # so _more_setup() will read in the input catalog?

        # match clusters and galaxies?

        # loop over clusters

        # compute lim mag stuff if depth not available (must test)

        self._process_cluster(add,some,parameters)

        # record pfree values if necessary

        # etc.

    def output(self, savemembers=True):
        """
        """
        # Can this be universal?

        # maybe want to be able to override to say which fields to save.
        # that would be clever.  Maybe with a cls?

        pass
