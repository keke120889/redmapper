from __future__ import print_function

import fitsio
import numpy as np
import esutil
from esutil.cosmology import Cosmo

import configuration
from cluster import ClusterCatalog
from background import Background
#from mask import HPMask
from mask import get_mask
from galaxy import GalaxyCatalog
from catalog import Catalog
from cluster import Cluster
from cluster import ClusterCatalog
from depthmap import DepthMap
from zlambda import Zlambda
from zlambda import ZlambdaCorrectionPar
from redmapper.redsequence import RedSequenceColorPar

class ClusterRunner(object):
    """
    """

    def __init__(self, conf):
        if not isinstance(conf, configuration.Configuration):
            # this needs to be read
            self.config = configuration.read_config(conf)
        else:
            self.config = conf

        # Will want to add stuff to check that everything needed is present?

        self._set_runmode()

    def _set_runmode(self):
        """
        """

        # must be overridden
        self.runmode = None


    def _setup(self):
        """
        """
        self.r0 = self.config.__getattribute__(self.runmode + '_r0')
        self.beta = self.config.__getattribute__(self.runmode + '_beta')

        # This always uses the "percolation" settings, maybe?
        self.rmask_0 = self.config.percolation_rmask_0
        self.rmask_beta = self.config.percolation_rmask_beta
        self.rmask_gamma = self.config.percolation_rmask_gamma
        self.rmask_zpivot = self.config.percolation_rmask_zpivot

        if self.config.percolation_lmask < 0.0:
            self.percolation_lmask = self.config.lval_reference
        else:
            self.percolation_lmask = self.config.percolation_lmask

        # maxrad is the maximum dist to match neighbors.  maxrad2 is for masking
        # (note this was "maxdist" in IDL which was a bad name)
        if self.beta == 0.0:
            # add a small cushion even if we have constant radius
            self.maxrad = 1.2 * self.r0
            self.maxrad2 = 1.2 * self.rmask_0
        else:
            # maximum possible richness is 300
            self.maxrad = self.r0 * (300./100.)**self.beta
            self.maxrad2 = self.rmask_0 * (300./100.)**self.rmask_beta

        if self.maxrad2 > self.maxrad:
            self.maxrad = self.maxrad2

        # read in background
        self.bkg = Background(self.config.bkgfile)

        # read in parameters
        self.zredstr = RedSequenceColorPar(self.config.parfile, fine=True)

        # And correction parameters
        try:
            self.zlambda_corr = ZlambdaCorrectionPar(self.config.zlambdafile,
                                                     self.config.zlambda_pivot)
        except:
            self.zlambda_corr = None

        # read in mask (if available)
        # This will read in the mask and generate maskgals if necessary
        #  Will work with any type of mask
        self.mask = get_mask(self.config)

        # read in the depth structure
        try:
            self.depthstr = DepthMap(self.config)
        except:
            self.depthstr = None

        #self.cosmo = Cosmo()
        self.cosmo = self.config.cosmo

        # read in the galaxies
        # WILL NEED TO TRACK IF READING ZREDS

        self.gals = GalaxyCatalog.from_galfile(self.config.galfile)

        # Defaults for whether to implement masking, etc
        self.do_percolation_masking = False
        self.do_lam_plusminus = False
        self.use_memradius = False
        self.use_memlum = False
        self.match_centers_to_galaxies = False

    def _more_setup(self, *args, **kwargs):
        # This is to be overridden if necessary
        # this can receive all the keywords.

        # THIS NEEDS TO SET MEM_MATCH_ID!!!
        # make a common convenience function ... hey!

        pass

    def _generate_mem_match_ids(self):
        min_id = self.cat.mem_match_id.min()
        max_id = self.cat.mem_match_id.max()

        if (min_id == max_id):
            # These are unset: generate them
            self.cat.mem_match_id = np.arange(self.cat.size) + 1
        else:
            # Make sure they are unique
            if np.unique(self.cat.mem_match_id).size != self.cat.size:
                raise RuntimeError("Input values for mem_match_id are not unique (and not all unset)")

    def _process_cluster(self, cluster):
        # This must be overridden
        pass

    def run(self, *args, **kwargs):
        """
        """

        # General setup
        self._setup()

        # Setup specific for a given task.  Will read in the galaxy catalog.
        self._more_setup(*args, **kwargs)

        # Match centers and galaxies if required
        if self.match_centers_to_galaxies:
            i0, i1, dist = self.gals.match_many(self.cat.ra, self.cat.dec, 1./3600.)
            self.cat.refmag[i0] = self.gals.refmag[i1]
            self.cat.refmag_err[i0] = self.gals.refmag_err[i1]
            self.cat.mag[i0, :] = self.gals.mag[i1, :]
            self.cat.mag_err[i0, :] = self.gals.mag[i1, :]

        # loop over clusters...
        # at the moment, we are doing the matching once per cluster.
        # if this proves too slow we can prematch bulk clusters as in the IDL code

        if self.do_percolation_masking:
            print("Setting pgal...")
            self.pgal = np.zeros(self.gals.size, dtype=np.float32)

        self.members = None

        for cluster in self.cat:
            match_radius = np.degrees(self.maxrad / self.cosmo.Da(0.0, cluster._z))
            cluster.find_neighbors(match_radius, self.gals)

            if self.do_percolation_masking:
                cluster.neighbors.pfree[:] = 1.0 - self.pgal[cluster.neighbors.index]
            else:
                cluster.neighbors.pfree[:] = 1.0

            print("Pfree goes from %.3f to %.3f" % (cluster.neighbors.pfree.min(),
                                                    cluster.neighbors.pfree.max()))

            if self.depthstr is None:
                # must approximate the limiting magnitude
                # will get this from my des_depth functions...
                pass
            else:
                # get from the depth structure
                self.depthstr.calc_maskdepth(self.mask.maskgals,
                                             cluster.ra, cluster.dec, cluster.mpc_scale())

                cluster.lim_exptime = np.median(self.mask.maskgals.exptime)
                cluster.lim_limmag = np.median(self.mask.maskgals.limmag)
                cluster.lim_limmag_hard = self.config.limmag_catalog

            # And survey masking (this may be a dummy)
            self.mask.set_radmask(cluster, cluster.mpc_scale())

            # And compute maskfrac here...approximate first computation
            inside, = np.where(self.mask.maskgals.r < 1.0)
            bad, = np.where(self.mask.maskgals.mark[inside] == 0)
            cluster.maskfrac = float(bad.size) / float(inside.size)

            # Note that _process_cluster has the index part maybe?
            bad_cluster = self._process_cluster(cluster)

            if bad_cluster:
                # This is a bad cluster and we can't continue
                continue

            # compute updated maskfrac (always)
            inside, = np.where(self.mask.maskgals.r < cluster.r_lambda)
            bad, = np.where(self.mask.maskgals.mark[inside] == 0)
            cluster.maskfrac = float(bad.size) / float(inside.size)

            # compute additional dlambda bits (if desired)
            if self.do_lam_plusminus:
                cluster_temp = cluster.copy()

                #cluster_temp.z = cluster.z_lambda - self.config.zlambda_epsilon
                cluster_temp.update_z(cluster.z_lambda - self.config.zlambda_epsilon)
                lam_zmeps = cluster_temp.calc_richness(self.mask)
                elambda_zmeps = cluster_temp.lambda_e
                #cluster_temp.z = cluster.z_lambda + self.config.zlambda_epsilon
                cluster_temp.update_z(cluster.z_lambda + self.config.zlambda_epsilon)
                lam_zpeps = cluster_temp.calc_richness(self.mask)
                elambda_zpeps = cluster_temp.lambda_e

                cluster.dlambda_dz = (np.log(lam_zpeps) - np.log(lam_zmeps)) / (2. * self.config.zlambda_epsilon)
                cluster.dlambda_dz2 = (np.log(lam_zpeps) + np.log(lam_zmeps) - 2.*np.log(cluster.Lambda)) / (self.config.zlambda_epsilon**2.)

                cluster.dlambdavar_dz = (elambda_zpeps**2. - elambda_zmeps**2.) / (2.*self.config.zlambda_epsilon)
                cluster.dlambdavar_dz2 = (elambda_zpeps**2. + elambda_zmeps**2. - 2.*cluster.Lambda_e**2.) / (self.config.zlambda_epsilon**2.)

            # and record pfree if desired
            if self.do_percolation_masking:
                r_mask = (self.rmask_0 * (cluster.Lambda/100.)**self.rmask_beta *
                          ((1. + cluster._z)/(1. + self.rmask_zpivot))**self.rmask_gamma)
                if (r_mask < cluster.r_lambda):
                    r_mask = cluster.r_lambda
                cluster.r_mask = r_mask
                print("And r_mask = %.3f" % (r_mask))

                lim = cluster.mstar() - 2.5*np.log10(self.percolation_lmask)

                u, = np.where((cluster.neighbors.refmag < lim) &
                              (cluster.neighbors.r < r_mask) &
                              (cluster.neighbors.p > 0.0))
                if (u.size > 0):
                    self.pgal[cluster.neighbors.index[u]] += cluster.neighbors.p[u]

            # and save members
            # Note that this is probably horribly inefficient for memory
            #  usage right now, but will start here and fix later if it
            #  is a problem.

            pfree_temp = cluster.neighbors.pfree[:]

            if self.use_memradius or self.use_memlum:
                ok = (cluster.neighbors.p > 0.01)

                if self.use_memradius:
                    ok &= (cluster.neighbors.r < self.config.percolation_memradius * cluster.r_lambda)
                if self.use_memlum:
                    ok &= (cluster.neighbors.refmag < (cluster.mstar() - 2.5*np.log10(self.config.percolation_memlum)))

                # And set pfree_temp to zero when it is not okay
                pfree_temp[~ok] = 0.0
            else:
                # Only save members where pmem > 0.01 (for space)
                ok = (cluster.neighbors.pmem > 0.01)
                pfree_temp[~ok] = 0.0

            memuse, = np.where(pfree_temp > 0.01)
            mem_temp = Catalog.zeros(memuse.size, dtype=self.config.member_dtype)

            mem_temp.mem_match_id[:] = cluster.mem_match_id
            mem_temp.z[:] = cluster.z_lambda
            mem_temp.ra[:] = cluster.neighbors.ra[memuse]
            mem_temp.dec[:] = cluster.neighbors.dec[memuse]
            mem_temp.r[:] = cluster.neighbors.r[memuse]
            mem_temp.p[:] = cluster.neighbors.p[memuse]
            mem_temp.pfree[:] = pfree_temp[memuse]
            mem_temp.theta_i[:] = cluster.neighbors.theta_i[memuse]
            mem_temp.theta_r[:] = cluster.neighbors.theta_r[memuse]
            mem_temp.refmag[:] = cluster.neighbors.refmag[memuse]
            mem_temp.refmag_err[:] = cluster.neighbors.refmag_err[memuse]
            # mem_temp.zred[:] = cluster.neighbors.zred[memuse]
            # mem_temp.zred_e[:] = cluster.neighbors.zred_e[memuse]
            mem_temp.chisq[:] = cluster.neighbors.chisq[memuse]
            mem_temp.ebv[:] = cluster.neighbors.ebv[memuse]
            mem_temp.mag[:, :] = cluster.neighbors.mag[memuse, :]
            mem_temp.mag_err[:, :] = cluster.neighbors.mag_err[memuse, :]

            if self.members is None:
                self.members = mem_temp
            else:
                self.members.append(mem_temp)


    def output(self, savemembers=True):
        """
        """
        # Can this be universal?

        # maybe want to be able to override to say which fields to save.
        # that would be clever.  Maybe with a cls?

        pass
