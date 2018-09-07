from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import unittest
import numpy.testing as testing
import numpy as np
import fitsio
from numpy import random

from redmapper import Entry
from redmapper import Cluster
from redmapper import Configuration
from redmapper import GalaxyCatalog
from redmapper import Background
from redmapper import RedSequenceColorPar
from redmapper import HPMask
from redmapper import DepthMap
from redmapper.utilities import calc_theta_i

class ClusterTestCase(unittest.TestCase):
    """
    This file tests multiple features of the cluster object.
    """
    def runTest(self):
        """
        First test the filters:
        nfw, lum, and bkg
        """

        # all new...

        random.seed(seed=12345)

        file_path = 'data_for_tests'

        cluster = Cluster()

        conf_filename = 'testconfig.yaml'
        cluster.config = Configuration(file_path + '/' + conf_filename)

        filename = 'test_cluster_members.fit'

        neighbors = GalaxyCatalog.from_fits_file(file_path + '/' + filename)

        cluster.set_neighbors(neighbors)

        zred_filename = 'test_dr8_pars.fit'
        cluster.zredstr = RedSequenceColorPar(file_path + '/' + zred_filename, fine=True)

        bkg_filename = 'test_bkg.fit'
        cluster.bkg = Background('%s/%s' % (file_path, bkg_filename))

        hdr=fitsio.read_header(file_path+'/'+filename,ext=1)
        #cluster.z = hdr['Z']
        #cluster.update_z(hdr['Z'])
        cluster.redshift = hdr['Z']
        richness_compare = hdr['LAMBDA']
        richness_compare_err = hdr['LAMBDA_E']
        scaleval_compare = hdr['SCALEVAL']
        cpars_compare = np.array([hdr['CPARS0'], hdr['CPARS1'], hdr['CPARS2'], hdr['CPARS3']])
        cval_compare = hdr['CVAL']
        mstar_compare = hdr['MSTAR']
        cluster.ra = hdr['RA']
        cluster.dec = hdr['DEC']


        mask = HPMask(cluster.config)
        mask.set_radmask(cluster)

        depthstr = DepthMap(cluster.config)
        depthstr.calc_maskdepth(mask.maskgals, cluster.ra, cluster.dec, cluster.mpc_scale)

        # Test the NFW profile on its own
        #  (this works to 5 decimal places because of the 2*pi*r scaling)
        nfw_python = cluster._calc_radial_profile()
        testing.assert_almost_equal(nfw_python, neighbors.nfw/(2.*np.pi*neighbors.r),5)

        # Test the luminosity
        # The problem is that the IDL code has multiple ways of computing the index.
        # And this should be fixed here.  I don't want to duplicate dumb ideas.

        #mstar = cluster.zredstr.mstar(cluster.z)
        #testing.assert_almost_equal(mstar, mstar_compare, 3)
        #maxmag = mstar - 2.5*np.log10(cluster.config.lval_reference)
        #lum_python = cluster._calc_luminosity(maxmag)
        #testing.assert_almost_equal(lum_python, neighbors.lumwt, 3)

        # Test theta_i
        #theta_i_python = calc_theta_i(neighbors.refmag, neighbors.refmag_err,
        #                              maxmag, cluster.zredstr.limmag)
        #testing.assert_almost_equal(theta_i_python, neighbors.theta_i, 3)

        # Test the background
        #  Note that this uses the input chisq values
        bkg_python = cluster.calc_bkg_density(cluster.neighbors.r,
                                              cluster.neighbors.chisq,
                                              cluster.neighbors.refmag)
        # this is cheating here...
        to_test, = np.where((cluster.neighbors.refmag < cluster.bkg.refmagbins[-1]))
        #testing.assert_almost_equal(bkg_python[to_test], neighbors.bcounts[to_test], 3)
        #testing.assert_allclose(bkg_python[to_test], neighbors.bcounts[to_test],
        #                        rtol=1e-4, atol=0)

        # skip this test for now.  Blah.

        # Now the cluster tests

        # cluster.neighbors.dist = np.degrees(cluster.neighbors.r / cluster.cosmo.Da(0, cluster.redshift))

        seed = 0
        random.seed(seed = 0)

        richness = cluster.calc_richness(mask)

        # these are regression tests.  Various mask issues make the matching
        #  to idl for the time being
        #testing.assert_almost_equal(cluster.Lambda, 23.86299324)
        testing.assert_almost_equal(cluster.Lambda, 24.366407, 5)
        testing.assert_almost_equal(cluster.lambda_e, 2.5137918, 5)

        #testing.assert_almost_equal(cluster.neighbors.theta_i,
        #                            neighbors.theta_i, 3)
        #testing.assert_almost_equal(cluster.neighbors.theta_r,
        #                            neighbors.theta_r, 3)
        #testing.assert_almost_equal(cluster.neighbors.p,
        #                            neighbors.p, 3)

        return

#class ClusterMembersTestCase(unittest.TestCase):

    #This next test MUST be done before the calc_richness test can be completed.
#    def test_member_finding(self): pass #Do this with a radius that is R_lambda of a 
    #lambda=300 cluster, so 8.37 arminutes or 0.1395 degrees
#    def test_richness(self): pass

if __name__=='__main__':
    unittest.main()

