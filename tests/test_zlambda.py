import unittest
import numpy.testing as testing
import numpy as np
import fitsio
from numpy import random

from redmapper.cluster import Cluster
from redmapper.configuration import Configuration
from redmapper.galaxy import GalaxyCatalog
from redmapper.background import Background
from redmapper.redsequence import RedSequenceColorPar
from redmapper.mask import HPMask
from redmapper.depthmap import DepthMap
from redmapper.zlambda import Zlambda
from redmapper.zlambda import ZlambdaCorrectionPar

class ClusterZlambdaTestCase(unittest.TestCase):
    """
    Unittest for testing the z_lambda functions

    INCOMPLETE

    """
    def runTest(self):
        file_path = 'data_for_tests'

        #The configuration
        #Used by the mask, background and richness calculation

        conf_filename = 'testconfig.yaml'
        config = Configuration(file_path + '/' + conf_filename)

        filename = 'test_cluster_members.fit'
        neighbors = GalaxyCatalog.from_fits_file(file_path + '/' + filename)

        zred_filename = 'test_dr8_pars.fit'
        zredstr = RedSequenceColorPar(file_path + '/' + zred_filename,fine = True)

        bkg_filename = 'test_bkg.fit'
        bkg = Background('%s/%s' % (file_path, bkg_filename))

        cluster = Cluster(config=config, zredstr=zredstr, bkg=bkg, neighbors=neighbors)

        hdr=fitsio.read_header(file_path+'/'+filename,ext=1)
        #cluster.z = hdr['Z']
        cluster.update_z(hdr['Z'])
        richness_compare = hdr['LAMBDA']
        richness_compare_err = hdr['LAMBDA_E']
        cluster.ra = hdr['RA']
        cluster.dec = hdr['DEC']

        #Set up the mask
        mask = HPMask(cluster.config) #Create the mask

        #mpc_scale = np.radians(1.) * cluster.cosmo.Dl(0, cluster.z) / (1 + cluster.z)**2
        mpc_scale = cluster.mpc_scale()
        mask.set_radmask(cluster, mpc_scale)

        #depthstr
        depthstr = DepthMap(cluster.config)
        depthstr.calc_maskdepth(mask.maskgals, cluster.ra, cluster.dec, mpc_scale)

        cluster.neighbors.dist = np.degrees(cluster.neighbors.r/cluster.cosmo.Dl(0,cluster._z))

        #set seed
        seed = 0
        random.seed(seed = seed)

        # make a zlambda object
        zlam = Zlambda(cluster)

        z_lambda, z_lambda_e = zlam.calc_zlambda(cluster._z, mask, calc_err=True, calcpz=True)

        #testing.assert_almost_equal(self.cluster.z_lambda, 0.22816455)
        #testing.assert_almost_equal(cluster.z_lambda, 0.227865)
        testing.assert_almost_equal(cluster.z_lambda, 0.22700983)
        #testing.assert_almost_equal(self.cluster.z_lambda_err, 0.00632813)
        #testing.assert_almost_equal(cluster.z_lambda_err,0.00630833)
        testing.assert_almost_equal(cluster.z_lambda_err, 0.00448909596)


        # zlambda_err test
        z_lambda_err = zlam._zlambda_calc_gaussian_err(cluster.z_lambda)

        #testing.assert_almost_equal(z_lambda_err, 0.00897011)
        #testing.assert_almost_equal(z_lambda_err, 0.00894175)
        testing.assert_almost_equal(z_lambda_err, 0.006303830)

        # and test the correction on its own
        corr_filename = 'test_dr8_zlambdacorr.fit'

        zlambda_corr = ZlambdaCorrectionPar(file_path + '/' + corr_filename, 30.0)

        zlam_in = 0.227865
        zlam_e_in = 0.00629995
        zlam_out = 0.228654
        zlam_e_out = 0.00840213

        zlam_new, zlam_e_new = zlambda_corr.apply_correction(24.5, zlam_in, zlam_e_in)
        #print(zlam_in, zlam_out, zlam_new)
        #print(zlam_e_in, zlam_e_out, zlam_e_new)

        testing.assert_almost_equal(zlam_new, zlam_out, 5)
        testing.assert_almost_equal(zlam_e_new, zlam_e_out, 5)


if __name__=='__main__':
    unittest.main()
