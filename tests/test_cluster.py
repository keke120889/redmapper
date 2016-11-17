import unittest
import numpy.testing as testing
import numpy as np
import fitsio

from esutil.cosmology import Cosmo
from redmapper.catalog import Entry
from redmapper.cluster import Cluster
from redmapper.config import Configuration
from redmapper.galaxy import GalaxyCatalog
from redmapper.background import Background
from redmapper.redsequence import RedSequenceColorPar


class BackgroundStub(Background):

    def __init__(self, filename):
        obkg = Entry.from_fits_file(filename)
        self.refmagbins = obkg.refmagbins
        self.chisqbins = obkg.chisqbins
        self.lnchisqbins = obkg.lnchisqbins
        self.zbins = obkg.zbins
        self.sigma_g = obkg.sigma_g
        self.sigma_lng = obkg.sigma_lng


class ClusterFiltersTestCase(unittest.TestCase):
    """
    This file tests multiple features of the cluster object.
    """

    def test_nfw_filter(self):
        """
        This test compres the NFW profiles calculated using the
        cluster._calc_radial_profile() function for specific galaxies (indices)
        in the cluster catalog to the outputs calculated in the IDL version.
        """
        test_indices = np.array([46, 38,  1,  2, 11, 24, 25, 16])
        py_nfw = self.cluster._calc_radial_profile()[test_indices]
        idl_nfw = np.array([0.23875841, 0.033541825, 0.032989189, 0.054912228, 
                            0.11075225, 0.34660992, 0.23695366, 0.25232968])
        testing.assert_almost_equal(py_nfw, idl_nfw)

    def test_lum_filter(self):
        """
        This tests the luminosity calculation found in the
        cluster._calc_luminosity() function for specific galaxies (indices)
        in the cluster catalog. It compares the output to precomputed
        outputs from the IDL code.
        """
        test_indices = np.array([47, 19,  0, 30, 22, 48, 34, 19])
        zred_filename = 'test_dr8_pars.fit'
        conf_filename = 'testconfig.yaml'
        zredstr = RedSequenceColorPar(self.file_path + '/' + zred_filename)
        confstr = Configuration(self.file_path + '/' + conf_filename)
        mstar = zredstr.mstar(self.cluster.z)
        maxmag = mstar - 2.5*np.log10(confstr.lval_reference)
        py_lum = self.cluster._calc_luminosity(zredstr, maxmag)[test_indices]
        idl_lum = np.array([0.31448608824729662, 0.51525195091720710, 
                            0.50794115714566024, 0.57002321121039334, 
                            0.48596850373287931, 0.53985704075616792, 
                            0.61754178397796256, 0.51525195091720710])
        testing.assert_almost_equal(py_lum, idl_lum)

    def test_bkg_filter(self):
        """
        This tests the background density calculation from the
        cluster_calc_bkg_density() function. This compares the
        output calculate from this function to the precomputed
        output from the IDL code.
        """
        test_indices = np.array([29, 16, 27,  5, 38, 35, 25, 43])
        bkg_filename = 'test_bkg.fit'
        bkg = BackgroundStub(self.file_path + '/' + bkg_filename)
        py_bkg = self.cluster._calc_bkg_density(bkg, Cosmo())[test_indices]
        idl_bkg = np.array([1.3140464045388294, 0.16422314236185420, 
                            0.56610846527410053, np.inf, 0.79559933744885403, 
                            np.inf, 0.21078853798218194, np.inf])
        testing.assert_almost_equal(py_bkg, idl_bkg)

    def setUp(self):
        """
        This sets up the cluster objct from the cluster_members file.
        """
        self.cluster = Cluster(np.empty(1))
        self.file_path = 'data_for_tests'
        filename = 'test_cluster_members.fit'
        self.cluster.members = GalaxyCatalog.from_fits_file(self.file_path + '/' + filename)
        self.cluster.z = self.cluster.members.z[0]

        
class ClusterMembersTestCase(unittest.TestCase):

    def test_member_finding(self): pass
    def test_richness(self): pass


if __name__=='__main__':
    unittest.main()

