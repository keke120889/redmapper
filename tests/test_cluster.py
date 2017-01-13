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
    def runTest(self):
        """
        First test the filters:
        nfw, lum, and bkg
        """
        self.cluster = Cluster(np.empty(1))
        self.file_path = 'data_for_tests'
        filename = 'test_cluster_members.fit'
        self.cluster.neighbors = GalaxyCatalog.from_fits_file(self.file_path + '/' + filename)
        self.cluster.z = self.cluster.neighbors.z[0]
        cosmo = Cosmo()

        # nfw
        test_indices = np.array([46, 38,  1,  2, 11, 24, 25, 16])
        py_nfw = self.cluster._calc_radial_profile()[test_indices]
        idl_nfw = np.array([0.23875841, 0.033541825, 0.032989189, 0.054912228, 
                            0.11075225, 0.34660992, 0.23695366, 0.25232968])
        testing.assert_almost_equal(py_nfw, idl_nfw)

        # lum
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

        # bkg
        test_indices = np.array([29, 16, 27,  5, 38, 35, 25, 43])
        bkg_filename = 'test_bkg.fit'
        bkg = BackgroundStub(self.file_path + '/' + bkg_filename)
        py_bkg = self.cluster._calc_bkg_density(bkg, cosmo)[test_indices]
        idl_bkg = np.array([1.3140464045388294, 0.16422314236185420, 
                            0.56610846527410053, np.inf, 0.79559933744885403, 
                            np.inf, 0.21078853798218194, np.inf])
        testing.assert_almost_equal(py_bkg, idl_bkg)

        """
        MIGHT MOVE THIS TO THE ClusterNeighborsTestCase CLASS
        This tests the calc_richness() function from cluster.py.
        The purpose of this function is to calculate the richness,
        sometimes referred to as lambda_chisq, of the cluster
        for a single iteration during the redmapper algorithm.

        THIS TEST IS STILL IN DEVELOPEMENT!!!

        NOTE: the calc_richness() function call
        requires that the neighbors have a 'dist' attribute to them.
        This MUST be calculated before here, and so should
        either be implemented in the setUp() function
        or should be a column in the test_cluster_members.fit file.

        """
        self.cluster = Cluster(np.empty(1))
        self.cluster.ra  = 142.12752
        self.cluster.dec = 65.103898
        self.cluster.z   = 0.228654
        self.file_path = 'data_for_tests'
        filename = 'pixelized_dr8_test/dr8_test_galaxies_master_table.fit'
        self.galcat = GalaxyCatalog.from_galfile(self.file_path +'/'+filename)
        self.cluster.find_neighbors(0.1395,self.galcat)#0.1395;radius in degrees

        #NOTE: self.cluster.neighbors doesn't contain 'dist' attribute
        #print "\tdir(self.cluster.neighbors): ",dir(self.cluster.neighbors)
        #richness_obj = self.cluster.calc_richness(zredstr, bkg, cosmo, confstr)
        #print "\tdir(richness_obj): ",richness_obj,dir(richness_obj)
        
class ClusterMembersTestCase(unittest.TestCase):

    #This next test MUST be done before the calc_richness test can be completed.
    def test_member_finding(self): pass #Do this with a radius that is R_lambda of a 
    #lambda=300 cluster, so 8.37 arminutes or 0.1395 degrees
    def test_richness(self): pass

if __name__=='__main__':
    unittest.main()

