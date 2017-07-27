import unittest
import numpy.testing as testing
import numpy as np
import fitsio
from numpy import random

from esutil.cosmology import Cosmo
from redmapper.catalog import Entry
from redmapper.cluster import Cluster
from redmapper.config import Configuration
from redmapper.galaxy import GalaxyCatalog
from redmapper.background import Background
from redmapper.redsequence import RedSequenceColorPar
from redmapper.mask import HPMask
from redmapper.depthmap import DepthMap

class ClusterFiltersTestCase(unittest.TestCase):
    """
    Unittest for testing the z_lambda functions
    
    INCOMPLETE
    
    """
    def runTest(self):
        
        self.file_path = 'data_for_tests'
        
        #The configuration
        #Used by the mask, background and richness calculation
        
        self.cluster = Cluster()
        
        conf_filename = 'testconfig.yaml'
        self.cluster.confstr = Configuration(self.file_path + '/' + conf_filename)
        
        filename = 'test_cluster_members.fit'
        self.cluster.neighbors = GalaxyCatalog.from_fits_file(self.file_path + '/' + filename)
        
        zred_filename = 'test_dr8_pars.fit'
        self.cluster.zredstr = RedSequenceColorPar(self.file_path + '/' + zred_filename,fine = True)
        
        bkg_filename = 'test_bkg.fit'
        self.cluster.bkg = Background('%s/%s' % (self.file_path, bkg_filename))
        
        hdr=fitsio.read_header(self.file_path+'/'+filename,ext=1)
        self.cluster.z = hdr['Z_LAMBDA']
        self.richness_compare = hdr['LAMBDA']
        self.richness_compare_err = hdr['LAMBDA_E']
        self.cluster.z = self.cluster.neighbors.z[0]
        self.cluster.ra = hdr['RA']
        self.cluster.dec = hdr['DEC']
        
        #Set up the mask
        mask = HPMask(self.cluster.confstr) #Create the mask
        
        mpc_scale = np.radians(1.) * self.cluster.cosmo.Dl(0, self.cluster.z) / (1 + self.cluster.z)**2
        mask.set_radmask(self.cluster, mpc_scale)
        
        #depthstr
        depthstr = DepthMap(self.cluster.confstr)
        depthstr.calc_maskdepth(mask.maskgals, self.cluster.ra, self.cluster.dec, mpc_scale)
        
        self.cluster.neighbors.dist = np.degrees(self.cluster.neighbors.r/self.cluster.cosmo.Dl(0,self.cluster.z))
        
        #set seed
        seed = 0
        random.seed(seed = seed)
        
        z_lambda = self.cluster.redmapper_zlambda(self.cluster.z, mask)
        
if __name__=='__main__':
    unittest.main() 