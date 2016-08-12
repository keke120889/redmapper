import unittest
import numpy.testing as testing
import numpy as np
import fitsio

import redmapper

class BackgroundTestCase(unittest.TestCase):

	def test_io(self):
		# test that we fail if we try a non-existent file
        self.assertRaises(IOError, redmapper.background.Background, 
        											'nonexistent.fit')
        # test that we fail if we read a non-fits file
        self.assertRaises(IOError, redmapper.background.Background, 
        							'%s/testconfig.yaml' % (self.file_path))
        # test that we fail if we try a file without the right header info
        self.assertRaises(ValueError, redmapper.background.Background, 
        							'%s/test_dr8_pars.fit' % (self.file_path))

    def test_sigma_g(self):
    	np.random.seed(0)
    	idl_sigma_g_outputs = [np.array([0, 0, 0]),
        			   		   np.array([0, 0, 0]),
        			   		   np.array([0, 0, 0]),
        			   		   np.array([0, 0, 0]),
        			   		   np.array([0, 0, 0])]
        idl_sigma_lng_outputs = [np.array([0, 0, 0]),
        			   		   	 np.array([0, 0, 0]),
        			   		   	 np.array([0, 0, 0]),
        			   		   	 np.array([0, 0, 0]),
        			   		   	 np.array([0, 0, 0])]
        for (a, b) in zip(idl_sigma_g_outputs, idl_sigma_lng_outputs):
        	idx = tuple(np.random.randint(i) for i in self.bkg.sigma_g.shape)
        	self.assert_almost_equal(self.bkg.sigma_g[idx], a, decimal=5)
        	self.assert_almost_equal(self.bkg.sigma_lng[idx], b, decimal=5)

    def test_lookup(self): pass

	def runTest(self):
		self.file_name, self.file_path = 'test_bkg.fit', 'data'
		self.test_io()
        # read in the parameters
        self.bkg = redmapper.background.Background('%s/%s' % (file_path, 
        													  file_name)) 
        self.test_sigma_g()
        self.test_lookup()