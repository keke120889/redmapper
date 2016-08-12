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
        self.assertRaises(AttributeError, redmapper.background.Background, 
                          '%s/test_dr8_pars.fit' % (self.file_path))

    def test_sigma_g(self):
        np.random.seed(0)
        idl_outputs = [0.32197464, 6.4165196, 0.0032830855, 1.4605126,
                       0.0098356586, 0.79848081, 0.011284498, 9.3293247]
        for out in idl_outputs:
            idx = tuple(np.random.randint(i) for i in self.bkg.sigma_g.shape)
            testing.assert_almost_equal(self.bkg.sigma_g[idx], out, decimal=5)

    def test_lookup(self): pass

    def setUp(self):
        self.file_name, self.file_path = 'test_bkg.fit', 'data'
        self.bkg = redmapper.background.Background('%s/%s' % (self.file_path, 
                                                              self.file_name))


if __name__=='__main__':
    unittest.main()