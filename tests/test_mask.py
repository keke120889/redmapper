import unittest
import numpy.testing as testing
import numpy as np
import fitsio

from redmapper.mask import HPMask
from redmapper.config import Configuration

class MaskTestCase(unittest.TestCase):
    def test_length(self):
        print self.mask.mside, self.mask.offset, self.mask.npix

    def setUp(self):
        """
        Tom - this is probably wrong at the moment. I cannot construct
        the HPMask because Entry() doesn't work when taking
        in a healpix array.
        """
        self.file_path = "data_for_tests"
        conf_filename = "testconfig.yaml"
        confstr = Configuration(self.file_path + "/" + conf_filename)
        self.mask = HPMask(confstr)

if __name__=='__main__':
    unittest.main()
