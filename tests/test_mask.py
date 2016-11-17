import unittest
import numpy.testing as testing
import numpy as np
import fitsio

from redmapper.mask import HPMask
from redmapper.config import Configuration

class MaskTestCase(unittest.TestCase):
    def test_length(self):
        print self.mask.nside, self.mask.offset, self.mask.npix

    def setUp(self):
        """
        TODO
        """
        self.file_path = "data_for_tests"
        conf_filename = "testconfig.yaml"
        confstr = Configuration(self.file_path + "/" + conf_filename)
        self.mask = HPMask(confstr)

if __name__=='__main__':
    unittest.main()
