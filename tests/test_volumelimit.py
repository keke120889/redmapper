from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import unittest
import numpy.testing as testing
import numpy as np
import healpy as hp
import tempfile
import shutil
import fitsio
import os

from redmapper import DepthMap
from redmapper import Configuration
from redmapper import VolumeLimitMask

class VolumeLimitMaskTestCase(unittest.TestCase):
    """
    Tests of redmapper.VolumeLimitMask volume-limit mask code.
    """
    def runTest(self):
        """
        Run tests of redmapper.VolumeLimitMask
        """
        file_path = "data_for_tests"
        conf_filename = "testconfig.yaml"
        config = Configuration(file_path + "/" + conf_filename)

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')
        config.outpath = self.test_dir

        vlim = VolumeLimitMask(config, config.vlim_lstar)

        # And test the values...

        ok, = np.where(vlim.zmax > 0)
        self.assertGreater(np.min(vlim.zmax[ok]), 0.3)

        ras = np.array([140.0, 141.0, 150.0])
        decs = np.array([65.25, 65.6, 30.0])

        zmax = vlim.calc_zmax(ras, decs, get_fracgood=False)
        testing.assert_almost_equal(zmax, [0.338, 0.341, 0.0])


    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)

if __name__=='__main__':
    unittest.main()
