import matplotlib
matplotlib.use('Agg')

import unittest
import numpy.testing as testing
import numpy as np
import fitsio
import copy
from numpy import random
import time
import tempfile
import shutil
import os
import esutil

from redmapper import Configuration
from redmapper import GalaxyCatalog
from redmapper.calibration import SelectSpecSeeds
from redmapper import Catalog

class SelectSpecSeedsTestCase(unittest.TestCase):
    """
    Tests for creating spectroscopic seeds for a run in
    redmapper.calibration.SelectSpecSeeds
    """

    def test_selectspecseeds(self):
        """
        Run tests on redmapper.calibration.SelectSpecSeeds
        """
        file_path = 'data_for_tests'
        configfile = 'testconfig.yaml'

        config = Configuration(os.path.join(file_path, configfile))

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')
        config.outpath = self.test_dir

        # Put in tests here...
        ## FIXME

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)

if __name__=='__main__':
    unittest.main()
