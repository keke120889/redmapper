from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import matplotlib
matplotlib.use('Agg')

import unittest
import os
import shutil
import numpy.testing as testing
import numpy as np
import fitsio
import tempfile
from numpy import random

from redmapper.configuration import Configuration
from redmapper.calibration import WcenCalibrator

class CenteringCalibratorTestCase(unittest.TestCase):
    def test_centeringcal(self):
        pass

if __name__=='__main__':
    unittest.main()
