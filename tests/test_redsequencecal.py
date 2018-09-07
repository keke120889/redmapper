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
from redmapper.calibration import RedSequenceCalibrator

class SelectRedSequenceCalTestCase(unittest.TestCase):
    def test_selectredsequencecal(self):
        file_path = 'data_for_tests'
        conf_filename = 'testconfig.yaml'
        config = Configuration(os.path.join(file_path, conf_filename))
        config.calib_use_pcol = True

        config.zrange = [0.1, 0.2]

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')
        config.outpath = self.test_dir

        config.parfile = os.path.join(config.outpath, '%s_testpars.fit' % (config.d.outbase))

        gal_filename = 'test_dr8_trainredseq_gals.fit'
        galfile = os.path.join(file_path, gal_filename)

        redsequencecal = RedSequenceCalibrator(config, galfile)
        redsequencecal.run(doRaise=False)

        # make sure that the pars and the plots were made
        self.assertTrue(os.path.isfile(config.parfile))
        for i in xrange(config.nmag - 1):
            plotfile = os.path.join(config.outpath, config.plotpath, '%s_%s-%s.png' % (config.d.outbase, config.bands[i], config.bands[i + 1]))
            self.assertTrue(plotfile)
        plotfile = os.path.join(config.outpath, config.plotpath, '%s_zred_plots.png' % (config.d.outbase))
        self.assertTrue(plotfile)
        plotfile = os.path.join(config.outpath, config.plotpath, '%s_zred2_plots.png' % (config.d.outbase))
        self.assertTrue(plotfile)
        plotfile = os.path.join(config.outpath, config.plotpath, '%s_offdiags.png' % (config.d.outbase))
        self.assertTrue(plotfile)

        # Read in the pars and check numbers
        pars = fitsio.read(config.parfile, ext=1)
        testing.assert_almost_equal(pars[0]['pivotmag'], np.array([17.11333275, 18.5870018]), 5)
        testing.assert_almost_equal(pars[0]['medcol'], np.array([[1.89056611, 0.92165649, 0.40067941, 0.33493668],
                                                                 [2.00396729, 1.23734939, 0.47769779, 0.32707551]]), 5)
        testing.assert_almost_equal(pars[0]['c01'], np.array([0.92372364, 1.07900822, 1.23833561]), 5)
        testing.assert_almost_equal(pars[0]['slope01'], np.array([-0.00433907, -0.02314611]), 5)
        testing.assert_almost_equal(pars[0]['covmat_amp'][1: 3, 1: 3, :],
                                    np.array([[[0.00137495, 0.00360158],
                                               [-0.00014885, 0.00035394]],
                                              [[-0.00014885, 0.00035394],
                                               [0.00034374, 0.00048195]]]), 5)

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)

if __name__=='__main__':
    unittest.main()
