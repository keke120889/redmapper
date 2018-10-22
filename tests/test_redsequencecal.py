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
        testing.assert_almost_equal(pars[0]['pivotmag'], np.array([17.11361885, 18.58653641]), 4)
        testing.assert_almost_equal(pars[0]['medcol'], np.array([[1.8905431, 0.92167693, 0.40061119, 0.33523971],
                                                                 [2.00397801, 1.23729753, 0.47771141, 0.32697698]]), 4)
        testing.assert_almost_equal(pars[0]['c01'], np.array([0.92372358, 1.07899547, 1.23832214]), 4)
        testing.assert_almost_equal(pars[0]['slope01'], np.array([-0.00438043, -0.02314976]), 4)
        testing.assert_almost_equal(pars[0]['covmat_amp'][1: 3, 1: 3, :],
                                    np.array([[[0.00134618, 0.00356992],
                                               [-0.00014466, 0.00035207]],
                                              [[-0.00014466, 0.00035207],
                                               [0.00032816, 0.00047648]]]), 4)

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)

if __name__=='__main__':
    unittest.main()
