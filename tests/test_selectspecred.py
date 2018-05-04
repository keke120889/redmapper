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
from redmapper.calibration.selectspecred import SelectSpecRedGalaxies

class SelectSpecRedTestCase(unittest.TestCase):
    def test_selectred(self):
        file_path = 'data_for_tests'
        conf_filename = 'testconfig.yaml'
        config = Configuration(os.path.join(file_path, conf_filename))

        config.galfile = os.path.join(file_path, 'test_dr8_trainred_gals.fit')
        config.specfile_train = os.path.join(file_path, 'test_dr8_trainred_spec.fit')
        config.zrange = [0.1,0.2]

        test_dir = tempfile.mkdtemp(dir='./', prefix="TestRedmapper-")
        config.outpath = test_dir

        config.redgalfile = os.path.join(test_dir, 'test_redgals.fits')
        config.redgalmodelfile = os.path.join(test_dir, 'test_redgalmodels.fits')

        selred = SelectSpecRedGalaxies(config)

        selred.run()

        # Check that files got made
        self.assertTrue(os.path.isfile(config.redgalfile))
        self.assertTrue(os.path.isfile(config.redgalmodelfile))
        self.assertTrue(os.path.isfile(os.path.join(config.outpath,
                                                    config.plotpath,
                                                    '%s_redgals_g-r.png' % (config.outbase))))

        redgals = fitsio.read(config.redgalfile, ext=1)
        redgalmodel = fitsio.read(config.redgalmodelfile, ext=1)

        self.assertEqual(redgals.size, 1063)
        testing.assert_almost_equal(redgalmodel['MEANCOL'][0][:, 1],
                                    np.array([0.78112811, 1.087502, 1.4756074]), 5)
        testing.assert_almost_equal(redgalmodel['MEANCOL_SCATTER'][0][:, 1],
                                    np.array([0.01882466, 0.03260772, 0.01009096]), 5)
        testing.assert_almost_equal(redgalmodel['MEDCOL'][0][:, 1],
                                    np.array([0.78392178, 1.08610117, 1.45235968]), 5)
        testing.assert_almost_equal(redgalmodel['MEDCOL_WIDTH'][0][:, 1],
                                    np.array([0.02155463, 0.04549022, 0.01675996]), 5)

        if os.path.exists(test_dir):
            shutil.rmtree(test_dir, True)


if __name__=='__main__':
    unittest.main()
