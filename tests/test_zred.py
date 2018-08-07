from __future__ import division, absolute_import, print_function
from past.builtins import xrange

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

from redmapper import Configuration
from redmapper import GalaxyCatalog
from redmapper import RedSequenceColorPar
from redmapper import ZredColor
from redmapper import ZredRunCatalog, ZredRunPixels

class ZredTestCase(unittest.TestCase):
    """
    """

    def test_zred(self):

        file_path = 'data_for_tests'

        zred_filename = 'test_dr8_pars.fit'
        zredstr = RedSequenceColorPar(file_path + '/' + zred_filename)

        galaxy_filename = 'test_dr8_gals_with_zred.fit'
        galaxies = GalaxyCatalog.from_fits_file(file_path + '/' + galaxy_filename)

        galaxies_input = copy.deepcopy(galaxies)

        # start with the first one...
        zredc = ZredColor(zredstr, adaptive=True)

        zredc.compute_zred(galaxies[0])

        starttime = time.time()
        for i, g in enumerate(galaxies):
            try:
                zredc.compute_zred(g)
            except:
                print("Crashed on %d" % (i))

        print("Ran %d galaxies in %.3f seconds" % (galaxies.size,
                                                   time.time() - starttime))

        # make sure we have reasonable consistency...
        # It seems that at least some of the discrepancies are caused by
        # slight bugs in the IDL implementation.

        delta_zred_uncorr = galaxies.zred_uncorr - galaxies_input.zred_uncorr

        use, = np.where(np.abs(delta_zred_uncorr) < 1e-3)

        testing.assert_array_less(0.9, float(use.size) / float(galaxies.size))

        delta_zred = galaxies.zred[use] - galaxies_input.zred[use]
        use2, = np.where(np.abs(delta_zred) < 1e-3)
        testing.assert_array_less(0.99, float(use2.size) / float(delta_zred.size))

        delta_zred2 = galaxies.zred2[use] - galaxies_input.zred2[use]
        use2, = np.where(np.abs(delta_zred) < 1e-3)
        testing.assert_array_less(0.99, float(use2.size) / float(delta_zred2.size))

        delta_zred_uncorr_e = galaxies.zred_uncorr_e[use] - galaxies_input.zred_uncorr_e[use]
        use2, = np.where(np.abs(delta_zred_uncorr_e) < 1e-3)
        testing.assert_array_less(0.98, float(use2.size) / float(delta_zred_uncorr_e.size))

        delta_zred_e = galaxies.zred_e[use] - galaxies_input.zred_e[use]
        use2, = np.where(np.abs(delta_zred_e) < 1e-3)
        testing.assert_array_less(0.98, float(use2.size) / float(delta_zred_e.size))

        delta_zred2_e = galaxies.zred2_e[use] - galaxies_input.zred2_e[use]
        use2, = np.where(np.abs(delta_zred2_e) < 1e-3)
        testing.assert_array_less(0.98, float(use2.size) / float(delta_zred2_e.size))

    def test_zred_runcat(self):
        """
        Test the running of a single fits catalog file through the zred runner
        """

        file_path = 'data_for_tests'
        configfile = 'testconfig.yaml'

        config = Configuration(os.path.join(file_path, configfile))

        test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')
        config.outpath = test_dir
        outfile = os.path.join(test_dir, 'test_zred_out.fits')

        tab = fitsio.read(config.galfile, ext=1)
        galfile = os.path.join(os.path.dirname(config.galfile), tab[0]['FILENAMES'][0])

        zredRuncat = ZredRunCatalog(config)

        zredRuncat.run(galfile, outfile)

        # This exercises the reading code
        gals = GalaxyCatalog.from_galfile(galfile, zredfile=outfile)

        self.assertGreater(np.min(gals.zred), 0.0)
        self.assertGreater(np.min(gals.chisq), 0.0)
        self.assertLess(np.max(gals.lkhd), 0.0)
        testing.assert_array_almost_equal(gals.zred[0: 3],
                                          np.array([0.10292412, 0.19617805, 0.13324176]))

        if os.path.exists(test_dir):
            shutil.rmtree(test_dir, True)

    def test_zred_runpixels(self):
        """
        Test the running of a pixelized catalog
        """

        file_path = 'data_for_tests'
        configfile = 'testconfig.yaml'

        config = Configuration(os.path.join(file_path, configfile))

        config.d.hpix = 2163
        config.d.nside = 64
        config.border = 0.0

        test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')
        config.zredfile = os.path.join(test_dir, 'zreds', 'testing_zreds_master_table.fit')

        # FIXME: try an illegal one...

        zredRunpix = ZredRunPixels(config)
        zredRunpix.run()

        # Check that the zred file has been built...

        self.assertTrue(os.path.isfile(config.zredfile))

        # Read in just the galaxies...
        gals0 = GalaxyCatalog.from_galfile(config.galfile, nside=config.d.nside, hpix=config.d.hpix, border=config.border)

        # And with the zreds...
        gals = GalaxyCatalog.from_galfile(config.galfile, zredfile=config.zredfile, nside=config.d.nside, hpix=config.d.hpix, border=config.border)

        # Confirm they're the same galaxies...
        testing.assert_array_almost_equal(gals0.ra, gals.ra)

        # Confirm the zreds are okay
        self.assertGreater(np.min(gals.zred), 0.0)
        self.assertGreater(np.min(gals.chisq), 0.0)
        self.assertLess(np.max(gals.lkhd), 0.0)

        # Spot check a few
        testing.assert_array_almost_equal(gals.zred[0: 3],
                                          np.array([0.10292412, 0.19617805, 0.13324176]))
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir, True)


if __name__=='__main__':
    unittest.main()
