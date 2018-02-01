import unittest
import numpy.testing as testing
import numpy as np
import fitsio
import copy
from numpy import random
import time

from redmapper.configuration import Configuration
from redmapper.galaxy import GalaxyCatalog
from redmapper.redsequence import RedSequenceColorPar
from redmapper.zred_color import ZredColor

class ZredTestCase(unittest.TestCase):
    """
    """

    def runTest(self):

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
