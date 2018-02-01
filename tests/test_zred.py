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

        # Getting closer... there are some offsets but may be okay.
        # Need to look at brighter ones...
        # zred_uncorr (small sequence needs to be looked at)
        # zred_uncorr_e (looks good)
        # lkhd (mostly good, some outliers to check)
        # chisq (mostly good, some outliers to check)

        # pretty sure that the differences are due to problems in the IDL code...
        

        # a problematic one...
        #test,=np.where((galaxies_input.zred_uncorr > 0.07) &
        #               (galaxies_input.zred_uncorr < 0.072) &
        #               (galaxies.zred_uncorr < 0.062))

        #print(test[0])
        
        
        #test,=np.where((galaxies_input.zred_uncorr_e > 0.04) &
        #               (galaxies_input.zred_uncorr_e < 0.045) &
        #               (galaxies.zred_uncorr_e < 0.035) &
        #               (galaxies.chisq < 3.0))#

        #print(test[0])

        

        # and speed tests...
        # Hot spots:
        #  catalog.py(__getattribute__)
        #    600000 calls, this hurts, may have to revisit
        #  zred_color.py (compute_zred) a bunch of time
        #  interpolate.py (_call_linear) BLAH
        #  _chisq_dist_pywrap (compute).  Speed test this
        #  _calculate_lndist (?!)
        #  calculate_chisq in redsequence (!)
