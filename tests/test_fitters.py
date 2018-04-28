from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import unittest
import numpy.testing as testing
import numpy as np
import fitsio
from numpy import random

from redmapper.fitters import EcgmmFitter
from redmapper.fitters import MedZFitter
from redmapper.fitters import RedSequenceFitter
from redmapper.utilities import make_nodes, CubicSpline

class FitterTestCase(unittest.TestCase):
    def runTest(self):
        random.seed(seed=1000)

        # Test Ecgmm Fitter

        file_path = 'data_for_tests'
        ecgmm_testdata_filename = 'test_ecgmm.fit'

        ecgmmdata, hdr = fitsio.read(file_path + '/' + ecgmm_testdata_filename, ext=1, header=True)

        ecfitter = EcgmmFitter(ecgmmdata['DELTA'], ecgmmdata['GALCOLOR_ERR'])
        wt, mu, sigma = ecfitter.fit([0.2], [-0.5, 0.0], [0.2, 0.05], offset=0.5)
        print(wt)

        #testing.assert_almost_equal(wt, [0.56762756, 0.43237244], 5)
        #testing.assert_almost_equal(mu, [-0.3184651, -0.1168626], 5)
        #testing.assert_almost_equal(sigma, [0.15283837, 0.04078598], 5)

        # Test make_nodes
        nodes = make_nodes([0.1,0.65], 0.05)
        testing.assert_almost_equal(nodes, [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
        nodes = make_nodes([0.1, 0.65], 0.1)
        testing.assert_almost_equal(nodes, [0.1, 0.2, 0.3, 0.4, 0.5, 0.65])
        nodes = make_nodes([0.1, 0.65], 0.1, maxnode=0.4)
        testing.assert_almost_equal(nodes, [0.1, 0.2, 0.3, 0.4, 0.65])

        # Test Median z Fitter
        nodes = make_nodes([0.1, 0.2], 0.05)
        fit_testdata_filename = 'test_rsfit.fit'

        fitdata = fitsio.read(file_path + '/' + fit_testdata_filename, ext=1)

        mzfitter = MedZFitter(nodes, fitdata['Z'], fitdata['GALCOLOR'])
        p0 = np.array([1.0, 1.1, 1.2])
        medpars = mzfitter.fit(p0)

        testing.assert_almost_equal(medpars, [0.94604157, 1.09283561, 1.27056759], 5)

        # Test median scatter fit
        spl = CubicSpline(nodes, medpars)
        m = spl(fitdata['Z'])

        mzfitter = MedZFitter(nodes, fitdata['Z'], np.abs(fitdata['GALCOLOR'] - m))
        p0 = np.array([0.05, 0.05, 0.05])
        medscatpars = mzfitter.fit(p0)

        testing.assert_almost_equal(medscatpars, [0.02410464, 0.02747469, 0.03136519], 5)

        # Test Red Sequence Fitter
        # First, just the mean
        rsfitter = RedSequenceFitter(nodes, fitdata['Z'], fitdata['GALCOLOR'], fitdata['GALCOLOR_ERR'])
        p0_mean = medpars
        p0_slope = np.zeros(nodes.size)
        p0_scatter = medscatpars * 1.4826
        meanpars, = rsfitter.fit(p0_mean, p0_slope, p0_scatter, fit_mean=True)

        testing.assert_almost_equal(meanpars, [0.94867445, 1.09166076, 1.26180069], 5)

        # Second, just the scatter
        p0_mean = meanpars
        scatpars, = rsfitter.fit(p0_mean, p0_slope, p0_scatter, fit_scatter=True)

        testing.assert_almost_equal(scatpars, [0.03197381, 0.03500057, 0.04059801], 5)

        # Third, just the slope ... need to write this!
        p0_scatter = scatpars
        # If we don't have the dmags then this will raise a ValueError
        self.assertRaises(ValueError, rsfitter.fit, p0_mean, p0_slope, p0_scatter, fit_slope=True)

        rsfitter = RedSequenceFitter(nodes, fitdata['Z'], fitdata['GALCOLOR'], fitdata['GALCOLOR_ERR'], dmags=fitdata['REFMAG'] - np.median(fitdata['REFMAG']))
        slopepars, = rsfitter.fit(p0_mean, p0_slope, p0_scatter, fit_slope=True)

        # These values seem reasonable, but this is just a toy test
        testing.assert_almost_equal(slopepars, [-0.00832335, -0.01538893, -0.02370383], 5)

        # Fourth, combined all 3
        # this test is a lot slower than the others, but still reasonable.
        p0_slope = slopepars
        meanpars2, slopepars2, scatpars2 = rsfitter.fit(p0_mean, p0_slope, p0_scatter, fit_mean=True, fit_slope=True, fit_scatter=True)

        testing.assert_almost_equal(meanpars2, [0.94397328, 1.09101342, 1.26557143], 5)
        testing.assert_almost_equal(slopepars2, [-0.01229146, -0.0158041 , -0.02707032], 5)
        testing.assert_almost_equal(scatpars2, [0.03115847, 0.03430587, 0.03987611], 5)

        # Fifth, fit_mean with truncation
        p0_mean = meanpars2
        p0_slope = slopepars2
        p0_scatter = scatpars2

        spl = CubicSpline(nodes, medpars)
        m = spl(fitdata['Z'])
        spl = CubicSpline(nodes, medscatpars)
        mscat = spl(fitdata['Z']) * 1.4826

        nsig = 1.5
        ok, = np.where(np.abs(fitdata['GALCOLOR'] - m) < (nsig * mscat))
        trunc = np.zeros((2, ok.size))
        trunc[0, :] = m[ok] - nsig * mscat[ok]
        trunc[1, :] = m[ok] + nsig * mscat[ok]
        rsfitter = RedSequenceFitter(nodes, fitdata['Z'][ok],
                                     fitdata['GALCOLOR'][ok], fitdata['GALCOLOR_ERR'][ok],
                                     dmags=fitdata['REFMAG'][ok] - np.median(fitdata['REFMAG'][ok]),
                                     trunc=trunc)
        meanpars3, = rsfitter.fit(p0_mean, p0_slope, p0_scatter, fit_mean=True)

        testing.assert_almost_equal(meanpars3, [0.93506074, 1.09253144, 1.27999684], 5)

if __name__=='__main__':
    unittest.main()

