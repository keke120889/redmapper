from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import unittest
import numpy.testing as testing
import numpy as np
import fitsio
from numpy import random

from redmapper.fitters import EcgmmFitter
from redmapper.fitters import MedZFitter
from redmapper.fitters import RedSequenceFitter, RedSequenceOffDiagonalFitter
from redmapper.fitters import CorrectionFitter
from redmapper.utilities import make_nodes, CubicSpline

class FitterTestCase(unittest.TestCase):
    def test_ecgmm(self):
        random.seed(seed=1000)

        # Test Ecgmm Fitter

        file_path = 'data_for_tests'
        ecgmm_testdata_filename = 'test_ecgmm.fit'

        ecgmmdata, hdr = fitsio.read(file_path + '/' + ecgmm_testdata_filename, ext=1, header=True)

        ecfitter = EcgmmFitter(ecgmmdata['DELTA'], ecgmmdata['GALCOLOR_ERR'])
        wt, mu, sigma = ecfitter.fit([0.2], [-0.5, 0.0], [0.2, 0.05], offset=0.5)

        testing.assert_almost_equal(wt, [0.56762756, 0.43237244], 3)
        testing.assert_almost_equal(mu, [-0.3184651, -0.1168626], 3)
        testing.assert_almost_equal(sigma, [0.15283837, 0.04078598], 3)

    def test_make_nodes(self):
        # Test make_nodes
        nodes = make_nodes([0.1,0.65], 0.05)
        testing.assert_almost_equal(nodes, [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
        nodes = make_nodes([0.1, 0.65], 0.1)
        testing.assert_almost_equal(nodes, [0.1, 0.2, 0.3, 0.4, 0.5, 0.65])
        nodes = make_nodes([0.1, 0.65], 0.1, maxnode=0.4)
        testing.assert_almost_equal(nodes, [0.1, 0.2, 0.3, 0.4, 0.65])

    def test_red_sequence_fitting(self):
        # Test Median z Fitter
        file_path = 'data_for_tests'

        nodes = make_nodes([0.1, 0.2], 0.05)
        fit_testdata_filename = 'test_rsfit.fit'

        fitdata = fitsio.read(file_path + '/' + fit_testdata_filename, ext=1)

        mzfitter = MedZFitter(nodes, fitdata['Z'], fitdata['GALCOLOR'])
        p0 = np.array([1.0, 1.1, 1.2])
        medpars = mzfitter.fit(p0)

        testing.assert_almost_equal(medpars, [0.94603754, 1.09285003, 1.27060679], 5)

        # Test median scatter fit
        spl = CubicSpline(nodes, medpars)
        m = spl(fitdata['Z'])

        mzfitter = MedZFitter(nodes, fitdata['Z'], np.abs(fitdata['GALCOLOR'] - m))
        p0 = np.array([0.05, 0.05, 0.05])
        medscatpars = mzfitter.fit(p0)

        testing.assert_almost_equal(medscatpars, [0.02407928, 0.02745547, 0.03131481], 5)

        # And the red sequence fitter part
        rsfitter = RedSequenceFitter(nodes, fitdata['Z'], fitdata['GALCOLOR'], fitdata['GALCOLOR_ERR'], use_scatter_prior=False)
        p0_mean = medpars
        p0_slope = np.zeros(nodes.size)
        p0_scatter = medscatpars * 1.4826
        meanpars, = rsfitter.fit(p0_mean, p0_slope, p0_scatter, fit_mean=True)

        testing.assert_almost_equal(meanpars, [0.94867445, 1.09166076, 1.26180069], 5)

        # Second, just the scatter
        p0_mean = meanpars
        scatpars, = rsfitter.fit(p0_mean, p0_slope, p0_scatter, fit_scatter=True)

        testing.assert_almost_equal(scatpars, [0.03197, 0.035, 0.0406], 5)

        # Third, just the slope ... need to write this!
        p0_scatter = scatpars
        # If we don't have the dmags then this will raise a ValueError
        self.assertRaises(ValueError, rsfitter.fit, p0_mean, p0_slope, p0_scatter, fit_slope=True)

        rsfitter = RedSequenceFitter(nodes, fitdata['Z'], fitdata['GALCOLOR'], fitdata['GALCOLOR_ERR'], dmags=fitdata['REFMAG'] - np.median(fitdata['REFMAG']))
        slopepars, = rsfitter.fit(p0_mean, p0_slope, p0_scatter, fit_slope=True)

        # These values seem reasonable, but this is just a toy test
        testing.assert_almost_equal(slopepars, [-0.00830984,-0.01538641, -0.02372566], 5)

        # Fourth, combined all 3
        # this test is a lot slower than the others, but still reasonable.
        p0_slope = slopepars
        meanpars2, slopepars2, scatpars2 = rsfitter.fit(p0_mean, p0_slope, p0_scatter, fit_mean=True, fit_slope=True, fit_scatter=True)

        testing.assert_almost_equal(meanpars2, [0.94397328, 1.09101342, 1.26557143], 5)
        testing.assert_almost_equal(slopepars2, [-0.01229, -0.0158, -0.02707], 5)
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
        trunc = nsig * mscat[ok]
        rsfitter = RedSequenceFitter(nodes, fitdata['Z'][ok],
                                     fitdata['GALCOLOR'][ok], fitdata['GALCOLOR_ERR'][ok],
                                     dmags=fitdata['REFMAG'][ok] - np.median(fitdata['REFMAG'][ok]),
                                     trunc=trunc)
        meanpars3, = rsfitter.fit(p0_mean, p0_slope, p0_scatter, fit_mean=True)

        testing.assert_almost_equal(meanpars3, [0.93789514, 1.09257947, 1.27767928], 5)

        # Finally, refit with scatter prior...
        rsfitter._use_scatter_prior = True
        meanpars3, slopepars3, scatpars3 = rsfitter.fit(p0_mean, p0_slope, p0_scatter, fit_mean=True, fit_slope=True, fit_scatter=True)

        testing.assert_almost_equal(meanpars3, [0.93834492, 1.09276382, 1.27604319], 5)
        testing.assert_almost_equal(slopepars3, [-0.01054824, -0.01296713, -0.01605626], 5)
        testing.assert_almost_equal(scatpars3, [0.03453208, 0.04054536, 0.03832689], 5)

    def test_off_diagonal_fitter(self):
        file_path = 'data_for_tests'

        # And test the off-diagonal fitter
        offdiag_testdata_filename = 'test_offdiag_values.fit'
        fitdata = fitsio.read(file_path + '/' + offdiag_testdata_filename, ext=1)

        odfitter = RedSequenceOffDiagonalFitter(fitdata['NODES'][0],
                                                fitdata['Z'][0],
                                                fitdata['D1'][0],
                                                fitdata['D2'][0],
                                                fitdata['S1'][0],
                                                fitdata['S2'][0],
                                                fitdata['MAGERR'][0],
                                                fitdata['J'][0],
                                                fitdata['K'][0],
                                                fitdata['PI'][0],
                                                fitdata['BI'][0],
                                                fitdata['COVMAT_PRIOR'][0],
                                                min_eigenvalue=fitdata['MIN_EIGENVALUE'][0])

        # also need to check inversion...
        full_covmats = np.zeros((4, 4, fitdata['NODES'][0].size))
        for i in xrange(4):
            full_covmats[i, i, :] = fitdata['SIGMA'][0][i, :]**2.

        p0 = np.array([0.0, 0.0])
        pars = odfitter.fit(p0, full_covmats=full_covmats)

        testing.assert_almost_equal(pars, fitdata['RVALS'][0], 2)
        testing.assert_almost_equal(pars, [0.594956, 0.65475681], 5)

    def test_red_sequence_fitter_with_probs(self):
        pass

    def test_zred_correction_fitter(self):

        # Test the zred correction fitter.
        # No tests right now for the slope fitting, because we don't
        # use it in the IDL code.
        file_path = 'data_for_tests'

        corr_testdata_filename = 'test_zredcorr_values.fit'
        fitdata = fitsio.read(file_path + '/' + corr_testdata_filename, ext=1)

        corrfitter = CorrectionFitter(fitdata['NODES'][0],
                                      fitdata['Z'][0],
                                      fitdata['DZ'][0],
                                      fitdata['DZ_ERR'][0],
                                      slope_nodes=fitdata['SNODES'][0],
                                      probs=fitdata['PI'][0],
                                      dmags=fitdata['DMAG'][0],
                                      ws=fitdata['W'][0])

        p0_mean = np.zeros(fitdata['NODES'][0].size)
        p0_slope = np.zeros(fitdata['SNODES'][0].size)
        p0_r = np.ones(fitdata['SNODES'][0].size)
        p0_bkg = np.zeros(fitdata['SNODES'][0].size) + 0.01
        pars_mean, = corrfitter.fit(p0_mean, p0_slope, p0_r, p0_bkg, fit_mean=True)

        testing.assert_almost_equal(pars_mean, np.array([0.00484127, 0.00159642, -0.00019032]), 5)

        p0_mean = pars_mean
        pars_r, = corrfitter.fit(p0_mean, p0_slope, p0_r, p0_bkg, fit_r=True)

        testing.assert_almost_equal(pars_r, np.array([0.7733483, 0.40378139]), 5)

        p0_r = pars_r
        pars_bkg, = corrfitter.fit(p0_mean, p0_slope, p0_r, p0_bkg, fit_bkg=True)

        testing.assert_almost_equal(pars_bkg, np.array([6.10995610e-10,   3.74807289e-04]), 5)
        p0_bkg = pars_bkg
        pars_mean, pars_r, pars_bkg = corrfitter.fit(p0_mean, p0_slope, p0_r, p0_bkg, fit_mean=True, fit_r=True, fit_bkg=True)

        testing.assert_almost_equal(pars_mean, fitdata['CVALS'][0], 4)
        testing.assert_almost_equal(pars_r, fitdata['RVALS'][0], 3)
        testing.assert_almost_equal(pars_bkg, fitdata['BKG_CVALS'][0], 5)



if __name__=='__main__':
    unittest.main()
