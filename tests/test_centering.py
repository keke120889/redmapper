import unittest
import numpy.testing as testing
import numpy as np
import fitsio
from numpy import random

from redmapper.cluster import Cluster
from redmapper.configuration import Configuration
from redmapper.centering import CenteringWcenZred
from redmapper.galaxy import GalaxyCatalog
from redmapper.redsequence import RedSequenceColorPar
from redmapper.background import Background
from redmapper.background import ZredBackground
from redmapper.zlambda import ZlambdaCorrectionPar


class CenteringWcenZredTestCase(unittest.TestCase):
    def runTest(self):
        file_path = 'data_for_tests'

        cluster = Cluster()

        conf_filename = 'testconfig.yaml'
        cluster.config = Configuration(file_path + '/' + conf_filename)

        filename = 'test_wcen_zred_data.fit'

        tempcat = fitsio.read(file_path + '/' + filename, ext=1)

        temp_neighbors = np.zeros(tempcat[0]['RAS'].size,
                                  dtype = [('RA', 'f8'),
                                           ('DEC', 'f8'),
                                           ('DIST', 'f4'),
                                           ('R', 'f4'),
                                           ('P', 'f4'),
                                           ('PFREE', 'f4'),
                                           ('PMEM', 'f4'),
                                           ('MAG', 'f4', 5),
                                           ('MAG_ERR', 'f4', 5),
                                           ('REFMAG', 'f4'),
                                           ('REFMAG_ERR', 'f4'),
                                           ('CHISQ', 'f4'),
                                           ('ZRED', 'f4'),
                                           ('ZRED_E', 'f4'),
                                           ('ZRED_CHISQ', 'f4')])
        temp_neighbors['RA'] = tempcat[0]['RAS']
        temp_neighbors['DEC'] = tempcat[0]['DECS']
        temp_neighbors['R'] = tempcat[0]['R']
        temp_neighbors['P'] = tempcat[0]['PVALS']
        temp_neighbors['PFREE'] = tempcat[0]['WVALS']
        temp_neighbors['PMEM'] = tempcat[0]['WTVALS']
        temp_neighbors['REFMAG'] = tempcat[0]['REFMAG_TOTAL']
        temp_neighbors['ZRED'] = tempcat[0]['GZREDS']
        temp_neighbors['ZRED_E'] = tempcat[0]['GZREDE']
        temp_neighbors['ZRED_CHISQ'] = tempcat[0]['GCHISQ']

        temp_neighbors['DIST'] = tempcat[0]['R'] / (np.radians(1.) * cluster.config.cosmo.Da(0, tempcat[0]['ZCLUSTER']))

        neighbors = GalaxyCatalog(temp_neighbors)
        cluster.set_neighbors(neighbors)

        zred_filename = 'test_dr8_pars.fit'
        cluster.zredstr = RedSequenceColorPar(file_path + '/' + zred_filename, fine=True)

        bkg_filename = 'test_bkg.fit'
        cluster.bkg = Background('%s/%s' % (file_path, bkg_filename))
        cluster.zredbkg = ZredBackground('%s/%s' % (file_path, bkg_filename))

        cluster.redshift = tempcat[0]['ZCLUSTER']
        cluster.ra = tempcat[0]['RAC']
        cluster.dec = tempcat[0]['DECC']
        cluster.r_lambda = 1.0 * (tempcat[0]['LAMBDA'] / 100.0)**0.2
        cluster.Lambda = tempcat[0]['LAMBDA']
        cluster.scaleval = tempcat[0]['SCALEVAL']

        corr_filename = 'test_dr8_zlambdacorr.fit'
        zlambda_corr = ZlambdaCorrectionPar(file_path + '/' + corr_filename, 30.0)

        # And the meat of it...

        cent = CenteringWcenZred(cluster, zlambda_corr=zlambda_corr)
        cent.find_center()

        testing.assert_almost_equal(cent.p_cen, tempcat[0]['PCEN'][tempcat[0]['GOOD']], 5)
        testing.assert_almost_equal(cent.q_cen, tempcat[0]['QCEN'][tempcat[0]['GOOD']], 4)
        testing.assert_almost_equal(cent.p_sat, tempcat[0]['PSAT'], 4)
        testing.assert_almost_equal(cent.p_fg, tempcat[0]['PFG'], 4)
        testing.assert_array_equal(cent.index, tempcat[0]['USE'][tempcat[0]['GOOD']])

        # Also will need to test BCG centering!
        
