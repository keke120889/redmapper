import unittest, os
import numpy.testing as testing
import numpy as np

import redmapper

class ReadConfigTestCase(unittest.TestCase):
    
    def runTest(self):
        file_name = 'testconfig.yaml'
        file_path = 'data_for_tests'

        confdict = redmapper.config.Configuration('%s/%s' % (file_path, file_name))

        self.assertEqual(confdict.galfile,'./data_for_tests/pixelized/dr8_catalog_test_master_table.fit')
        self.assertEqual(confdict.specfile,'./data_for_tests/dr8_test_spec.fit')
        self.assertEqual(confdict.parfile,'./data_for_tests/test_dr8_pars.fit')
        self.assertEqual(confdict.bkgfile,'./data_for_tests/dr8_bkg.fit')
        self.assertEqual(confdict.zrange,[0.05,0.60])
        self.assertEqual(confdict.outbase,"dr8_testing")
        self.assertEqual(confdict.chisq_max,20.0)
        self.assertEqual(confdict.lval_reference,0.2)
        self.assertEqual(confdict.mask_mode,0)
        # all other values in the confdict are None

if __name__=='__main__':
    unittest.main()
