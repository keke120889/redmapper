import unittest
import numpy.testing as testing
import numpy as np

import redmapper

class ReadConfigTestCase(unittest.TestCase):
    def runTest(self):
        file_name = 'testconfig.yaml'
        file_path = 'data'

        confdict = redmapper.config.read_config('%s/%s' % (file_path, file_name))

        self.assertEqual(confdict['galfile'],'./data/pixelized/dr8_catalog_test_master_table.fit')
        self.assertEqual(confdict['specfile'],'./data/dr8_test_spec.fit')
        # and more...



if __name__=='__main__':
    unittest.main()
