import unittest
import numpy.testing as testing
import numpy as np

import redmapper

class SplineTestCase(unittest.TestCase):
    def runTest(self):
        # create test data
        xx = np.array([0.05,0.15,0.25,0.35,0.45,0.60],dtype=np.float64)
        yy = np.array([15.68556786,17.98072052,18.93479919,19.67167091,19.79622269,20.11798096], dtype=np.float64)

        # will want to add error checking and exceptions to CubicSpline

        spl=redmapper.utilities.CubicSpline(xx, yy)
        vals=spl(np.array([0.01, 0.44, 0.55, 0.665]))

        testing.assert_almost_equal(vals,np.array([14.6480,19.7928,19.9738,20.3013],dtype=np.float64))
        


# copy this for a new utility test
class UtilityTemplateTestCase(unittest.TestCase):
    def runTest(self):
        pass
    
    
if __name__=='__main__':
    unittest.main()
