import unittest
import numpy.testing as testing
import numpy as np

import redmapper

class SplineTestCase(unittest.TestCase):
    def runTest(self):
        # create test data
        # these numbers are from redMaPPer 6.3.1, DR8
        xx = np.array([0.05,0.15,0.25,0.35,0.45,0.60],dtype=np.float64)
        yy = np.array([15.685568,17.980721,18.934799,19.671671,19.796223,20.117981], dtype=np.float64)

        # will want to add error checking and exceptions to CubicSpline

        spl=redmapper.utilities.CubicSpline(xx, yy)
        vals=spl(np.array([0.01, 0.44, 0.55, 0.665]))

        # these numbers are also from redMaPPer 6.3.1, DR8
        testing.assert_almost_equal(vals,np.array([14.648017,19.792828,19.973761,20.301322],dtype=np.float64),decimal=6)
        

class MStarTestCase(unittest.TestCase):
    def runTest(self):
        # make sure invalid raises proper exception
        self.assertRaises(IOError,redmapper.utilities.MStar,'blah','junk')
                          
        # make an SDSS test...
        ms = redmapper.utilities.MStar('sdss','i03')

        mstar = ms([0.1,0.2,0.3,0.4,0.5])
        # test against IDL...
        testing.assert_almost_equal(mstar,np.array([16.2375,17.8500,18.8281,19.5878,20.1751]),decimal=4)
        # and against regressions...
        testing.assert_almost_equal(mstar,np.array([ 16.23748776,  17.85000035,  18.82812871,  19.58783337,  20.17514801]))

        
class FunctionsTestCase(unittest.TestCase):
    def runTest(self):
        ra,dec = 40.1234, 55.9876
        testing.assert_almost_equal(redmapper.utilities.astro_to_sphere(ra,dec),np.array([ 0.5936,  0.7003]),decimal=4)
        #Need a test for chisq_pdf
        #TO DO


# copy this for a new utility test
class UtilityTemplateTestCase(unittest.TestCase):
    def runTest(self):
        pass


if __name__=='__main__':
    unittest.main()
