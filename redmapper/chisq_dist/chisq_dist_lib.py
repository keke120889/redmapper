import numpy as np
from . import _chisq_dist_pywrap

class ChisqDist(object):
    def __init__(self, covmat, c, slope, pivotmag, refmag, magerr, color, refmagerr=None, lupcorr=None):
        # ensure all correct length, etc.

        # we'll do that here in Python, and save the c code.
        # Though of course that makes the c code more fragile,
        # but it should always be accessed through here.

        self.covmat = covmat.astype('f8')
        self.c = c.astype('f8')
        self.slope = slope.astype('f8')
        self.pivotmag = np.atleast_1d(pivotmag.astype('f8'))
        self.refmag = np.atleast_1d(refmag.astype('f8'))
        self.magerr = magerr.astype('f8')
        self.color = color.astype('f8')

        if (refmagerr is None):
            self.refmagerr = np.zeros(refmag.size,dtype='f8')
        else:
            self.refmagerr = np.atleast_1d(refmagerr.astype('f8'))


        # need to figure out the mode here...
        if (self.c.ndim == 1) :
            # mode 0: one redshift, many galaxies
            self.mode = 0
            self.ncol = self.c.size
        elif (self.c.ndim == 2):
            if (len(self.magerr.shape) == 1):
                # nz mode
                # c is matrix, nz x ncol
                self.mode = 1
            else :
                # ngal_nz mode
                # c is matrix, ngal x ncol
                self.mode = 2
            self.ncol = self.c.shape[1]
        else:
            raise ValueError("c must be 1D or 2D")

        #self.ncol = self.c.shape[0]
        self.nmag = self.ncol + 1

        # common
        if (self.slope.ndim != self.c.ndim) :
            raise ValueError("slope must be the same dim as c")

        # and check for each mode.  New!
        # note that we've already checked "c" (it's the reference)
        if (self.mode == 0):
            # only one redshift for mode 0
            self.nz = 1

            # this is one redshift, many galaxies
            if (self.slope.shape[0] != self.ncol):
                raise ValueError("slope must have ncol elements for mode 0")
            if (self.pivotmag.shape[0] != 1):
                raise ValueError("pivotmag must have 1 element for mode 0")
            if (self.covmat.ndim != 2):
                raise ValueError("covmat must be 2 dimensions for mode 0")
            if (self.covmat.shape[0] != self.ncol or self.covmat.shape[1] != self.ncol):
                raise ValueError("covmat must be ncol x ncol for mode 0")

            # ngal is number of refmag entries
            self.ngal = self.refmag.size

            # dumb
            #if (self.c.shape[1] != self.ngal):
            #    raise ValueError("c must have ncol x ngal elements for mode 0")
            #if (self.slope.shape[1] != self.ngal):
            #    raise ValueError("slope must have ncol x ngal elements for mode 0")

            if (self.magerr.ndim != 2):
                raise ValueError("magerr must be 2D for mode 0")
            if (self.magerr.shape[0] != self.ngal) or (self.magerr.shape[1] != self.nmag):
                raise ValueError("magerr must be ngal x nmag for mode 0")
            if (self.color.ndim != 2):
                raise ValueError("color must be 2D for mode 0")
            if (self.color.shape[0] != self.ngal) or (self.color.shape[1] != self.ncol):
                raise ValueError("color must be ngal x ncol for mode 0")

            if (lupcorr is None):
                self.lupcorr = np.zeros((self.ngal, self.ncol),dtype='f8')
            else :
                self.lupcorr = lupcorr.astype('f8')

            if (self.lupcorr.shape[0] != self.ngal) or (self.lupcorr.shape[1] != self.ncol):
                raise ValueError("lupcorr must be ncol x ngal for mode 0")

            if (self.refmagerr.size != self.ngal) :
                raise ValueError("refmagerr must be ngal elements for mode 0")

        elif (self.mode == 1):
            # this is many redshifts, one galaxy

            # nz is the second dimension of c (nope)
            self.nz = self.c.shape[0]
            self.ngal = 1

            if (self.slope.shape[0] != self.nz or self.slope.shape[1] != self.ncol):
                raise ValueError("slope must have ncol x nz elements for mode 1")
            if (self.pivotmag.size != self.nz):
                raise ValueError("pivotmag must be nz elements for mode 1")

            if (self.covmat.ndim != 3):
                raise ValueError("covmat must be 3 dimensions for mode 1")
            if (self.covmat.shape[2] != self.nz):
                raise ValueError("Third dimension of covmat must be nz for mode 1")        
            if (self.covmat.shape[0] != self.ncol or self.covmat.shape[1] != self.ncol):
                raise ValueError("covmat must be ncol x ncol x nz for mode 1")

            if (self.magerr.ndim != 1):
                raise ValueError("magerr must be 1D for mode 1")
            if (self.magerr.size != self.nmag):
                raise ValueError("magerr must be nmag length for mode 1")

            if (self.color.ndim != 1):
                raise ValueError("color must be 1D for mode 1")
            if (self.color.size != self.ncol):
                raise ValueError("color must be ncol length for mode 1")

            if (lupcorr is None):
                self.lupcorr = np.zeros((self.nz, self.ncol),dtype='f8')
            else :
                self.lupcorr = lupcorr.astype('f8')
                
            if (self.lupcorr.shape[0] != self.nz) or (self.lupcorr.shape[1] != self.ncol):
                raise ValueError("lupcorr must be ncol x nz for mode 1")

            if (self.refmagerr.shape[0] != 1) :
                raise ValueError("refmagerr must be 1 elements for mode 1")

        elif (self.mode == 2):
            # nz is the second dimension of c (I think)
            self.nz = self.c.shape[0]
            self.ngal = self.nz
            
            if (self.slope.shape[0] != self.nz or self.slope.shape[1] != self.ncol):
                raise ValueError("slope must have ncol x nz elements for mode 2")
            if (self.pivotmag.size != self.nz):
                raise ValueError("pivotmag must be nz elements for mode 2")
            if (self.covmat.ndim != 3):
                raise ValueError("covmat must be 3 dimensions for mode 2")
            if (self.covmat.shape[2] != self.nz):
                raise ValueError("Third dimension of covmat must be nz for mode 2")
            
            if (self.magerr.ndim != 2):
                raise ValueError("magerr must be 2D for mode 2")
            if (self.magerr.shape[0] != self.ngal) or (self.magerr.shape[1] != self.nmag):
                raise ValueError("magerr must be ngal x nmag for mode 2")
            if (self.color.ndim != 2):
                raise ValueError("color must be 2D for mode 2")
            if (self.color.shape[0] != self.ngal) or (self.color.shape[1] != self.ncol):
                raise ValueError("color must be ngal x ncol for mode 2")

            if (lupcorr is None):
                self.lupcorr = np.zeros((self.ngal, self.ncol),dtype='f8')
            else :
                self.lupcorr = lupcorr.astype('f8')
                
            if (self.lupcorr.shape[0] != self.ngal) or (self.lupcorr.shape[1] != self.ncol):
                raise ValueError("lupcorr must be ngal x ncol for mode 2")

            if (self.refmagerr.size != self.ngal) :
                raise ValueError("refmagerr must be ngal elements for mode 2")
        else:
            raise ValueError("Illegal mode (unpossible)")

                             
        #if (self.mode == 0) :
        #    self.ncalc = self.ngal
        #else :
        #    self.ncalc = self.nz
            
        self._chisq_dist = _chisq_dist_pywrap.ChisqDist(self.mode,
                                                        self.ngal,
                                                        self.nz,
                                                        self.ncol,
                                                        self.covmat,
                                                        self.c,
                                                        self.slope,
                                                        self.pivotmag,
                                                        self.refmag,
                                                        self.refmagerr,
                                                        self.magerr,
                                                        self.color,
                                                        self.lupcorr)

    def compute_chisq(self,chisq_mode=True,nophotoerr=False):
        return self._chisq_dist.compute(chisq_mode,nophotoerr)
        
        
