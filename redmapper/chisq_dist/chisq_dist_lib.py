from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import numpy as np
from . import _chisq_dist_pywrap

class ChisqDist(object):
    pass


def compute_chisq(covmat, c, slope, pivotmag, refmag, magerr, color, refmagerr=None, lupcorr=None, calc_chisq=True, calc_lkhd=False, nophotoerr=False):

    _covmat = covmat.astype('f8')
    _c = c.astype('f8')
    _slope = slope.astype('f8')
    _pivotmag = np.atleast_1d(pivotmag.astype('f8'))
    _refmag = np.atleast_1d(refmag.astype('f8'))
    _magerr = magerr.astype('f8')
    _color = color.astype('f8')

    if (refmagerr is None):
        _refmagerr = np.zeros(refmag.size,dtype='f8')
    else:
        _refmagerr = np.atleast_1d(refmagerr.astype('f8'))

    # need to figure out the mode here...
    if (c.ndim == 1) :
        # mode 0: one redshift, many galaxies
        mode = 0
        ncol = c.size
    elif (c.ndim == 2):
        if (len(magerr.shape) == 1):
            # nz mode
            # c is matrix, nz x ncol
            mode = 1
        else :
                # ngal_nz mode
            # c is matrix, ngal x ncol
            mode = 2
        ncol = c.shape[1]
    else:
        raise ValueError("c must be 1D or 2D")

    #self.ncol = self.c.shape[0]
    nmag = ncol + 1

    # common
    if (slope.ndim != c.ndim) :
        raise ValueError("slope must be the same dim as c")

    # and check for each mode.  New!
    # note that we've already checked "c" (it's the reference)
    if (mode == 0):
        # only one redshift for mode 0
        nz = 1

        # this is one redshift, many galaxies
        if (slope.shape[0] != ncol):
            raise ValueError("slope must have ncol elements for mode 0")
        if (_pivotmag.shape[0] != 1):
            raise ValueError("pivotmag must have 1 element for mode 0")
        if (covmat.ndim != 2):
            raise ValueError("covmat must be 2 dimensions for mode 0")
        if (covmat.shape[0] != ncol or covmat.shape[1] != ncol):
            raise ValueError("covmat must be ncol x ncol for mode 0")

        # ngal is number of refmag entries
        ngal = _refmag.size

        # dumb
        #if (self.c.shape[1] != self.ngal):
        #    raise ValueError("c must have ncol x ngal elements for mode 0")
        #if (self.slope.shape[1] != self.ngal):
        #    raise ValueError("slope must have ncol x ngal elements for mode 0")

        if (magerr.ndim != 2):
            raise ValueError("magerr must be 2D for mode 0")
        if (magerr.shape[0] != ngal) or (magerr.shape[1] != nmag):
            raise ValueError("magerr must be ngal x nmag for mode 0")
        if (color.ndim != 2):
            raise ValueError("color must be 2D for mode 0")
        if (color.shape[0] != ngal) or (color.shape[1] != ncol):
            raise ValueError("color must be ngal x ncol for mode 0")

        if (lupcorr is None):
            _lupcorr = np.zeros((ngal, ncol),dtype='f8')
        else :
            _lupcorr = lupcorr.astype('f8')

        if (_lupcorr.shape[0] != ngal) or (_lupcorr.shape[1] != ncol):
            raise ValueError("lupcorr must be ncol x ngal for mode 0")

        if (_refmagerr.size != ngal) :
            raise ValueError("refmagerr must be ngal elements for mode 0")

    elif (mode == 1):
        # this is many redshifts, one galaxy

        # nz is the second dimension of c (nope)
        nz = c.shape[0]
        ngal = 1

        if (slope.shape[0] != nz or slope.shape[1] != ncol):
            raise ValueError("slope must have ncol x nz elements for mode 1")
        if (_pivotmag.size != nz):
            raise ValueError("pivotmag must be nz elements for mode 1")

        if (covmat.ndim != 3):
            raise ValueError("covmat must be 3 dimensions for mode 1")
        if (covmat.shape[2] != nz):
            raise ValueError("Third dimension of covmat must be nz for mode 1")        
        if (covmat.shape[0] != ncol or covmat.shape[1] != ncol):
            raise ValueError("covmat must be ncol x ncol x nz for mode 1")

        if (magerr.ndim != 1):
            raise ValueError("magerr must be 1D for mode 1")
        if (magerr.size != nmag):
            raise ValueError("magerr must be nmag length for mode 1")

        if (color.ndim != 1):
            raise ValueError("color must be 1D for mode 1")
        if (color.size != ncol):
            raise ValueError("color must be ncol length for mode 1")

        if (lupcorr is None):
            _lupcorr = np.zeros((nz, ncol),dtype='f8')
        else :
            _lupcorr = lupcorr.astype('f8')

        if (_lupcorr.shape[0] != nz) or (_lupcorr.shape[1] != ncol):
            raise ValueError("lupcorr must be ncol x nz for mode 1")

        if (_refmagerr.shape[0] != 1) :
            raise ValueError("refmagerr must be 1 elements for mode 1")

    elif (mode == 2):
        # nz is the second dimension of c (I think)
        nz = c.shape[0]
        ngal = nz

        if (slope.shape[0] != nz or slope.shape[1] != ncol):
            raise ValueError("slope must have ncol x nz elements for mode 2")
        if (_pivotmag.size != nz):
            raise ValueError("pivotmag must be nz elements for mode 2")
        if (covmat.ndim != 3):
            raise ValueError("covmat must be 3 dimensions for mode 2")
        if (covmat.shape[2] != nz):
            raise ValueError("Third dimension of covmat must be nz for mode 2")

        if (magerr.ndim != 2):
            raise ValueError("magerr must be 2D for mode 2")
        if (magerr.shape[0] != ngal) or (magerr.shape[1] != nmag):
            raise ValueError("magerr must be ngal x nmag for mode 2")
        if (color.ndim != 2):
            raise ValueError("color must be 2D for mode 2")
        if (color.shape[0] != ngal) or (color.shape[1] != ncol):
            raise ValueError("color must be ngal x ncol for mode 2")

        if (lupcorr is None):
            _lupcorr = np.zeros((ngal, ncol),dtype='f8')
        else :
            _lupcorr = lupcorr.astype('f8')

        if (_lupcorr.shape[0] != ngal) or (_lupcorr.shape[1] != ncol):
            raise ValueError("lupcorr must be ngal x ncol for mode 2")

        if (_refmagerr.size != ngal) :
            raise ValueError("refmagerr must be ngal elements for mode 2")
    else:
        raise ValueError("Illegal mode (unpossible)")

    _chisq_dist = _chisq_dist_pywrap.ChisqDist(mode,
                                               ngal,
                                               nz,
                                               ncol,
                                               _covmat,
                                               _c,
                                               _slope,
                                               _pivotmag,
                                               _refmag,
                                               _refmagerr,
                                               _magerr,
                                               _color,
                                               _lupcorr)

    if calc_chisq:
        chisq = _chisq_dist.compute(True, nophotoerr)

    if calc_lkhd:
        lkhd = _chisq_dist.compute(False, nophotoerr)

    if calc_chisq and not calc_lkhd:
        return chisq
    elif not calc_chisq and calc_lkhd:
        return lkhd
    else:
        return (chisq, lkhd)

