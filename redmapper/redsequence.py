from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import fitsio
import esutil
import numpy as np
from scipy import interpolate

from .chisq_dist import ChisqDist
from .chisq_dist import compute_chisq
from .catalog import Catalog
from .utilities import CubicSpline, MStar
from .utilities import schechter_pdf

class RedSequenceColorPar(object):
    """
    Class which contains a red-sequence parametrization

    parameters
    ----------
    filename: string
        red sequence parameter file
    zbinsize: float, optional
        interpolation bin size in redshift
    minsig: float, optional
        minimum intrinsic scatter.  Default 0.01
    fine: bool, optional
        finest binning, set in filename header.  Default False
    zrange: float*2, optional
        redshift range [zlo,zhi].  Default from filename header
        (maximum range)

    """
    def __init__(self, filename, zbinsize=None, minsig=0.01, fine=False, zrange=None, config=None, limmag=None):

        if filename is None:
            if config is None:
                raise ValueError("Must have either filename or config")
            if limmag is None:
                limmag = config.limmag_catalog
            else:
                limmag = limmag
            if zrange is None:
                zrange = config.zrange
            alpha = config.calib_lumfunc_alpha
            mstar_survey = config.mstar_survey
            mstar_band = config.mstar_band
            ncol = config.nmag - 1

            # A compromize zbinsize
            zbinsize = 0.001

            has_file = False
        else:
            pars,hdr=fitsio.read(filename, ext=1, header=True, upper=True)
            try:
                if limmag is None:
                    limmag = hdr['LIMMAG']
                else:
                    limmag = limmag
                if (zrange is None):
                    zrange = np.array([hdr['ZRANGE0'],hdr['ZRANGE1']])
                alpha = hdr['ALPHA']
                mstar_survey = hdr['MSTARSUR']
                mstar_band = hdr['MSTARBAN']
                ncol = hdr['NCOL']
            except:
                raise ValueError("Missing field from parameter header.")
            has_file = True

        if len(zrange) != 2:
            raise ValueError("zrange must have 2 elements");

        if zbinsize is None:
            try:
                if fine:
                    zbinsize=hdr['ZBINFINE']
                else:
                    zbinsize=hdr['ZBINCOAR']
            except:
                raise ValueError("Missing field from parameter header.")

        try:
            lowzmode=hdr['LOWZMODE']
        except:
            lowzmode = 0

        nmag = ncol + 1
        self.nmag = nmag

        if has_file:
            bvalues=np.zeros(nmag)
            do_lupcorr = False
            try:
                for i in xrange(nmag):
                    bvalues[i] = hdr['BVALUE%1d' % (i+1)]
                do_lupcorr = True
            except:
                bvalues[:] = 0.0

            try:
                ref_ind = hdr['REF_IND']
            except:
                try:
                    ref_ind = hdr['I_IND']
                except:
                    raise ValueError("Need REF_IND or I_IND")

        nz = np.round((zrange[1]-zrange[0])/zbinsize).astype('i4') #nr of bins
        self.z = zbinsize*np.arange(nz) + zrange[0]                #z bins

        # append a high-end overflow bin
        # for computation this will be the same as the top bin, and at the end
        # we set it to some absurdly large number.
        self.z = np.append(self.z,self.z[self.z.size-1])
        nz=nz+1

        self.zbinsize=zbinsize
        self.zbinscale=int(1./zbinsize)

        ms=MStar(mstar_survey,mstar_band)

        refmagbinsize=0.01
        if (lowzmode):
            refmagrange=np.array([10.0,limmag],dtype='f4')
            # FIXME
            lumrefmagrange=np.array([10.0,ms(zrange[1])-2.5*np.log10(0.1)])
        else:
            refmagrange=np.array([12.0,limmag],dtype='f4')
            lumrefmagrange=np.array([12.0,ms(zrange[1])-2.5*np.log10(0.1)])
        self.refmagbins = np.arange(refmagrange[0], refmagrange[1], refmagbinsize, dtype='f8')
        self.lumrefmagbins=np.arange(lumrefmagrange[0],lumrefmagrange[1],refmagbinsize,dtype='f8')

        # and for fast look-ups...
        self.refmagbins = np.append(self.refmagbins,self.refmagbins[self.refmagbins.size-1])
        self.lumrefmagbins = np.append(self.lumrefmagbins,self.lumrefmagbins[self.lumrefmagbins.size-1])

        self.refmagbinsize = refmagbinsize
        self.refmagbinscale = int(1./refmagbinsize)
        self.refmaginteger = (self.refmagbins*self.refmagbinscale).astype(np.int64)
        self.lumrefmaginteger = (self.lumrefmagbins*self.refmagbinscale).astype(np.int64)

        if has_file:
            # is this an old or new structure?
            if 'PIVOTMAG_Z' in pars.dtype.names:
                refmag_name = 'REFMAG'
                pivotmag_name = 'PIVOTMAG'
            else :
                refmag_name = 'IMAG'
                pivotmag_name = 'REFMAG'

            # mark the extrapolated values
            self.extrapolated = np.zeros(nz,dtype=np.bool_)
            loz,=np.where(self.z < np.min(pars[pivotmag_name+'_Z']))
            hiz,=np.where(self.z > np.max(pars[pivotmag_name+'_Z']))
            if (loz.size > 0) : self.extrapolated[loz] = True
            if (hiz.size > 0) : self.extrapolated[hiz] = True

            # set the pivotmag
            self.pivotmag = np.zeros(self.z.size, dtype=np.float64)
            spl=CubicSpline(pars[0][pivotmag_name+'_Z'],pars[0][pivotmag_name])
            self.pivotmag[:] = spl(self.z)

            # and the max/min refmag
            spl=CubicSpline(pars[0][pivotmag_name+'_Z'], pars[0]['MAX'+refmag_name])
            self.maxrefmag = spl(self.z)
            spl=CubicSpline(pars[0][pivotmag_name+'_Z'], pars[0]['MIN'+refmag_name])
            self.minrefmag = spl(self.z)

            # c/slope
            self.c = np.zeros((nz,ncol),dtype=np.float64)
            self.slope = np.zeros((nz,ncol),dtype=np.float64)
            for j in xrange(ncol):
                jstring='%02d' % (j)
                spl=CubicSpline(pars[0]['Z'+jstring],pars[0]['C'+jstring])
                self.c[:,j] = spl(self.z)
                spl=CubicSpline(pars[0]['ZS'+jstring],pars[0]['SLOPE'+jstring])
                self.slope[:,j] = spl(self.z)

            # sigma/covmat
            self.sigma = np.zeros((ncol,ncol,nz),dtype=np.float64)
            self.covmat = np.zeros((ncol,ncol,nz),dtype=np.float64)

            # diagonals
            for j in xrange(ncol):
                spl=CubicSpline(pars[0]['COVMAT_Z'],pars[0]['SIGMA'][j,j,:])
                self.sigma[j,j,:] = np.clip(spl(self.z), minsig, None)

                self.covmat[j,j,:] = self.sigma[j,j,:]*self.sigma[j,j,:]

            # off-diagonals
            for j in xrange(ncol):
                for k in xrange(j+1,ncol):
                    spl=CubicSpline(pars[0]['COVMAT_Z'],pars[0]['SIGMA'][j,k,:])
                    self.sigma[j,k,:] = spl(self.z)

                    too_high,=np.where(self.sigma[j,k,:] > 0.9)
                    if (too_high.size > 0):
                        self.sigma[j,k,too_high] = 0.9
                    too_low,=np.where(self.sigma[j,k,:] < -0.9)
                    if (too_low.size > 0):
                        self.sigma[j,k,too_low] = -0.9

                    self.sigma[k,j,:] = self.sigma[j,k,:]

                    self.covmat[j,k,:] = self.sigma[k,j,:] * self.sigma[j,j,:] * self.sigma[k,k,:]
                    self.covmat[k,j,:] = self.covmat[j,k,:]

            # volume factor
            spl=CubicSpline(pars[0]['VOLUME_FACTOR_Z'],pars[0]['VOLUME_FACTOR'])
            self.volume_factor = spl(self.z)

            # corrections
            spl=CubicSpline(pars[0]['CORR_Z'],pars[0]['CORR'])
            self.corr = spl(self.z)
            spl=CubicSpline(pars[0]['CORR_SLOPE_Z'],pars[0]['CORR_SLOPE'])
            self.corr_slope = spl(self.z)

            spl=CubicSpline(pars[0]['CORR_Z'],pars[0]['CORR2'])
            self.corr2 = spl(self.z)
            spl=CubicSpline(pars[0]['CORR_SLOPE_Z'],pars[0]['CORR2_SLOPE'])
            self.corr2_slope = spl(self.z)

            if 'CORR_R' in pars.dtype.names:
                # protect against stupidity
                if (pars[0]['CORR_R'][0] <= 0.0) :
                    self.corr_r = np.ones(nz)
                else:
                    spl=CubicSpline(pars[0]['CORR_SLOPE_Z'],pars[0]['CORR_R'])
                    self.corr_r = spl(self.z)

                test,=np.where(self.corr_r < 0.5)
                if (test.size > 0) : self.corr_r[test] = 0.5

                if (pars[0]['CORR2_R'][0] <= 0.0):
                    self.corr2_r = np.ones(nz)
                else:
                    spl=CubicSpline(pars[0]['CORR_SLOPE_Z'],pars[0]['CORR2_R'])
                    self.corr2_r = spl(self.z)

                test,=np.where(self.corr2_r < 0.5)
                if (test.size > 0) : self.corr2_r[test] = 0.5

            else :
                self.corr_r = np.ones(nz)
                self.corr2_r = np.ones(nz)

        # mstar
        # create LUT
        self._mstar = ms(self.z)

        # luminosity function integrations
        self.lumnorm = np.zeros((self.lumrefmagbins.size,nz))
        self.alpha = alpha
        for i in xrange(nz):
            f = schechter_pdf(self.lumrefmagbins, alpha=self.alpha, mstar=self._mstar[i])
            self.lumnorm[:,i] = refmagbinsize*np.cumsum(f)

        if has_file:
            # lupcorr (annoying!)
            self.lupcorr = np.zeros((self.refmagbins.size,nz,ncol),dtype='f8')
            if (do_lupcorr):
                bnmgy = bvalues*1e9

                for i in xrange(nz):
                    mags = np.zeros((self.refmagbins.size,nmag))
                    lups = np.zeros((self.refmagbins.size,nmag))

                    mags[:,ref_ind] = self.refmagbins

                    # go redward
                    for j in xrange(ref_ind+1,nmag):
                        mags[:,j] = mags[:,j-1] - (self.c[i,j-1]+self.slope[i,j-1]*(mags[:,ref_ind]-self.pivotmag[i]))
                    # blueward
                    for j in xrange(ref_ind-1,-1,-1):
                        mags[:,j] = mags[:,j+1] + (self.c[i,j]+self.slope[i,j]*(mags[:,ref_ind]-self.pivotmag[i]))

                    # and the luptitude conversion
                    for j in xrange(nmag):
                        flux = 10.**((mags[:,j]-22.5)/(-2.5))
                        lups[:,j] = 2.5*np.log10(1.0/bvalues[j]) - np.arcsinh(0.5*flux/bnmgy[j])/(0.4*np.log(10.0))

                    magcol = mags[:,0:ncol] - mags[:,1:ncol+1]
                    lupcol = lups[:,0:ncol] - lups[:,1:ncol+1]

                    self.lupcorr[:,i,:] = lupcol - magcol

        # set top overflow bins to very large number
        self.z[self.z.size-1] = 1000.0
        self.zinteger=np.round(self.z*self.zbinscale).astype(np.int64)
        self.refmagbins[self.refmagbins.size-1] = 1000.0
        self.refmaginteger = np.round(self.refmagbins*self.refmagbinscale).astype(np.int64)
        self.lumrefmagbins[self.lumrefmagbins.size-1] = 1000.0
        self.lumrefmaginteger = np.round(self.lumrefmagbins*self.refmagbinscale).astype(np.int64)
        self.ncol = ncol
        self.alpha = alpha
        self.mstar_survey = mstar_survey
        self.mstar_band = mstar_band
        self.limmag = limmag

        # don't make this into a catalog
        #super(RedSequenceColorPar, self).__init__(zredstr)


    def mstar(self,z):
        """
        M_star lookup

        parameters
        ----------
        z: array of floats
           redshift

        returns
        -------
        mstar: array of floats

        """
        # lookup and return mstar...awesome.
        zind = self.zindex(z)
        return self._mstar[zind]

    def zindex(self,z):
        """
        redshift index lookup

        parameters
        ----------
        z: array of floats

        returns
        -------
        indices: array of integers
            redshift indices

        """
        # return the z index/indices with rounding.

        zind = np.searchsorted(self.zinteger,np.round(np.atleast_1d(z)*self.zbinscale).astype(np.int64))
        #zind = np.searchsorted(self.zinteger,(np.atleast_1d(z)*self.zbinscale).astype(np.int64))
        if (zind.size == 1):
            return np.asscalar(zind)
        else:
            return zind

        # and check for top overflows.  Minimum is always 0
        #test,=np.where(zind == self.z.size)
        #if (test.size > 0): zind[test] = self.z.size-1

    def refmagindex(self,refmag):
        """
        reference magnitude index lookup

        parameters
        ----------
        refmag: array of floats

        returns
        -------
        indices: array of integers
            refmag indices
        """
        # return the refmag index/indices with rounding

        refmagind = np.searchsorted(self.refmaginteger,np.round(np.atleast_1d(refmag)*self.refmagbinscale).astype(np.int64))
        if (refmagind.size == 1):
            return np.asscalar(refmagind)
        else:
            return refmagind

    def lumrefmagindex(self,lumrefmag):
        """
        luminosity reference magnitude index lookup

        parameters
        ----------
        lumrefmag: array of floats

        returns
        -------
        indices: array of integers
            lumrefmag indices

        """

        lumrefmagind = np.searchsorted(self.lumrefmaginteger,np.round(np.atleast_1d(lumrefmag)*self.refmagbinscale).astype(np.int64))
        if (lumrefmagind.size == 1):
            return np.asscalar(lumrefmagind)
        else:
            return lumrefmagind

    def calculate_chisq_redshifts(self, galaxy, zs, calc_lkhd=False, z_is_index=False):
        """
        Compute chisq for a single galaxy at a series of redshifts.  Optimized for the
        redshift case.

        Parameters
        ----------
        galaxy: Galaxy
        zs: redshift or bins
        z_is_index: bool, default=False
           The zs are indices and not redshifts
        calc_lkhd: bool, default=False
           Calculate likelihood rather than chisq

        Returns
        -------
        chisqs: array of floats
        """

        calc_chisq = not calc_lkhd

        if z_is_index:
            zinds = zs
        else:
            zinds = self.zindex(zs)

        magind = self.refmagindex(galaxy.refmag)
        galcolor = galaxy.galcol

        return compute_chisq(self.covmat[:,:,zinds], self.c[zinds,:],
                             self.slope[zinds,:], self.pivotmag[zinds],
                             np.array(galaxy.refmag), galaxy.mag_err,
                             galcolor, refmagerr=np.array(galaxy.refmag_err),
                             lupcorr=self.lupcorr[magind,zinds,:],
                             calc_chisq=calc_chisq, calc_lkhd=calc_lkhd)


    def calculate_chisq(self, galaxies, z, calc_lkhd=False, z_is_index=False):
        """
        compute chisq for a set of galaxies at redshift z OR a galaxy at many redshifts OR
          many galaxies at many redshifts

        parameters
        ----------
        galaxies: GalaxyCatalog
            galaxies which need chisq values
        z: redshift
            redshift to compute chisq
        calc_lkhd: bool, optional
            compute likelihood rather than chisq (default False)
        z_is_index: bool, optional
            The input redshift is already a redshift index

        returns
        -------
        chisqs: array of floats
        """

        if calc_lkhd:
            calc_chisq = False
        else:
            calc_chisq = True

        if z_is_index:
            zind = z
        else:
            zind = self.zindex(z)
        magind = self.refmagindex(galaxies.refmag)
        galcolor = galaxies.galcol

        # Need to check if this is a single galaxy AND redshift
        if np.atleast_1d(zind).size == 1 and len(galaxies) == 1:
            zind = np.atleast_1d(zind)
            magind = np.atleast_1d(magind)

        return compute_chisq(self.covmat[:,:,zind], self.c[zind,:],
                             self.slope[zind,:], self.pivotmag[zind],
                             galaxies.refmag, galaxies.mag_err,
                             galcolor, refmagerr=galaxies.refmag_err,
                             lupcorr=self.lupcorr[magind,zind,:],
                             calc_chisq=calc_chisq, calc_lkhd=calc_lkhd)


    def plot_redsequence_diag(self, fig, ind, bands):
        """
        """

        not_extrap, = np.where(~self.extrapolated)

        ax = fig.add_subplot(221)
        ax.plot(self.z[: -1], self.c[: -1, ind], 'r--')
        ax.plot(self.z[not_extrap], self.c[not_extrap, ind], 'r-')
        ax.set_xlabel('Redshift')
        ax.set_ylabel('<%s - %s>' % (bands[ind], bands[ind + 1]))

        ax = fig.add_subplot(222)
        ax.plot(self.z[: -1], self.slope[: -1, ind], 'r--')
        ax.plot(self.z[not_extrap], self.slope[not_extrap, ind], 'r-')
        ax.set_xlabel('Redshift')
        ax.set_ylabel('(%s - %s) slope' % (bands[ind], bands[ind + 1]))

        ax = fig.add_subplot(223)
        ax.plot(self.z[: -1], self.sigma[ind, ind, : -1], 'r--')
        ax.plot(self.z[not_extrap], self.sigma[ind, ind, not_extrap], 'r-')
        ax.set_xlabel('Redshift')
        ax.set_ylabel('(%s - %s) sigma' % (bands[ind], bands[ind + 1]))

        fig.tight_layout()

    def plot_redsequence_offdiags(self, fig, bands):
        """
        """

        noff = (self.ncol * self.ncol - self.ncol) / 2

        nrow = (noff + 1) / 2

        not_extrap, = np.where(~self.extrapolated)

        ctr = 1
        for j in xrange(self.ncol):
            for k in xrange(j + 1, self.ncol):
                ax = fig.add_subplot(nrow, 2, ctr)
                ax.plot(self.z[: -1], self.sigma[j, k, : -1], 'r--')
                ax.plot(self.z[not_extrap], self.sigma[j, k, not_extrap], 'r-')
                ax.set_xlabel('Redshift')
                ax.set_ylabel('Corr %s-%s / %s-%s' % (bands[j], bands[j + 1], bands[k], bands[k + 1]))
                ctr += 1

        fig.tight_layout()

    def __repr__(self):
        return "Representation here."
