import fitsio
import esutil as eu
import numpy as np
from scipy import interpolate

from catalog import Catalog
from utilities import CubicSpline
from utilities import MStar

class RedSequenceColorPar(object):
    def __init__(self, filename, zbinsize=None, minsig=0.01, fine=False, zrange=None):

        pars,hdr=fitsio.read(filename,ext=1,header=True)

        try:
            limmag = hdr['LIMMAG']
            if (zrange is None):
                zrange = np.array([hdr['ZRANGE0'],hdr['ZRANGE1']])
                    
            alpha = hdr['ALPHA']
            mstar_survey = hdr['MSTARSUR']
            mstar_band = hdr['MSTARBAN']
            ncol = hdr['NCOL']
        except:
            raise ValueError("Missing field from parameter header.")

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

        nz = np.round((zrange[1]-zrange[0])/zbinsize).astype('i4')
        self.z = zbinsize*np.arange(nz) + zrange[0]

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
        spl=CubicSpline(pars[0][pivotmag_name+'_Z'],pars[0][pivotmag_name])
        self.pivotmag = spl(self.z)

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
            self.sigma[j,j,:] = spl(self.z)

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
        #ms = MStar(mstar_survey,mstar_band)
        self._mstar = ms(self.z)

        # luminosity function integrations
        self.lumnorm = np.zeros((self.lumrefmagbins.size,nz))
        self.alpha = alpha
        for i in xrange(nz):
            f=10.**(0.4*(self.alpha+1.0)*(self._mstar[i]-self.lumrefmagbins))*np.exp(-10.**(0.4*(self._mstar[i]-self.lumrefmagbins)))
            self.lumnorm[:,i] = refmagbinsize*np.cumsum(f)
            

        # lupcorr (annoying!)
        #self.lupcorr = np.zeros((ncol,nz,self.refmagbins.size))
        #self.lupcorr = np.zeros((ncol,self.refmagbins.size,nz))
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

                #self.lupcorr[i,:,:] = lupcol - magcol
                #self.lupcorr[:,:,i] = lupcol - magcol
                self.lupcorr[:,i,:] = lupcol - magcol

        # set top overflow bins to very large number
        self.z[self.z.size-1] = 1000.0
        self.zinteger=np.round(self.z*self.zbinscale).astype(np.int64)
        self.refmagbins[self.refmagbins.size-1] = 1000.0
        self.refmaginteger = np.round(self.refmagbins*self.refmagbinscale).astype(np.int64)
        self.lumrefmagbins[self.lumrefmagbins.size-1] = 1000.0
        self.lumrefmaginteger = np.round(self.lumrefmagbins*self.refmagbinscale).astype(np.int64)
        self.ncol = ncol

        # make this into a catalog
        #super(RedSequenceColorPar, self).__init__(zredstr)


    def mstar(self,z):
        # lookup and return mstar...awesome.
        #ind = np.searchsorted(self.z,z)
        zind = self.zindex(z)
        return self._mstar[zind]

    def zindex(self,z):
        # return the z index/indices with rounding.

        zind = np.searchsorted(self.zinteger,np.round(np.atleast_1d(z)*self.zbinscale).astype(np.int64))
        if (zind.size == 1):
            return np.asscalar(zind)
        else:
            return zind
        
        # and check for top overflows.  Minimum is always 0
        #test,=np.where(zind == self.z.size)
        #if (test.size > 0): zind[test] = self.z.size-1

    def refmagindex(self,refmag):
        # return the refmag index/indices with rounding

        refmagind = np.searchsorted(self.refmaginteger,np.round(np.atleast_1d(refmag)*self.refmagbinscale).astype(np.int64))
        if (refmagind.size == 1):
            return np.asscalar(refmagind)
        else :
            return refmagind

    def lumrefmagindex(self,lumrefmag):

        lumrefmagind = np.searchsorted(self.lumrefmaginteger,np.round(np.atleast_1d(lumrefmag)*self.refmagbinscale).astype(np.int64))
        if (lumrefmagind.size == 1):
            return np.asscalar(lumrefmagind)
        else:
            return lumrefmagind

    def calculate_chisq(self, galaxies, z):
        zind = self.zindex(z)
        magind = self.refmagindex(galaxies.refmag)
        galcolor = galaxies.mag[:, :self.ncol] - galaxies.mag[:, 1:]
        chisq_dist = redmapper.chisq_dist.ChisqDist(self.covmat[:,:,zind],self.c[zind,:],self.slope[zind,:],self.pivotmag[zind],galaxies.refmag,galaxies.mag_err,galcolor,refmagerr=galaxies.refmag_err,lupcorr=self.lupcorr[magind,zind,:])
        chisq = chisq_dist.compute_chisq(chisq_mode=True)
        return chisq

    def calculate_zred(self,blah):
        # I think this can be housed here.  Not urgent.
        pass

    def __repr__(self):
        return "Representation here."
