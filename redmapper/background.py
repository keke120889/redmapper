import fitsio
import numpy as np
from catalog import Entry
from scipy.interpolate import RegularGridInterpolator


class Background(object):
    """Docstring."""

    def __init__(self, filename):
        obkg = Entry.from_fits_file(filename)
        self.zbinsize, self.chisqbinsize, self.imagbinsize = 0.001, 0.5, 0.01
        
        self.lnchisqbins = obkg.lnchisqbins
        self.zbins = np.arange(obkg.zrange[0], obkg.zrange[1], zbinsize)
        self.chisqbins = np.arange(obkg.chisqbinrange[0], 
                                        obkg.chisqbinrange[1], chisqbinsize)
        self.imagbins = np.arange(obkg.imagrange[0], 
                                            obkg.imagrange[1], imagbinsize)
        
        self.sigma_g = _interp(obkg.zbins, obkg.chisqbins, obkg.imagbins,
                                    obkg.sigma_g, xnew=self.zbins, 
                                    ynew=self.chisqbins, znew=self.imagbins)
        self.sigma_lng = _interp(obkg.zbins, obkg.lnchisqbins, obkg.imagbins,
                                    obkg.sigma_lng, xnew=self.zbins, 
                                    znew=self.imagbins)
        self.n = np.sum(self.sigma_g, axis=1) * self.chisqbinsize
        
    @staticmethod
    def _interp(x, y, z, values, xnew=None, ynew=None, znew=None):
        if xnew is None: xnew = x
        if ynew is None: ynew = y
        if znew is None: znew = z
        xpts, ypts, zpts = np.meshgrid(xnew, ynew, znew)
        coord = np.vstack((xpts.flatten(), ypts.flatten(), zpts.flatten())).T
        interp_fn = RegularGridInterpolator((x, y, z), values)
        result = np.reshape(interp_fn(coord), (len(xnew), len(ynew), len(znew)))
        return np.where(result < 0, 0, result)

## for testing only
class IDLBackground(object):

    def __init__(self, filename):
        obkg = Entry.from_fits_file(filename)
        self.zbinsize, self.chisqbinsize, self.imagbinsize = 0.001, 0.5, 0.01

        imagbins = np.arange(obkg.imagrange[0], obkg.imagrange[1], imagbinsize)
        nimagbins = imagbins.size

        nchisqbins = obkg.chisqbins.size
        nlnchisqbins = obkg.lnchisqbins.size

        nzbins = obkg.zbins.size

        sigma_g_new = np.zeros((nzbins, nchisqbins, nimagbins))
        sigma_lng_new = np.zeros((nzbins, nchisqbins, nimagbins))

        for i in range(nzbins):
            for j in range(nchisqbins):
                sigma_g_new[i,j,:] = np.interp(imagbins, obkg.imagbins, obkg.sigma_g[i,j,:])
                sigma_g_new[i,j,:] = np.where(sigma_g_new[i,j,:] < 0, 0, sigma_g_new[i,j,:])
                sigma_lng_new[i,j,:] = np.interp(imagbins, obkg.imagbins, obkg.sigma_lng[i,j,:])
                sigma_lng_new[i,j,:] = np.where(sigma_lng_new[i,j,:] < 0, 0, sigma_lng_new[i,j,:])

        sigma_g = sigma_g_new.copy()
        sigma_lng = sigma_lng_new.copy()

        chisqbins = np.arange(obkg.chisqrange[0], obkg.chisqrange[1], chisqbinsize)
        nchisqbins = chisqbins.size

        sigma_g_new = np.zeros((nzbins, nchisqbins, nimagbins))

        for i in range(nzbins):
            for j in range(nimagbins):
                sigma_g_new[i,:,j] = np.interp(chisqbins, obkg.chisqbins, sigma_g[i,:,j])
                sigma_g_new[i,:,j] = np.where(sigma_g_new[i,:,j] < 0, 0, sigma_g_new[i,:,j])

        sigma_g = sigma_g_new.copy()

        zbins = np.arange(obkg.zrange[0], obkg.zrange[1], nzbins)
        nzbins = zbins.size

        for i in range(nchisqbins):
            for j in range(nimagbins):
                sigma_g_new[:,i,j] = np.interp(zbins, obkg.zbins, sigma_g[:,i,j])
                sigma_g_new[:,i,j] = np.where(sigma_g_new[:,i,j] < 0, 0, sigma_g_new[:,i,j])

        for i in range(nlnchisqbins):
            for j in range(nimagbins):
                sigma_lng_new[:,i,j] = np.interp(zbins, obkg.zbins, sigma_lng[:,i,j])
                sigma_lng_new[:,i,j] = np.where(sigma_lng_new[:,i,j] < 0, 0, sigma_lng_new[:,i,j])

        n_new = np.zeros((nzbins, nimagbins))
        for i in range(nzbins):
            n_new[i,:] = np.sum(sigma_g_new[i,:,:], axis=0) * chisqbinsize

        self.imagbins = imagbins
        self.chisqbins = chisqbins
        self.lnchisqbins = lnchisqbins
        self.zbins = zbins
        self.sigma_g = sigma_g_new
        self.sigma_lng = sigma_lng_new
        self.n = n_new

