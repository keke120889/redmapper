import fitsio
import numpy as np
from catalog import Entry
from scipy.interpolate import RegularGridInterpolator


class Background(Entry):
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

