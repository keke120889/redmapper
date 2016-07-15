import fitsio
import numpy as np
from catalog import Entry


class Background(Entry):
    """Docstring."""

    def __init__(self, filename, scale=1.0):
        imagbinsize, zbinsize, chisqbinsize = 0.01, 0.001, 0.5
        obkg = Entry.from_fits_file(filename)
        imagbins = np.arange(obkg.imagrange[0], obkg.imagrange[1], 
                                                            imagbinsize)
        sigma_g = np.zeros(shape=(obkg.imagbins.size, obkg.chisqbins.size,
                                                            obkg.zbins.size))
        sigma_lng = np.zeros(shape=(obkg.imagbins.size, obkg.lnchisqbins.size,
                                                            obkg.zbins.size))
        



        