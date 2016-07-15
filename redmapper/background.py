import fitsio
import numpy as np
from catalog import Entry


class Background(Entry):
    """Docstring."""

    def __init__(self, filename, scale=1.0):
        arr = fitsio.read(filename, ext=1)
        for fieldname in arr.dtype.names:
            setattr(self, fieldname.lower(), arr[fieldname])
        self._init_attrs()

    def _init_attrs(self):
        self.imagbinsize, self.zbinsize, self.chisqbinsize = 0.01, 0.001, 0.5

        