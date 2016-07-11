import fitsio
import esutil as eu
import numpy as np
import itertools


class Catalog(object):

    """ docstring """

    def __init__(self, *arrays):
        if any([len(arr) != len(arrays[0]) for arr in arrays]):
            raise ValueError("Input arrays must have the same length.")
        self._arrays = list(arrays)
        self.dtype = itertools.chain([arr.dtype for arr in self._arrays])

    @classmethod
    def fromfilename(cls, filename):
        array = fitsio.read(filename)
        return cls(array)

    def __getattribute__(self, attr):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:  # attr must be a fieldname
            pass
        for arr in self._arrays:
            if attr.upper() in arr.dtype.names:
                return arr[attr.upper()]
        return object.__getattribute__(self, attr)

    def __getitem__(self, key):
        return Catalog(*(arr.__getitem__(key) 
                                        for arr in self._arrays))

    def __len__(self): return len(self._arrays[0])

    def add_fields(self, array):
        if len(array) != len(self):
            raise ValueError("Input arrays must have the same length.")
        self._arrays.append(array)

