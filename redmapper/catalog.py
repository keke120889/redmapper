import fitsio
import esutil as eu
import numpy as np
import itertools
from numpy.lib.recfunctions import merge_arrays


class Catalog(object):

    """ docstring """

    singleton_class = DataEntry

    def __init__(self, *arrays):
        if any([len(arr) != len(arrays[0]) for arr in arrays]):
            raise ValueError("Input arrays must have the same length.")
        self._ndarray = merge_arrays(arrays, flatten=True)
        
    @classmethod
    def from_fits_file(cls, filename):
        array = fitsio.read(filename)
        return cls(array)

    def __len__(self): return len(self._ndarray)

    def __getattribute__(self, attr):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:  # attr must be a fieldname
            pass
        if attr.upper() in self._ndarray.dtype.names:
            return self._ndarray[attr.upper()]
        return object.__getattribute__(self, attr)

    def __setattr__(self, attr, val):
        try:
            object.__setattr__(self, attr, val)
        except AttributeError:
            pass
        if attr.upper() in self._ndarray.dtype.names:
            self._ndarray.__setattr__(self, attr, val)
        object.__setattr__(self, attr, val)

    def __getitem__(self, key):
    	if isinstance(key, int):
    		return singleton_class(np.array(self._ndarray.__getitem__(key)))
        return type(self)(self._ndarray.__getitem__(key))

    def __setitem__(self, key, val):
        self._ndarray.__setitem__(key, val)

    def add_fields(self, array):
        if len(array) != len(self):
            raise ValueError("Input arrays must have the same length.")
        self._ndarray.append(array)


class DataEntry(Catalog):

	def __init__(self, *arrays):
		if len(arrays[0]) != 1:
			raise ValueError("Input arrays must have length one.")
		super(DataEntry, self).__init__(arrays)

    @classmethod
    def from_dict(cls, dict): pass

	def __getattribute__(self, attr):
        return super(DataEntry, self).__getattribute__(attr)[0]

	def __getitem__(self, key):
        raise TypeError("\'" + type(self) + 
        					"\' object has no attribute \'__getitem__\'")

    def __len__(self):
        raise TypeError("object of type \'" + type(self) + "\' has no len()")

