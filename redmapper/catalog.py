import fitsio
import esutil as eu
import numpy as np
import itertools
from numpy.lib.recfunctions import merge_arrays

### This should be used as an abstract base class
class DataObject(object):

    """ docstring """

    def __init__(self, *arrays):
        self._ndarray = merge_arrays(arrays, flatten=True)

    @classmethod
    def from_fits_file(cls, filename):
        array = fitsio.read(filename)
        return cls(array)

    def __getattribute__(self, attr):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:  # attr must be a fieldname
            pass
        if attr.upper() in self._ndarray.dtype.names:
            return self._ndarray[attr.upper()]
        return object.__getattribute__(self, attr)

    def __setattr__(self, attr, val):
        if attr == '_ndarray':
            object.__setattr__(self, attr, val)
        elif attr.upper() in self._ndarray.dtype.names:
            self._ndarray[attr.upper()] = val
        else:
            object.__setattr__(self, attr, val)

    @property
    def dtype(self): return self._ndarray.dtype

    def add_fields(self, array):
        self._ndarray = merge_arrays([self._ndarray, array], flatten=True)


class Entry(DataObject):

    """ docstring """

    def __init__(self, *arrays):
        if any([arr.size != 1 for arr in arrays]):
            raise ValueError("Input arrays must have length one.")
        super(Entry, self).__init__(*arrays)

    @classmethod
    def from_dict(cls, dict): pass

    def __getattribute__(self, attr):
        return super(Entry, self).__getattribute__(attr)


class Catalog(DataObject):

    """ docstring """

    entry_class = Entry

    def __len__(self): return len(self._ndarray)

    def __getitem__(self, key):
    	if isinstance(key, int):
    		return self.entry_class(self._ndarray.__getitem__(key))
        return type(self)(self._ndarray.__getitem__(key))

    def __setitem__(self, key, val):
        self._ndarray.__setitem__(key, val)

