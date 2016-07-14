import fitsio
import esutil as eu
import numpy as np
import itertools
from numpy.lib.recfunctions import merge_arrays


class DataObject(object):
    """Abstract base class to encapsulate info from FITS files."""

    def __init__(self, *arrays):
        """Constructs DataObject from arbitrary number of ndarrays.

        Each ndarray can have an arbitrary number of fields. Field
        names should all be capitalized and words in multi-word field 
        names should be separated by underscores if necessary. ndarrays
        have a 'size' property---their sizes should all be equivalent.

        Args:
            arrays (numpy.ndarray): ndarrays with equivalent sizes.
        """
        self._ndarray = merge_arrays(arrays, flatten=True)

    @classmethod
    def from_fits_file(cls, filename):
        """Constructs DataObject directly from FITS file.

        Makes use of Erin Sheldon's fitsio reading routine. The fitsio
        package wraps cfitsio for maximum efficiency.

        Args:
            filename (string): the file path and name.

        Returns:
            DataObject, properly constructed.
        """
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
    def dtype(self): 
        """numpy.dtype: dtype associated with the DataObject."""
        return self._ndarray.dtype

    def add_fields(self, array):
        """Adds additional fields to the DataObject.

        The array should be prepopulated with values if they are known
        to avoid unecessary copying.

        Args:
            array (numpy.ndarray): ndarray with new fields.
        """
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

