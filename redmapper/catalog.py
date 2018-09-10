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
        for array in arrays:
            self._lower_array(array)

        if len(arrays) == 1:
            self._ndarray = arrays[0]
        else:
            self._ndarray = merge_arrays(arrays, flatten=True)



    @classmethod
    def from_fits_file(cls, filename, ext=1, rows=None):
        """
        Constructs DataObject directly from FITS file.

        Makes use of Erin Sheldon's fitsio reading routine. The fitsio
        package wraps cfitsio for maximum efficiency.

        Args:
            filename (string): the file path and name.
            ext: optional extension (default == 1)

        Returns:
            DataObject, properly constructed.
        """
        array = fitsio.read(filename, ext=ext, rows=rows, lower=True, trim_strings=True)
        return cls(array)

    @classmethod
    def from_fits_ext(cls, fits_ext):
        """
        """
        array = fits_ext.read(upper=True)
        return cls(array)

    @classmethod
    def zeros(cls, size, dtype):
        return cls(np.zeros(size, dtype=dtype))

    def __getattr__(self, attr):
        try:
            return self._ndarray[attr.lower()]
        except:
            return object.__getattribute__(self, attr)

    def __setattr__(self, attr, val):
        if attr == '_ndarray':
            object.__setattr__(self, attr, val)
        elif attr.lower() in self._ndarray.dtype.names:
            self._ndarray[attr.lower()] = val
        else:
            object.__setattr__(self, attr, val)

    @property
    def dtype(self):
        """numpy.dtype: dtype associated with the DataObject."""
        return self._ndarray.dtype

    def add_fields(self, newdtype):
        array = np.zeros(self._ndarray.size, newdtype)
        self._lower_array(array)
        self._ndarray = merge_arrays([self._ndarray, array], flatten=True)

    def to_fits_file(self, filename, clobber=False, header=None, extname=None, indices=None):
        if self._ndarray.size == 1:
            temp_array = np.zeros(1, dtype=self._ndarray.dtype)
            temp_array[0] = self._ndarray
            fitsio.write(filename, temp_array, clobber=clobber, header=header, extname=extname)
        else:
            if indices is None:
                fitsio.write(filename, self._ndarray, clobber=clobber, header=header, extname=extname)
            else:
                fitsio.write(filename, self._ndarray[indices], clobber=clobber, header=header, extname=extname)

    def _lower_array(self, array):
        names = list(array.dtype.names)
        array.dtype.names = [n.lower() for n in names]

    def __repr__(self):
        # return the representation of the underlying array
        return repr(self._ndarray)

    def __str__(self):
        # return the string of the underlying array
        return str(self._ndarray)

    def __dir__(self):
        # lower case list of all the available variables
        # also need to know original __dir__!
        #return [x.lower() for x in self._ndarray.dtype.names]
        return sorted(set(
                dir(type(self)) +
                self.__dict__.keys() +
                [x.lower() for x in self._ndarray.dtype.names]))


class Entry(DataObject):
    """Entries are extensions of DataObjects.

    The __init__ method simply calls the 
    constructor for DataObject after it has verified that
    there is only a single entry being passed in.
    """

    #def __init__(self, *arrays):
    #    if any([arr.size != 1 for arr in arrays]):
    #        raise ValueError("Input arrays must have length one.")
    #    super(Entry, self).__init__(*arrays)
    def __init__(self, array):
        if array.size != 1:
            raise ValueError("Input array must have length one.")
        # If this is an array of length 1, we want it to be a scalar-ish
        if len(array.shape) == 0:
            super(Entry, self).__init__(array)
        else:
            super(Entry, self).__init__(array[0])

    @classmethod
    def from_dict(cls, dict): pass

    def add_fields(self, newdtype):
        array = np.zeros(self._ndarray.size, newdtype)
        self._lower_array(array)
        self._ndarray = merge_arrays([self._ndarray, array], flatten=True)[0]


    def __getattr__(self, attr):
        try:
            #return self._ndarray[attr.lower()][0]
            return self._ndarray[attr.lower()]
        except:
            return object.__getattribute__(self, attr)


class Catalog(DataObject):
    """Catalogs are extensions of DataObjects.

    Catalogs are composed of may Entry objects.
    Tom - I am not sure that this object is complete. TODO
    Eli - It might be.  The tricks here are so you can access
           these with "catalog.key" rather than "catalog['KEY']"
    """

    entry_class = Entry

    @property
    def size(self): return self._ndarray.size

    def append(self, append_cat):
        if isinstance(append_cat, Catalog):
            self._ndarray = np.append(self._ndarray, append_cat._ndarray)
        else:
            self._ndarray = np.append(self._ndarray, append_cat)

    def extend(self, n_new):
        temp = np.zeros(n_new, dtype=self._ndarray.dtype)
        self._ndarray = np.append(self._ndarray, temp)

    def __len__(self): return self.size

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.entry_class(self._ndarray.__getitem__(key))
        return type(self)(self._ndarray.__getitem__(key))

    def __setitem__(self, key, val):
        self._ndarray.__setitem__(key, val)

