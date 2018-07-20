from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import fitsio
import esutil
from esutil.htm import Matcher
import numpy as np
import itertools
import healpy as hp
import os

from .catalog import Catalog, Entry

zred_extra_dtype = [('ZRED', 'f4'),
                    ('ZRED_E', 'f4'),
                    ('ZRED2', 'f4'),
                    ('ZRED2_E', 'f4'),
                    ('ZRED_UNCORR', 'f4'),
                    ('ZRED_UNCORR_E', 'f4'),
                    ('LKHD', 'f4'),
                    ('CHISQ', 'f4')]


class Galaxy(Entry):
    """
    """

    @property
    def galcol(self):
        return self.mag[:-1] - self.mag[1:]


class GalaxyCatalog(Catalog):
    """
    Name:
        GalaxyCatalog
    Purpose:
        An object for holding galaxy catalogs, including all of
        the attributes that a galaxy would have.
    """

    entry_class = Galaxy

    def __init__(self, *arrays, **kwargs):
        super(GalaxyCatalog, self).__init__(*arrays)
        self._htm_matcher = None
        self.depth = 10 if 'depth' not in kwargs else kwargs['depth']
        #self._smatch_catalog = None
        #self._smatch_nside = 4096 if 'smatch_nside' not in kwargs else kwargs['smatch_nside']

    @classmethod
    def from_galfile(cls, filename, zredfile=None, nside=0, hpix=0, border=0.0):
        """
        Name:
            from_galfile
        Purpose:
            do the actual reading in of the data fields in some galaxy
            catalog file
        Calling Squence:
            TODO
        Inputs:
            filename: a name of a galaxy catalog file
        Optional Inputs:
            nside: integer
                healpix nside of sub-pixel
            hpix: integer
                healpix pixel (ring order) of sub-pixel
            zredfile: string
                name of zred file
        Outputs:
            A galaxy catalog object
        """
        if zredfile is not None:
            use_zred = True
        else:
            use_zred = False

        if hpix == 0:
            _hpix = None
        else:
            _hpix = hpix
        # do we have appropriate keywords
        if _hpix is not None and nside is None:
            raise ValueError("If hpix is specified, must also specify nside")
        if border < 0.0:
            raise ValueError("Border must be >= 0.0.")
        # ensure that nside is valid, and hpix is within range (if necessary)
        if nside > 0:
            if not hp.isnsideok(nside):
                raise ValueError("Nside not valid")
            if _hpix is not None:
                if _hpix < 0 or _hpix >= hp.nside2npix(nside):
                    raise ValueError("hpix out of range.")

        # check that the file is there and the right format
        # this will raise an exception if it's not there.
        hdr = fitsio.read_header(filename, ext=1)
        pixelated = hdr.get("PIXELS", 0)
        fitsformat = hdr.get("FITS", 0)

        # check zredfile
        if use_zred:
            zhdr = fitsio.read_header(zredfile, ext=1)
            zpixelated = zhdr.get("PIXELS", 0)

        if not pixelated:
            cat = fitsio.read(filename, ext=1, upper=True)
            if use_zred:
                zcat = fitsio.read(zredfile, ext=1, upper=True)
                if zcat.size != cat.size:
                    raise ValueError("zredfile is a different length (%d) than catfile (%d)" % (zcat.size, cat.size))
                return cls(cat, zcat)
            else:
                return cls(cat)
        else:
            if use_zred:
                if not zpixelated:
                    raise ValueError("galfile is pixelated but zredfile is not")

        # this is to keep us from trying to use old IDL galfiles
        if not fitsformat:
            raise ValueError("Input galfile must describe fits files.")

        # now we can read in the galaxy table summary file...
        tab = Entry.from_fits_file(filename, ext=1)
        nside_tab = tab.nside
        if nside > nside_tab:
            raise ValueError("""Requested nside (%d) must not be larger than
                                    table nside (%d).""" % (nside, nside_tab))

        if use_zred:
            #ztab = fitsio.read(zredfile, ext=1, upper=True)
            ztab = Entry.from_fits_file(zredfile, ext=1)
            zpath = os.path.dirname(zredfile)

        # which files do we want to read?
        path = os.path.dirname(os.path.abspath(filename))

        indices = get_subpixel_indices(tab, hpix=_hpix, border=border, nside=nside)

        # Make sure all the zred files are there
        if use_zred:
            # The default mode, copied from the IDL code, is that we just don't
            # read in any galaxy pixels that don't have an associated zred.
            # I don't know if this is what we want going forward, but I'll leave
            # it like this at the moment.
            # Also, we are assuming that the files actually match up in terms of length, etc.
            mark = np.zeros(indices.size, dtype=np.bool)
            for i, f in enumerate(ztab.filenames[indices]):
                if os.path.isfile(os.path.join(zpath, f.decode())):
                    mark[i] = True

            bad, = np.where(~mark)
            if bad.size == indices.size:
                raise ValueError("There are no zred files associated with the galaxy pixels.")

            indices = np.delete(indices, bad)

        # create the catalog array to read into
        elt = fitsio.read(os.path.join(path, tab.filenames[indices[0]].decode()), ext=1, rows=0, upper=True)
        cat = np.zeros(np.sum(tab.ngals[indices]), dtype=elt.dtype)

        if use_zred:
            zelt = fitsio.read(os.path.join(zpath, ztab.filenames[indices[0]].decode()), ext=1, rows=0, upper=True)
            zcat = np.zeros(cat.size, dtype=zelt.dtype)

        # read the files
        ctr = 0
        for index in indices:
            cat[ctr: ctr + tab.ngals[index]] = fitsio.read(os.path.join(path, tab.filenames[index].decode()), ext=1, upper=True)
            if use_zred:
                # Note that this effectively checks that the numbers of rows in each file match properly (though the exception will be cryptic...)
                zcat[ctr: ctr + tab.ngals[index]] = fitsio.read(os.path.join(zpath, ztab.filenames[index].decode()), ext=1, upper=True)
            ctr += tab.ngals[index]

        # In the IDL version this is trimmed to the precise boundary requested.
        # that's easy in simplepix.  Not sure how to do in healpix.
        if use_zred:
            return cls(cat, zcat)
        else:
            return cls(cat)

    @property
    def galcol(self):
        galcol = self.mag[:,:-1] - self.mag[:,1:]
        return galcol

    @property
    def galcol_err(self):
        galcol_err = np.sqrt(self.mag_err[:, :-1]**2. + self.mag_err[:, 1:]**2.)
        return galcol_err

    def add_zred_fields(self):
        """
        """

        dtype_augment = [dt for dt in zred_extra_dtype if dt[0].lower() not in self.dtype.names]
        if len(dtype_augment) > 0:
            self.add_fields(dtype_augment)

    def match_one(self, ra, dec, radius):
        """
        match an ra/dec to the galaxy catalog

        parameters
        ----------
        ra: input ra
        dec: input dec
        radius: float
           radius in degrees

        returns
        -------
        #(indices, dists)
        #indices: array of integers
        #    indices in galaxy catalog of matches
        #dists: array of floats
        #    match distance for each match (degrees)
        """

        if self._htm_matcher is None:
            self._htm_matcher = Matcher(self.depth, self.ra, self.dec)

        _, indices, dists = self._htm_matcher.match(ra, dec, radius, maxmatch=0)

        return indices, dists

    def match_many(self, ras, decs, radius, maxmatch=0):
        """
        match many ras/decs to the galaxy catalog

        parameters
        ----------
        ras: input ras
        decs: input decs
        radius: float
           radius/radii in degrees
        maxmatch: int, optional
           set to 0 for multiple matches, or max number of matches

        returns
        -------
        (i0, i1, dists)
        i0: array of integers
            indices for input ra/dec
        i1: array of integers
            indices for galaxy catalog
        dists: array of floats
            match distance (degrees)
        """

        if self._htm_matcher is None:
            self._htm_matcher = Matcher(self.depth, self.ra, self.dec)

        return self._htm_matcher.match(ras, decs, radius, maxmatch=maxmatch)

def get_subpixel_indices(galtable, hpix=None, border=0.0, nside=0):
    """
    """

    if hpix is None or nside == 0:
        return np.arange(galtable.filenames.size)

    theta, phi = hp.pix2ang(galtable.nside, galtable.hpix)
    ipring_big = hp.ang2pix(nside, theta, phi)
    indices, = np.where(ipring_big == hpix)
    if border > 0.0:
        # now we need to find the extra boundary...
        boundaries = hp.boundaries(nside, hpix, step=galtable.nside/nside)
        inhpix = galtable.hpix[indices]
        for i in xrange(boundaries.shape[1]):
            pixint = hp.query_disc(galtable.nside, boundaries[:, i],
                                   border*np.pi/180., inclusive=True, fact=8)
            inhpix = np.append(inhpix, pixint)
        inhpix = np.unique(inhpix)
        _, indices = esutil.numpy_util.match(inhpix, galtable.hpix)

    return indices


