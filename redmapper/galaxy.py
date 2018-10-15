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

class GalaxyCatalogMaker(object):
    """
    """

    def __init__(self, outbase, info_dict, nside=32):

        # Record values
        self.outbase = outbase
        self.nside = nside

        # Check the info dict
        self.lim_ref = info_dict['LIM_REF']
        self.ref_ind = info_dict['REF_IND']
        self.area = info_dict['AREA']
        self.nmag = info_dict['NMAG']
        self.mode = info_dict['MODE']
        self.zeropoint = info_dict['ZP']
        try:
            self.b = info_dict['B']
        except KeyError:
            self.b = np.zeros(self.nmag)

        try:
            self.u_ind = info_dict['U_IND']
        except KeyError:
            self.u_ind = None
        try:
            self.g_ind = info_dict['G_IND']
        except KeyError:
            self.g_ind = None
        try:
            self.r_ind = info_dict['R_IND']
        except KeyError:
            self.r_ind = None
        try:
            self.i_ind = info_dict['I_IND']
        except KeyError:
            self.i_ind = None
        try:
            self.z_ind = info_dict['Z_IND']
        except KeyError:
            self.z_ind = None
        try:
            self.y_ind = info_dict['Y_IND']
        except KeyError:
            self.y_ind = None

        self.filename = '%s_master_table.fit' % (self.outbase)

        # Start the file
        self.outpath = os.path.dirname(self.filename)
        self.outbase_nopath = os.path.basename(self.outbase)

        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)

        # create a table
        self.ngals = np.zeros(hp.nside2npix(self.nside), dtype=np.int32)

    def split_galaxies(self, gals):
        """
        """
        self.append_galaxies(gals)
        self.finalize_catalog()

    def append_galaxies(self, gals):
        """
        """

        theta = (90.0 - gals['dec']) * np.pi / 180.
        phi = gals['ra'] * np.pi / 180.

        ipring = hp.ang2pix(self.nside, theta, phi)

        h, rev = esutil.stat.histogram(ipring, min=0, max=self.ngals.size-1, rev=True)

        gdpix, = np.where(h > 0)
        for pix in gdpix:
            i1a = rev[rev[pix]: rev[pix + 1]]

            fname = os.path.join(self.outpath, '%s_%07d.fit' % (self.outbase_nopath, pix))

            if (self.ngals[pix] == 0) and (os.path.isfile(fname)):
                raise RuntimeError("We think there are 0 galaxies in pixel %d, but the file exists." % (pix))

            if self.ngals[pix] == 0:
                # Create a new file
                fitsio.write(fname, gals[i1a])
            else:
                fits = fitsio.FITS(fname, mode='rw')
                fits[1].append(gals[i1a])
                fits.close()

            self.ngals[pix] += i1a.size

    def finalize_catalog(self):
        """
        """

        hpix, = np.where(self.ngals > 0)

        filename_dtype = 'a%d' % (len(self.outbase_nopath) + 15)

        dtype = [('nside', 'i2'),
                 ('hpix', 'i4', hpix.size),
                 ('ra_pix', 'f8', hpix.size),
                 ('dec_pix', 'f8', hpix.size),
                 ('ngals', 'i4', hpix.size),
                 ('filenames', filename_dtype, hpix.size),
                 ('lim_ref', 'f4'),
                 ('ref_ind', 'i2'),
                 ('area', 'f8'),
                 ('nmag', 'i4'),
                 ('mode', 'a10'),
                 ('b', 'f8', self.nmag),
                 ('zeropoint', 'f4')]
        if self.u_ind is not None:
            dtype.append(('u_ind', 'i2'))
        if self.g_ind is not None:
            dtype.append(('g_ind', 'i2'))
        if self.r_ind is not None:
            dtype.append(('r_ind', 'i2'))
        if self.i_ind is not None:
            dtype.append(('i_ind', 'i2'))
        if self.z_ind is not None:
            dtype.append(('z_ind', 'i2'))
        if self.y_ind is not None:
            dtype.append(('y_ind', 'i2'))

        tab = Entry(np.zeros(1, dtype=dtype))

        tab.nside = self.nside
        tab.hpix = hpix

        theta, phi = hp.pix2ang(self.nside, hpix)
        tab.ra_pix = phi * 180. / np.pi
        tab.dec_pix = 90.0 - theta * 180. / np.pi

        tab.ngals = self.ngals[hpix]
        for i, pix in enumerate(hpix):
            tab.filenames[i] = '%s_%07d.fit' % (self.outbase_nopath, pix)
        tab.lim_ref = self.lim_ref
        tab.ref_ind = self.ref_ind
        tab.area = self.area
        tab.nmag = self.nmag
        tab.mode = self.mode
        tab.b = self.b
        tab.zeropoint = self.zeropoint

        if self.u_ind is not None:
            tab.u_ind = self.u_ind
        if self.g_ind is not None:
            tab.g_ind = self.g_ind
        if self.r_ind is not None:
            tab.r_ind = self.r_ind
        if self.i_ind is not None:
            tab.i_ind = self.i_ind
        if self.z_ind is not None:
            tab.z_ind = self.z_ind
        if self.y_ind is not None:
            tab.y_ind = self.y_ind

        hdr = fitsio.FITSHDR()
        hdr['PIXELS'] = 1
        hdr['FITS'] = 1

        tab.to_fits_file(self.filename, header=hdr, clobber=True)
