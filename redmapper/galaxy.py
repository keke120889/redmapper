"""Classes for describing a galaxy catalog for redmapper.

This file contains the classes for reading, using, and making galaxy tables.
"""

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
    Class to describe a single galaxy.

    This just has convenience methods for individual elements in a GalaxyCatalog.
    """

    @property
    def galcol(self):
        """
        Get the array of galaxy colors.

        Returns
        -------
        galcol: `np.array`
           Float array of galaxy colors.
        """
        return self.mag[:-1] - self.mag[1:]


class GalaxyCatalog(Catalog):
    """
    Class to describe a redmapper galaxy Catalog.

    This contains ways of reading galaxy catalogs, getting colors, and matching
    galaxies to clusters.
    """
    entry_class = Galaxy

    def __init__(self, *arrays, **kwargs):
        """
        Instantiate a GalaxyCatalog

        Parameters
        ----------
        *arrays: `np.ndarray`
           Parameters for arrays to build galaxy catalog (e.g. galaxy and zred
           information)
        depth: `int`, optional
           HTM matcher depth, default is 10.
        """
        super(GalaxyCatalog, self).__init__(*arrays)
        self._htm_matcher = None
        self.depth = 10 if 'depth' not in kwargs else kwargs['depth']

    @classmethod
    def from_galfile(cls, filename, zredfile=None, nside=0, hpix=0, border=0.0, truth=False):
        """
        Generate a GalaxyCatalog from a redmapper "galfile."

        Parameters
        ----------
        filename: `str`
           Filename of the redmapper "galfile" galaxy file.
           This file may be a straight fits file or (recommended) a
           galaxy table "master_table.fit" summary file.
        zredfile: `str`, optional
           Filename of the redmapper zred "zreds_master_table.fit" summary file.
        nside: `int`, optional
           Nside of healpix sub-region to read in.  Default is 0 (full catalog).
        hpix: `int`, optional
           Healpix number (ring format) of sub-region to read in.
           Default is 0 (full catalog).
        border: `float`, optional
           Border around hpix (in degrees) to read in.  Default is 0.0.
        truth: `bool`, optional
           Read in truth information if available (e.g. mocks)?  Default is False.
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
        # FIXME: filter out any TRUTH information if necessary
        # will need to also get the list of columns from the thingamajig.

        # and need to be able to cut?
        elt = fitsio.read(os.path.join(path, tab.filenames[indices[0]].decode()), ext=1, rows=0, lower=True)
        dtype_in = elt.dtype.descr
        if not truth:
            mark = []
            for dt in dtype_in:
                if (dt[0] != 'ztrue' and dt[0] != 'm200' and dt[0] != 'central' and
                    dt[0] != 'halo_id'):
                    mark.append(True)
                else:
                    mark.append(False)

            dtype = [dt for i, dt in enumerate(dtype_in) if mark[i]]
            columns = [dt[0] for dt in dtype]
        else:
            dtype = dtype_in
            columns = None

        cat = np.zeros(np.sum(tab.ngals[indices]), dtype=dtype)

        if use_zred:
            zelt = fitsio.read(os.path.join(zpath, ztab.filenames[indices[0]].decode()), ext=1, rows=0, upper=False)
            zcat = np.zeros(cat.size, dtype=zelt.dtype)

        # read the files
        ctr = 0
        for index in indices:
            cat[ctr: ctr + tab.ngals[index]] = fitsio.read(os.path.join(path, tab.filenames[index].decode()), ext=1, lower=True, columns=columns)
            if use_zred:
                # Note that this effectively checks that the numbers of rows in each file match properly (though the exception will be cryptic...)
                zcat[ctr: ctr + tab.ngals[index]] = fitsio.read(os.path.join(zpath, ztab.filenames[index].decode()), ext=1, upper=False)
            ctr += tab.ngals[index]

        if _hpix is not None and nside > 0 and border > 0.0:
            # Trim to be closer to the border if necessary...

            nside_cutref = 512
            boundaries = hp.boundaries(nside, hpix, step=nside_cutref/nside)
            theta, phi = hp.pix2ang(nside_cutref, np.arange(hp.nside2npix(nside_cutref)))
            ipring_coarse = hp.ang2pix(nside, theta, phi)
            inhpix, = np.where(ipring_coarse == hpix)

            for i in xrange(boundaries.shape[1]):
                pixint = hp.query_disc(nside_cutref, boundaries[:, i], np.radians(border), inclusive=True, fact=8)
                inhpix = np.append(inhpix, pixint)
            inhpix = np.unique(inhpix)

            theta = np.radians(90.0 - cat['dec'])
            phi = np.radians(cat['ra'])
            ipring = hp.ang2pix(nside_cutref, theta, phi)
            _, indices = esutil.numpy_util.match(inhpix, ipring)

            if use_zred:
                return cls(cat[indices], zcat[indices])
            else:
                return cls(cat[indices])
        else:
            # No cuts
            if use_zred:
                return cls(cat, zcat)
            else:
                return cls(cat)

    @property
    def galcol(self):
        """
        Get the array of galaxy colors.

        Returns
        -------
        galcol: `np.array`
           Float array of colors [ngal, nmag - 1]
        """
        galcol = self.mag[:,:-1] - self.mag[:,1:]
        return galcol

    @property
    def galcol_err(self):
        """
        Get the array of galaxy color errors.

        Returns
        -------
        galcol_err: `np.array`
           Float array of errors [ngal, nmag - 1]
        """
        galcol_err = np.sqrt(self.mag_err[:, :-1]**2. + self.mag_err[:, 1:]**2.)
        return galcol_err

    def add_zred_fields(self):
        """
        Add default columns for storing zreds.

        Note that this will do nothing if the columns are already there.

        Modifies GalaxyCatalog in place.
        """
        dtype_augment = [dt for dt in zred_extra_dtype if dt[0].lower() not in self.dtype.names]
        if len(dtype_augment) > 0:
            self.add_fields(dtype_augment)

    def match_one(self, ra, dec, radius):
        """
        Match one ra/dec position to the galaxy catalog.

        This is typically used when finding all the neighbors around a cluster
        over a relatively large area.

        Parameters
        ----------
        ra: `float`
           Right ascension to match to.
        dec: `float`
           Declination to match to.
        radius: `float`
           Match radius (degrees)

        Returns
        -------
        indices: `np.array`
           Integer array of GalaxyCatalog indices that match to input ra/dec
        dists: `np.array`
           Float array of distance (degrees) from each galaxy in indices
        """
        if self._htm_matcher is None:
            self._htm_matcher = Matcher(self.depth, self.ra, self.dec)

        _, indices, dists = self._htm_matcher.match(ra, dec, radius, maxmatch=0)

        return indices, dists

    def match_many(self, ras, decs, radius, maxmatch=0):
        """
        Match many ra/dec positions to the galaxy catalog.

        This is typically used when matching a galaxy catalog to another
        catalog (e.g. a spectroscopic catalog).  Running this with a large
        number of positions with a large radii is possible to run out of
        memory!

        Parameters
        ----------
        ras: `np.array`
           Float arrays of right ascensions to match to.
        decs: `np.array`
           Float arrays of declinations to match to.
        radius: `np.array` or `float`
           Float array or float match radius in degrees.
        maxmatch: `int`
           Maximum number of galaxy matches to each ra/dec.
           Default is 0 (no maximum)

        Returns
        -------
        i0: `np.array`
           Integer array of indices matched to input ra/dec
        i1: `np.array`
           Integer array of indices matched to galaxy ra/dec
        dists: `np.array`
           Float array of match distances for each i0/i1 pair (degrees).
        """
        if self._htm_matcher is None:
            self._htm_matcher = Matcher(self.depth, self.ra, self.dec)

        return self._htm_matcher.match(ras, decs, radius, maxmatch=maxmatch)

def get_subpixel_indices(galtable, hpix=None, border=0.0, nside=0):
    """
    Routine to get subpixel indices from a galaxy table.

    Parameters
    ----------
    galtable: `redmapper.Catalog`
       A redmapper galaxy table master catalog
    hpix: `int`, optional
       Healpix number (ring format) of sub-region.  Default is 0 (full catalog).
    border: `float`, optional
       Border around hpix (in degrees) to find pixels.  Default is 0.0.
    nside: `int`, optional
       Nside of healpix subregion.  Default is 0 (full catalog).

    Returns
    -------
    indices: `np.array`
       Integer array of indices of galaxy table pixels in the subregion.
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
    Class to generate a redmapper galaxy catalog from an input catalog.

    The input galaxy catalog must have the following values:

    'id': unique galaxy id ('i8')
    'ra': right ascension ('f8')
    'dec': declination ('f8')
    'refmag': total magnitude in reference band ('f4')
    'refmag_err': error in total reference magnitude ('f4')
    'mag': array of nband magnitudes, sorted by wavelength (e.g. grizy) ('f4', nmag)
    'mag_err': error in magnitudes ('f4', nmag)
    'ebv': E(B-V) at galaxy location (for systematics checks) ('f4')
    'ztrue': ztrue from a simulated catalog (optional) ('f4')
    'm200': m200 of halo from a simulated catalog (optional) ('f4')
    'central': central? 1 if yes (from simulated catalog) (optional) ('i2')
    'halo_id': Unique halo identifier from a simulated catalog (optional) ('i8')

    The typical usage will be:

    maker = redmapper.GalaxyCatalogMaker(filename_base, info_dict)
    for input_file in input_files:
        # insert code to translate to file format
        maker.append_galaxies(galaxies)
    maker.finalize_catalog()
    """

    def __init__(self, outbase, info_dict, nside=32):
        """
        Instantiate a GalaxyCatalogMaker

        Parameters
        ----------
        outbase: `str`
           Output filename base string
        info_dict: `dict`
           Dictionary with following keys:
           ['LIM_REF']: overall limiting magnitude in reference band
           ['REF_IND']: magnitude index for reference band
           ['AREA']: total area of the catalog (deg^2)
           ['NMAG']: number of bands in the catalog
           ['MODE']: Catalog mode (SDSS, DES, LSST...)
           ['ZP']: Reference zeropoint, used if inputs are luptitudes
           ['B']: float array (nmag) of luptitude softening parameters (optional)
           ['U_IND']: u-band index (optional)
           ['G_IND']: g-band index (optional)
           ['R_IND']: r-band index (optional)
           ['I_IND']: i-band index (optional)
           ['Z_IND']: z-band index (optional)
           ['Y_IND']: y-band index (optional)
        nside: `int`
           Split catalog into subpixels with nside of nside.
        """

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

        self.is_finalized = False

    def split_galaxies(self, gals):
        """
        Split a full galaxy catalog into pixels.

        Parameters
        ----------
        gals: `np.ndarray`
           Galaxy catalog structure.  See class docstring for what is in the catalog.
        """
        if self.is_finalized:
            raise RuntimeError("Cannot split galaxies for an already finalized catalog.")

        self.append_galaxies(gals)
        self.finalize_catalog()

    def append_galaxies(self, gals):
        """
        Append a set of galaxies to a galaxy catalog.

        Parameters
        ----------
        gals: `np.ndarray`
           Galaxy catalog structure.  See class docstring for what is in this catalog.
        """
        if self.is_finalized:
            raise RuntimeError("Cannot append galaxies for an already finalized catalog.")

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
        Finish writing a galaxy catalog master table.

        This should be the last step.
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

        self.is_finalized = True
