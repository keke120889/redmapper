from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import numpy as np
import copy
import fitsio
import re
import os

import multiprocessing
from multiprocessing import Pool

import types
try:
    import copy_reg as copyreg
except ImportError:
    import copyreg

from .utilities import _pickle_method

from .zred_color import ZredColor
from .galaxy import GalaxyCatalog, get_subpixel_indices, zred_extra_dtype
from .catalog import Catalog, Entry
from .redsequence import RedSequenceColorPar

class ZredRunCatalog(object):
    """
    """

    def __init__(self, config):

        self.config = copy.deepcopy(config)
        self.config.cosmo = None

    def run(self, galaxyfile, outfile, clobber=False, nperproc=None, maxperproc=500000):
        """
        """

        self.galaxyfile = galaxyfile
        self.outfile = outfile

        hdr = fitsio.read_header(self.galaxyfile, ext=1)
        ngal = hdr['NAXIS2']

        zredstr = RedSequenceColorPar(self.config.parfile)
        self.zredc = ZredColor(zredstr, adaptive=True)

        zreds = Catalog(np.zeros(ngal, dtype=[('ZRED', 'f4'),
                                              ('ZRED_E', 'f4'),
                                              ('ZRED2', 'f4'),
                                              ('ZRED2_E', 'f4'),
                                              ('ZRED_UNCORR', 'f4'),
                                              ('ZRED_UNCORR_E', 'f4'),
                                              ('LKHD', 'f4'),
                                              ('CHISQ', 'f4')]))

        if nperproc is None:
            nperproc = int(float(ngal) / (self.config.calib_nproc - 0.1))
            nperproc = np.clip(nperproc, None, maxperproc)

        inds = np.arange(0, ngal, nperproc)
        worker_list = [(ind, np.clip(ind + nperproc, None, ngal)) for ind in inds]

        pool = Pool(processes=self.config.calib_nproc)
        retvals = pool.map(self._worker, worker_list, chunksize=1)
        pool.close()
        pool.join()

        for ind_range, zred, zred_e, zred2, zred2_e, zred_uncorr, zred_uncorr_e, lkhd, chisq in retvals:
            zreds.zred[ind_range[0]: ind_range[1]] = zred
            zreds.zred_e[ind_range[0]: ind_range[1]] = zred_e
            zreds.zred2[ind_range[0]: ind_range[1]] = zred2
            zreds.zred2_e[ind_range[0]: ind_range[1]] = zred2_e
            zreds.zred_uncorr[ind_range[0]: ind_range[1]] = zred_uncorr
            zreds.zred_uncorr_e[ind_range[0]: ind_range[1]] = zred_uncorr_e
            zreds.lkhd[ind_range[0]: ind_range[1]] = lkhd
            zreds.chisq[ind_range[0]: ind_range[1]] = chisq

        zreds.to_fits_file(outfile, clobber=clobber)

    def _worker(self, ind_range):

        # Need a GalaxyCatalog from a fits...
        in_cat = fitsio.read(self.galaxyfile, ext=1, upper=True,
                             rows=np.arange(ind_range[0], ind_range[1]))
        galaxies = GalaxyCatalog(in_cat)
        galaxies.add_zred_fields()

        for g in galaxies:
            self.zredc.compute_zred(g)

        return (ind_range,
                galaxies.zred, galaxies.zred_e,
                galaxies.zred2, galaxies.zred2_e,
                galaxies.zred_uncorr, galaxies.zred_uncorr_e,
                galaxies.lkhd, galaxies.chisq)


class ZredRunPixels(object):
    """
    """

    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.config.cosmo = None

    def run(self, single_process=False, no_zred_table=False, verbose=False):
        """
        """

        if not self.config.galfile_pixelized:
            raise ValueError("Code only runs with a pixelized galfile.")

        self.verbose = verbose
        self.single_process = single_process

        self.zredpath = os.path.dirname(self.config.zredfile)
        self.galpath = os.path.dirname(self.config.galfile)

        test = re.search('^(.*)_zreds_master_table.fit',
                         os.path.basename(self.config.zredfile))
        if test is None:
            raise ValueError("zredfile filename not in proper format (must end with _zreds_master_table.fit)")

        self.outbase = test.groups()[0]

        # Make the output directory if necessary
        if not os.path.exists(self.zredpath):
            os.makedirs(self.zredpath)

        zredstr = RedSequenceColorPar(self.config.parfile)
        self.zredc = ZredColor(zredstr, adaptive=True)

        self.galtable = Entry.from_fits_file(self.config.galfile)
        indices = list(get_subpixel_indices(self.galtable, hpix=self.config.d.hpix, border=self.config.border, nside=self.config.d.nside))

        if not self.single_process:
            pool = Pool(processes=self.config.calib_nproc)
            retvals = pool.map(self._worker, indices, chunksize=1)
            pool.close()
            pool.join()
        else:
            self.ctr = 0
            self.total_galaxies = np.sum(self.galtable.ngals[indices])
            if (self.verbose):
                print("Computing zred for %d galaxies in %d pixels." % (self.total_galaxies, len(indices)))
            retvals = map(self._worker, indices)


        if no_zred_table:
            return retvals

        self.make_zred_table(retvals)

    def _worker(self, index):

        # Read in just one single pixel
        galaxies = GalaxyCatalog.from_galfile(self.config.galfile,
                                              nside=self.galtable.nside,
                                              hpix=self.galtable.hpix[index],
                                              border=0.0)
        galaxies.add_zred_fields()

        if self.single_process:
            ctr = self.ctr
            ngal = self.total_galaxies
        else:
            ctr = 0
            ngal = galaxies.size

        for g in galaxies:
            if self.verbose and (ctr % 1000) == 0:
                print("Computing zred on galaxy %d of %d" % (ctr, ngal))
            self.zredc.compute_zred(g)
            ctr += 1

        if self.single_process:
            self.ctr = ctr

        # And write out the pixel file ... but just the zreds
        zreds = np.zeros(galaxies.size, dtype=zred_extra_dtype)
        for name, t in zred_extra_dtype:
            zreds[name][:] = galaxies._ndarray[name.lower()][:]

        outfile_nopath = '%s_zreds_%07d.fit' % (self.outbase, self.galtable.hpix[index])
        outfile = os.path.join(self.zredpath, outfile_nopath)

        fitsio.write(outfile, zreds, clobber=True)

        return (index, outfile)

    def make_zred_table(self, indices_and_filenames):
        """
        """

        # figure out longest filename
        maxlen = 0
        for index, filename in indices_and_filenames:
            if len(os.path.basename(filename)) > maxlen:
                maxlen = len(os.path.basename(filename))


        gal_dtype = self.galtable.dtype

        zred_dtype = []
        for dt in gal_dtype.descr:
            if dt[0].lower() == 'filenames':
                dt = ('filenames', 'S%d' % (maxlen + 1), dt[2])
            zred_dtype.append(dt)

        zredtable = Entry(np.zeros(1,dtype=zred_dtype))
        for name in gal_dtype.names:
            zredtable._ndarray[name] = self.galtable._ndarray[name]

        zredtable.filenames = ''
        zredtable.ngals = 0

        for index, filename in indices_and_filenames:
            # Make sure file exists
            if not os.path.isfile(filename):
                raise ValueError("Could not find zredfile: %s" % (filename))
            # check size of file
            hdr = fitsio.read_header(filename, ext=1)
            if hdr['NAXIS2'] != self.galtable.ngals[index]:
                raise ValueError("Length mismatch for zredfile: %s" % (filename))

            zredtable.filenames[index] = os.path.basename(filename)
            zredtable.ngals[index] = self.galtable.ngals[index]

        hdr = fitsio.FITSHDR()
        hdr['PIXELS'] = 1

        zredtable.to_fits_file(self.config.zredfile, header=hdr, clobber=True)

