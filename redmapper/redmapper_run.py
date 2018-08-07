from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import numpy as np
import copy
import fitsio
import re
import os
import esutil
import healpy as hp
from esutil.cosmology import Cosmo

import multiprocessing
from multiprocessing import Pool

import types
try:
    import copy_reg as copyreg
except ImportError:
    import copyreg

from .utilities import _pickle_method

from .catalog import Catalog, Entry
from .run_firstpass import RunFirstPass
from .run_likelihoods import RunLikelihoods
from .run_percolation import RunPercolation

class RedmapperRun(object):
    """
    """

    def __init__(self, config):
        self.config = config.copy()

        # Record the cosmology parameters because we need to rebuild the object
        # in the multiprocessing.
        self._H0 = config.cosmo.H0()
        self._omega_l = config.cosmo.omega_l()
        self._omega_m = config.cosmo.omega_m()
        self.config.cosmo = None

    def run(self, specmode=False, specseed=None, seedfile=None, check=True,
            percolation_only=False, consolidate_like=False):
        """
        """

        self.specmode = specmode
        self.seedfile = seedfile
        self.check = check
        self.percolation_only = percolation_only

        self.single_pixel, nside_split, pixels_split = self._get_pixel_splits()

        # run each individual one
        self.config.d.nside = nside_split

        pool = Pool(processes=self.config.calib_run_nproc)
        if self.specmode:
            retvals = pool.map(self._spec_worker, pixels_split, chunksize=1)
        else:
            retvals = pool.map(self._worker, pixels_split, chunksize=1)
        pool.close()
        pool.join()

        # Consolidate (if necessary)
        if not self.single_pixel:
            pixnums = [x[0] for x in retvals]
            likefiles = [x[2] for x in retvals]
            percfiles = [x[3] for x in retvals]

            self._consolidate(pixnums, percfiles, 'final', members=True, check=check)

            if consolidate_like:
                self._consolidate(pixnums, likefiles, 'like', members=False, check=check)

        # And done

    def _get_pixel_splits(self):
        """
        """

        if self.config.calib_run_nproc == 1:
            return (True, 0, 0)

        tab = Entry.from_fits_file(self.config.galfile, ext=1)

        # We start with the pixel and resolution in the config file
        if self.config.d.hpix == 0:
            nside_splits = [1]
            pixels_splits = [0]
        else:
            nside_splits = [self.config.d.nside]
            pixels_splits = [self.config.d.hpix]

        # We know that we can fit the whole thing in 1 pixel
        nsplit = [1]

        nside_test = nside_splits[0]
        while nside_test < tab.nside:
            # Increment the nside_test
            nside_test *= 2

            # Generate all the pixels
            pixels = np.arange(hp.nside2npix(nside_test))

            # Which of these actually match the parent?
            if self.config.d.hpix > 0:
                theta, phi = hp.pix2ang(self.config.d.nside, self.config.d.hpix)
                hpix_test = hp.ang2pix(nside_test, theta, phi)
                a, b = esutil.numpy_util.match(hpix_test, pixels)
                pixels = pixels[b]

            # And which of these match the galaxies?
            theta, phi = hp.pix2ang(tab.nside, tab.hpix)
            hpix_test = hp.ang2pix(nside_test, theta, phi)
            a, b = esutil.numpy_util.match(pixels, hpix_test)

            nsplit.append(np.unique(a).size)
            nside_splits.append(nside_test)
            pixels_splits.append(np.unique(pixels[a]))

        test, = np.where(np.array(nsplit) <= self.config.calib_run_nproc)
        # the last one is the one we want
        nside_split = nside_splits[test[-1]]
        pixels_split = pixels_splits[test[-1]]

        return (False, nside_split, pixels_split)

    def _consolidate(self, pixnums, filenames, filetype, members=False, check=True):
        """
        """

        outfile = os.path.join(self.config.outpath, '%s_%s.fit' % (self.config.d.outbase, filetype))
        memfile = os.path.join(self.config.outpath, '%s_%s_members.fit' % (self.config.d.outbase, filetype))

        if check:
            outfile_there = os.path.isfile(outfile)
            memfile_there = os.path.isfile(memfile)

            if (outfile_there and memfile_there and members):
                # All files are accounted for
                return
            if outfile_there and not members:
                return

        # How many clusters are there?  (This is the maxmimum before cuts)
        ncluster = 0
        for f in filenames:
            hdr = fitsio.read_header(f, ext=1)
            ncluster += hdr['NAXIS2']

        element = Entry.from_fits_file(filenames[0], ext=1, rows=0)
        dtype = element._ndarray.dtype

        ubercat = Catalog(np.zeros(ncluster, dtype=dtype))
        ctr = 0

        ubermem = None

        for pixnum, f in zip(pixnums, filenames):
            cat = Catalog.from_fits_file(f, ext=1)

            # Cut to minlambda, maxfrac, and within a pixel
            theta = (90.0 - cat.dec) * np.pi / 180.
            phi = cat.ra * np.pi / 180.

            ipring = hp.ang2pix(self.config.d.nside, theta, phi)

            use, = np.where((ipring == pixnum) &
                            (cat.maskfrac < self.config.max_maskfrac) &
                            (cat.Lambda / cat.scaleval > self.config.percolation_minlambda))

            # Make sure we have surviving clusters
            if use.size == 0:
                continue

            cat = cat[use]

            if members:
                parts = f.split('.fit')
                mem = Catalog.from_fits_file(parts[0] + '_members.fit')

                # We are going to replace the mem_match_ids in the consolidated catalog,
                # because the ones generated in the pixels aren't unique
                new_ids = np.arange(use.size, dtype=np.int32) + ctr + 1

                a, b = esutil.numpy_util.match(cat.mem_match_id, mem.mem_match_id)
                cat.mem_match_id = new_ids
                mem.mem_match_id[b] = cat.mem_match_id[a]

                # and we only want to store the members that matched!
                mem = mem[b]

            # And copy the fields
            for n in dtype.names:
                ubercat._ndarray[n][ctr: ctr + cat.size] = cat._ndarray[n]

            if members:
                if ubermem is None:
                    ubermem = mem
                else:
                    ubermem.append(mem)

            ctr += cat.size

        # Crop the catalog to the range that we had clusters
        ubercat = ubercat[0:ctr]

        # Now we need a final sorting by likelihood and mem_match_id replacement
        if members:
            st = np.argsort(ubercat.lnlike)[::-1]
            ubercat = ubercat[st]

            a, b = esutil.numpy_util.match(ubercat.mem_match_id, ubermem.mem_match_id)

            ubercat.mem_match_id = np.arange(ubercat.size, dtype=np.int32) + 1
            ubermem.mem_match_id[b] = ubercat.mem_match_id[a]

        # And write out...
        # We can clobber because if it was already there and we wanted to check
        # that already happened
        fitsio.write(outfile, ubercat._ndarray, clobber=True)

        if members:
            fitsio.write(memfile, ubermem._ndarray, clobber=True)

    def _worker(self, pixnum):

        # Not sure what to do with cosmo...
        config = self.config.copy()
        config.cosmo = Cosmo(H0=self._H0, omega_l=self._omega_l, omega_m=self._omega_m)

        # Set the specific config stuff here
        config.pixnum = pixnum

        if not self.single_pixel:
            config.d.outbase = '%s_%05d' % (self.config.d.outbase, pixnum)

        # Need to add checks about success, and whether a file was output
        #  (especially border pixels in sims)

        firstpass = RunFirstPass(config, specmode=False)
        if os.path.isfile(firstpass.filename) or not self.check:
            # Need to set config.seedfile if necessary
            firstpass.run()
            firstpass.output(savemembers=False, withversion=False, clobber=True)

        config.catfile = firstpass.filename
        # unsure about writing out logging/comments here in parallel bit

        like = RunLikelihoods(config)
        if os.path.isfile(like.filename) or not self.check:
            like.run()
            like.output(savemembers=False, withversion=False, clobber=True)

        config.catfile = like.filename

        perc = RunPercolation(config)
        if os.path.isfile(perc.filename) or not self.check:
            perc.run()
            perc.output(savemembers=True, withversion=False, clobber=True)

        return (pixnum, firstpass.filename, like.filename, perc.filename)

    def _spec_worker(self, pixnum):

        config = self.config.copy()
        config.cosmo = Cosmo(H0=self._H0, omega_l=self._omega_l, omega_m=self._omega_m)

        config.pixnum = pixnum

        if not self.single_pixel:
            config.d.outbase = '%s_%05d' % (self.config.d.outbase, pixnum)

        firstpass = RunFirstPass(config, specmode=True)
        if not os.path.isfile(firstpass.filename) or not self.check:
            firstpass.run()
            firstpass.output(savemembers=False, withversion=False, clobber=True)

        config.catfile = firstpass.filename

        like = RunLikelihoods(config)
        if not os.path.isfile(like.filename) or not self.check:
            like.run(keepz=True)
            like.output(savemembers=False, withversion=False, clobber=True)

        config.catfile = like.filename

        config.percolation_niter = 1
        perc = RunPercolation(config)
        if not os.path.isfile(perc.filename) or not self.check:
            perc.run(keepz=True)
            perc.output(savemembers=True, withversion=False, clobber=True)

        return (pixnum, firstpass.filename, like.filename, perc.filename)

