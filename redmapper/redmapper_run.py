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
            percolation_only=False, consolidate_like=False, keepz=False):
        """
        """

        self.specmode = specmode
        self.seedfile = seedfile
        self.check = check
        self.percolation_only = percolation_only
        self.keepz = keepz

        if self.specmode and not self.keepz:
            raise RuntimeError("Must set keepz=True when specmode=True")
        if self.percolation_only and self.specmode:
            raise RuntimeError("Cannot set both percolation_only=True and specmode=True")

        nside_split, pixels_split = self._get_pixel_splits()

        print("Running on %d pixels" % (len(pixels_split)))

        # run each individual one
        self.config.d.nside = nside_split

        orig_seedfile = self.config.seedfile
        if seedfile is not None:
            # Use the specified seedfile if desired
            self.config.seedfile = seedfile
        pool = Pool(processes=self.config.calib_run_nproc)
        if self.percolation_only:
            #retvals = pool.map(self._percolation_only_worker, pixels_split, chunksize=1)
            retvals = map(self._percolation_only_worker, pixels_split)
        else:
            #retvals = pool.map(self._worker, pixels_split, chunksize=1)
            retvals = map(self._worker, pixels_split)
        pool.close()
        pool.join()

        # Reset the seedfile
        self.config.seedfile = orig_seedfile

        # Consolidate (adds additional mask cuts)
        hpixels = [x[0] for x in retvals]
        likefiles = [x[2] for x in retvals]
        percfiles = [x[3] for x in retvals]

        finalfile = self._consolidate(hpixels, percfiles, 'final', members=True, check=check)

        if consolidate_like:
            likefile = self._consolidate(hpixels, likefiles, 'like', members=False, check=check)

        # And done
        if consolidate_like:
            return (finalfile, likefile)
        else:
            return finalfile

    def _get_pixel_splits(self):
        """
        """

        if self.config.calib_run_nproc == 1:
            return (self.config.d.nside, [self.config.d.hpix])

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
                theta, phi = hp.pix2ang(nside_test, pixels)
                hpix_test = hp.ang2pix(self.config.d.nside, theta, phi)
                a, b = esutil.numpy_util.match(self.config.d.hpix, hpix_test)
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

        return (nside_split, pixels_split)

    def _consolidate(self, hpixels, filenames, filetype, members=False, check=True):
        """
        """

        outfile = self.config.redmapper_filename(filetype)
        memfile = self.config.redmapper_filename(filetype+'_members')

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

        for hpix, f in zip(hpixels, filenames):
            cat = Catalog.from_fits_file(f, ext=1)

            # Cut to minlambda, maxfrac, and within a pixel
            theta = (90.0 - cat.dec) * np.pi / 180.
            phi = cat.ra * np.pi / 180.

            if self.config.d.nside > 0:
                ipring = hp.ang2pix(self.config.d.nside, theta, phi)
            else:
                # Set all the pixels to 0
                ipring = np.zeros(cat.size, dtype=np.int32)

            use, = np.where((ipring == hpix) &
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
        ubercat.to_fits_file(outfile, clobber=True)

        if members:
            ubermem.to_fits_file(memfile, clobber=True)

        return outfile

    def _worker(self, hpix):

        print("Running on pixel %d" % (hpix))

        # Not sure what to do with cosmo...
        config = self.config.copy()
        config.cosmo = Cosmo(H0=self._H0, omega_l=self._omega_l, omega_m=self._omega_m)

        # Set the specific config stuff here
        config.d.hpix = hpix

        config.d.outbase = '%s_%05d' % (self.config.d.outbase, hpix)

        # Need to add checks about success, and whether a file was output
        #  (especially border pixels in sims)
        firstpass = RunFirstPass(config, specmode=self.specmode)

        if not os.path.isfile(firstpass.filename) or not self.check:
            firstpass.run(keepz=self.keepz)
            firstpass.output(savemembers=False, withversion=False, clobber=True)
        else:
            print("Firstpass file %s already present.  Skipping..." % (firstpass.filename))

        config.catfile = firstpass.filename

        like = RunLikelihoods(config)

        if not os.path.isfile(like.filename) or not self.check:
            like.run(keepz=self.keepz)
            like.output(savemembers=False, withversion=False, clobber=True)
        else:
            print("Likelihood file %s already present.  Skipping..." % (like.filename))

        config.catfile = like.filename

        perc = RunPercolation(config)

        if not os.path.isfile(perc.filename) or not self.check:
            perc.run(keepz=self.keepz)
            perc.output(savemembers=True, withversion=False, clobber=True)
        else:
            print("Percolation file %s already present.  Skipping..." % (perc.filename))

        return (hpix, firstpass.filename, like.filename, perc.filename)

    def _percolation_only_worker(self, hpix):
        config = self.config.copy()
        config.cosmo = Cosmo(H0=self._H0, omega_l=self._omega_l, omega_m=self._omega_m)

        config.d.hpix = hpix

        config.d.outbase = '%s_%05d' % (self.config.d.outbase, hpix)

        perc = RunPercolation(config)
        if not os.path.isfile(perc.filename) or not self.check:
            perc.run(keepz=self.keepz)
            perc.output(savemembers=True, withversion=False, clobber=True)

        return (hpix, None, None, perc.filename)
