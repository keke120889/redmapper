from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import fitsio
import numpy as np
import healpy as hp
import time
import copy
import os

import multiprocessing
from multiprocessing import Pool

import types
try:
    import copy_reg as copyreg
except ImportError:
    import copyreg

from .catalog import Entry
from .galaxy import GalaxyCatalog
from .redsequence import RedSequenceColorPar
from .chisq_dist import ChisqDist
from .depthmap import DepthMap
from .utilities import interpol, cic
from .utilities import _pickle_method

copyreg.pickle(types.MethodType, _pickle_method)

class Background(object):
    """
    Name:
        Background
    Purpose:
        An object used to hold the background. This also
        contains the functionality to interpolate between
        known background points.

    parameters
    ----------
    filename: string
       background filename
    """

    def __init__(self, filename):
        #"""
        #docstring for the constructor
        #"""
        # Get the raw object background from the fits file
        obkg = Entry.from_fits_file(filename, ext='CHISQBKG')

        # Set the bin size in redshift, chisq and refmag spaces
        self.zbinsize = 0.001
        self.chisqbinsize = 0.5
        self.refmagbinsize = 0.01

        # Create the refmag bins
        refmagbins = np.arange(obkg.refmagrange[0], obkg.refmagrange[1], self.refmagbinsize)
        nrefmagbins = refmagbins.size

        # Create the chisq bins
        nchisqbins = obkg.chisqbins.size
        nlnchisqbins = obkg.lnchisqbins.size

        # Read out the number of redshift bins from the object background
        nzbins = obkg.zbins.size

        # Set up some arrays to populate
        sigma_g_new = np.zeros((nrefmagbins, nchisqbins, nzbins))
        sigma_lng_new = np.zeros((nrefmagbins, nchisqbins, nzbins))

        # Do linear interpolation to get the sigma_g value
        # between the raw background points.
        # If any values are less than 0 then turn them into 0.
        for i in range(nzbins):
            for j in range(nchisqbins):
                sigma_g_new[:,j,i] = np.interp(refmagbins, obkg.refmagbins, obkg.sigma_g[:,j,i])
                sigma_g_new[:,j,i] = np.where(sigma_g_new[:,j,i] < 0, 0, sigma_g_new[:,j,i])
                sigma_lng_new[:,j,i] = np.interp(refmagbins, obkg.refmagbins, obkg.sigma_lng[:,j,i])
                sigma_lng_new[:,j,i] = np.where(sigma_lng_new[:,j,i] < 0, 0, sigma_lng_new[:,j,i])

        sigma_g = sigma_g_new.copy()
        sigma_lng = sigma_lng_new.copy()

        chisqbins = np.arange(obkg.chisqrange[0], obkg.chisqrange[1], self.chisqbinsize)
        nchisqbins = chisqbins.size

        sigma_g_new = np.zeros((nrefmagbins, nchisqbins, nzbins))

        # Now do the interpolation in chisq space
        for i in range(nzbins):
            for j in range(nrefmagbins):
                sigma_g_new[j,:,i] = np.interp(chisqbins, obkg.chisqbins, sigma_g[j,:,i])
                sigma_g_new[j,:,i] = np.where(sigma_g_new[j,:,i] < 0, 0, sigma_g_new[j,:,i])

        sigma_g = sigma_g_new.copy()

        zbins = np.arange(obkg.zrange[0], obkg.zrange[1], self.zbinsize)
        nzbins = zbins.size

        sigma_g_new = np.zeros((nrefmagbins, nchisqbins, nzbins))
        sigma_lng_new = np.zeros((nrefmagbins, nlnchisqbins, nzbins))

        # Now do the interpolation in redshift space
        for i in range(nchisqbins):
            for j in range(nrefmagbins):
                sigma_g_new[j,i,:] = np.interp(zbins, obkg.zbins, sigma_g[j,i,:])
                sigma_g_new[j,i,:] = np.where(sigma_g_new[j,i,:] < 0, 0, sigma_g_new[j,i,:])

        for i in range(nlnchisqbins):
            for j in range(nrefmagbins):
                sigma_lng_new[j,i,:] = np.interp(zbins, obkg.zbins, sigma_lng[j,i,:])
                sigma_lng_new[j,i,:] = np.where(sigma_lng_new[j,i,:] < 0, 0, sigma_lng_new[j,i,:])

        n_new = np.zeros((nrefmagbins, nzbins))
        for i in range(nzbins):
            n_new[:,i] = np.sum(sigma_g_new[:,:,i], axis=1) * self.chisqbinsize

        # Save all meaningful fields
        # to be attributes of the background object.
        self.refmagbins = refmagbins
        self.chisqbins = chisqbins
        self.lnchisqbins = obkg.lnchisqbins
        self.zbins = zbins
        self.sigma_g = sigma_g_new
        self.sigma_lng = sigma_lng_new
        self.n = n_new

    def sigma_g_lookup(self, z, chisq, refmag, allow0=False):
        """
        Name:
            sigma_g_lookup
        Purpose:
            return the value of sigma_g at points in redshift, chisq and refmag space
        Inputs:
            z: redshift
            chisq: chisquared value
            refmag: reference magnitude
        Optional Inputs:
            allow0 (boolean): if we allow sigma_g to be zero 
                and the chisq is very high. Set to False by default.
        Outputs:
            lookup_vals: the looked-up values of sigma_g
        """
        zmin = self.zbins[0]
        chisqindex = np.searchsorted(self.chisqbins, chisq) - 1
        refmagindex = np.searchsorted(self.refmagbins, refmag) - 1
        # Look into changing to searchsorted
        ind = np.clip(np.round((z-zmin)/(self.zbins[1]-zmin)),0, self.zbins.size-1).astype(np.int32)

        badchisq, = np.where((chisq < self.chisqbins[0]) |
                             (chisq > (self.chisqbins[-1] + self.chisqbinsize)))
        badrefmag, = np.where((refmag <= self.refmagbins[0]) |
                              (refmag > (self.refmagbins[-1] + self.refmagbinsize)))

        chisqindex[badchisq] = 0
        refmagindex[badrefmag] = 0

        zindex = np.full_like(chisqindex, ind)
        lookup_vals = self.sigma_g[refmagindex, chisqindex, zindex]
        lookup_vals[badchisq] = np.inf
        lookup_vals[badrefmag] = np.inf

        if not allow0:
            lookup_vals[np.where((lookup_vals == 0) & (chisq > 5.0))] = np.inf
        return lookup_vals

class ZredBackground(object):
    """
    """

    def __init__(self, filename):
        obkg = Entry.from_fits_file(filename, ext='ZREDBKG')

        # Will want to make configurable
        self.refmagbinsize = 0.01
        self.zredbinsize = 0.001

        # Create the refmag bins
        refmagbins = np.arange(obkg.refmagrange[0], obkg.refmagrange[1], self.refmagbinsize)
        nrefmagbins = refmagbins.size

        # Leave the zred bins the same
        nzredbins = obkg.zredbins.size

        # Set up arrays to populate
        # sigma_g_new = np.zeros((nzredbins, nrefmagbins))
        sigma_g_new = np.zeros((nrefmagbins, nzredbins))

        floor = np.min(obkg.sigma_g)

        for i in xrange(nzredbins):
            #sigma_g_new[i, :] = np.clip(interpol(obkg.sigma_g[i, :], obkg.refmagbins, refmagbins), floor, None)
            sigma_g_new[:, i] = np.clip(interpol(obkg.sigma_g[:, i], obkg.refmagbins, refmagbins), floor, None)

        sigma_g = sigma_g_new.copy()

        # And update zred
        zredbins = np.arange(obkg.zredrange[0], obkg.zredrange[1], self.zredbinsize)
        nzredbins = zredbins.size

        #sigma_g_new = np.zeros((nzredbins, nrefmagbins))
        sigma_g_new = np.zeros((nrefmagbins, nzredbins))

        for i in xrange(nrefmagbins):
            #sigma_g_new[:, i] = np.clip(interpol(sigma_g[:, i], obkg.zredbins, zredbins), floor, None)
            sigma_g_new[i, :] = np.clip(interpol(sigma_g[i, :], obkg.zredbins, zredbins), floor, None)

        self.zredbins = zredbins
        self.zredrange = obkg.zredrange
        self.zred_index = 0
        self.refmag_index = 1
        self.refmagbins = refmagbins
        self.refmagrange = obkg.refmagrange
        self.sigma_g = sigma_g_new

    def sigma_g_lookup(self, zred, refmag):
        """
        """

        zredindex = np.searchsorted(self.zredbins, zred) - 1
        refmagindex = np.searchsorted(self.refmagbins, refmag) - 1

        badzred, = np.where((zredindex < 0) |
                            (zredindex >= self.zredbins.size))
        zredindex[badzred] = 0
        badrefmag, = np.where((refmagindex < 0) |
                              (refmagindex >= self.refmagbins.size))
        refmagindex[badrefmag] = 0

        #lookup_vals = self.sigma_g[zredindex, refmagindex]
        lookup_vals = self.sigma_g[refmagindex, zredindex]

        lookup_vals[badzred] = np.inf
        lookup_vals[badrefmag] = np.inf

        return lookup_vals

class BackgroundGenerator(object):
    """
    """

    def __init__(self, config):
        # We need to delete "cosmo" from the config for pickling/multiprocessing
        self.config = copy.deepcopy(config)
        self.config.cosmo = None

    def run(self, clobber=False, natatime=100000, deepmode=False):
        """
        """

        self.natatime = natatime
        self.deepmode = deepmode

        if not clobber:
            if os.path.isfile(self.config.bkgfile):
                with fitsio.FITS(self.config.bkgfile) as fits:
                    if 'CHISQBKG' in [ext.get_extname() for ext in fits[1: ]]:
                        self.config.logger.info("CHISQBKG already in %s and clobber is False" % (self.config.bkgfile))
                        return

        # get the ranges
        self.refmagrange = np.array([12.0, self.config.limmag_catalog])
        self.nrefmagbins = np.ceil((self.refmagrange[1] - self.refmagrange[0]) / self.config.bkg_refmagbinsize).astype(np.int32)
        self.refmagbins = np.arange(self.nrefmagbins) * self.config.bkg_refmagbinsize + self.refmagrange[0]

        self.chisqrange = np.array([0.0, self.config.chisq_max])
        self.nchisqbins = np.ceil((self.chisqrange[1] - self.chisqrange[0]) / self.config.bkg_chisqbinsize).astype(np.int32)
        self.chisqbins = np.arange(self.nchisqbins) * self.config.bkg_chisqbinsize + self.chisqrange[0]

        self.lnchisqbinsize = 0.2
        self.lnchisqrange = np.array([-2.0, 6.0])
        self.nlnchisqbins = np.ceil((self.lnchisqrange[1] - self.lnchisqrange[0]) / self.lnchisqbinsize).astype(np.int32)
        self.lnchisqbins = np.arange(self.nlnchisqbins) * self.lnchisqbinsize + self.lnchisqrange[0]

        self.nzbins = np.ceil((self.config.zrange[1] - self.config.zrange[0]) / self.config.bkg_zbinsize).astype(np.int32)
        self.zbins = np.arange(self.nzbins) * self.config.bkg_zbinsize + self.config.zrange[0]

        # this is the background hist
        sigma_g = np.zeros((self.nrefmagbins, self.nchisqbins, self.nzbins))
        sigma_lng = np.zeros((self.nrefmagbins, self.nlnchisqbins, self.nzbins))

        # We need the areas from the depth map
        depthstr = DepthMap(self.config)
        self.areas = depthstr.calc_areas(self.refmagbins)


        # Split into bins for parallel running
        logrange = np.log(np.array([self.config.zrange[0] - 0.001,
                                    self.config.zrange[1] + 0.001]))
        logbinsize = (logrange[1] - logrange[0]) / self.config.calib_nproc
        zedges = (np.exp(logrange[0]) + np.exp(logrange[1])) - np.exp(logrange[0] + np.arange(self.config.calib_nproc + 1) * logbinsize)

        worker_list = []
        for i in xrange(self.config.calib_nproc):
            ubins, = np.where((self.zbins < zedges[i]) & (self.zbins > zedges[i + 1]))
            gd, = np.where(ubins < self.zbins.size)
            ubins = ubins[gd]

            zbinmark = np.zeros(self.zbins.size, dtype=np.bool)
            zbinmark[ubins] = True

            worker_list.append(zbinmark)

        pool = Pool(processes=self.config.calib_nproc)
        retvals = pool.map(self._worker, worker_list, chunksize=1)
        pool.close()
        pool.join()

        # And store the results
        for zbinmark, sigma_g_sub, sigma_lng_sub in retvals:
            sigma_g[:, :, zbinmark] = sigma_g_sub
            sigma_lng[:, :, zbinmark] = sigma_lng_sub

        # And save them
        dtype = [('zbins', 'f4', self.zbins.size),
                 ('zrange', 'f4', 2),
                 ('zbinsize', 'f4'),
                 ('chisq_index', 'i4'),
                 ('refmag_index', 'i4'),
                 ('chisqbins', 'f4', self.chisqbins.size),
                 ('chisqrange', 'f4', 2),
                 ('chisqbinsize', 'f4'),
                 ('lnchisqbins', 'f4', self.lnchisqbins.size),
                 ('lnchisqrange', 'f4', 2),
                 ('lnchisqbinsize', 'f4'),
                 ('areas', 'f4', self.areas.size),
                 ('refmagbins', 'f4', self.refmagbins.size),
                 ('refmagrange', 'f4', 2),
                 ('refmagbinsize', 'f4'),
                 ('sigma_g', 'f4', sigma_g.shape),
                 ('sigma_lng', 'f4', sigma_lng.shape)]

        chisq_bkg = Entry(np.zeros(1, dtype=dtype))
        chisq_bkg.zbins[:] = self.zbins
        chisq_bkg.zrange[:] = self.config.zrange
        chisq_bkg.zbinsize = self.config.bkg_zbinsize
        chisq_bkg.chisq_index = 0
        chisq_bkg.refmag_index = 1
        chisq_bkg.chisqbins[:] = self.chisqbins
        chisq_bkg.chisqrange[:] = self.chisqrange
        chisq_bkg.chisqbinsize = self.config.bkg_chisqbinsize
        chisq_bkg.lnchisqbins[:] = self.lnchisqbins
        chisq_bkg.lnchisqrange[:] = self.lnchisqrange
        chisq_bkg.lnchisqbinsize = self.lnchisqbinsize
        chisq_bkg.areas[:] = self.areas
        chisq_bkg.refmagbins[:] = self.refmagbins
        chisq_bkg.refmagrange[:] = self.refmagrange
        chisq_bkg.refmagbinsize = self.config.bkg_refmagbinsize
        chisq_bkg.sigma_g[:, :] = sigma_g
        chisq_bkg.sigma_lng[:, :] = sigma_lng

        chisq_bkg.to_fits_file(self.config.bkgfile, extname='CHISQBKG', clobber=clobber)


    def _worker(self, zbinmark):
        """
        """

        starttime = time.time()

        zbins_use = self.zbins[zbinmark]
        zrange_use = np.array([zbins_use[0], zbins_use[-1] + self.config.bkg_zbinsize])

        # We need to load in the red sequence structure -- just in the specific redshift range
        zredstr = RedSequenceColorPar(self.config.parfile, zrange=zrange_use)

        zredstrbinsize = zredstr.z[1] - zredstr.z[0]
        zpos = np.searchsorted(zredstr.z, zbins_use)

        # How many galaxies total?
        if self.config.galfile_pixelized:
            master = Entry.from_fits_file(self.config.galfile)

            if (self.config.d.hpix > 0):
                # We need to take a sub-region
                theta, phi = hp.pix2ang(master.nside, master.hpix)
                ipring_big = hp.ang2pix(self.config.d.nside, theta, phi)
                subreg_indices, = np.where(ipring_big == self.config.d.hpix)
            else:
                subreg_indices = np.arange(master.hpix.size)

            ngal = np.sum(master.ngals[subreg_indices])
            npix = subreg_indices.size
        else:
            hdr = fitsio.read_header(self.config.galfile, ext=1)

            ngal = hdr['NAXIS2']
            npix = 0

        nmag = self.config.nmag
        ncol = nmag - 1

        # default values are all guaranteed to be out of range
        chisqs = np.zeros((ngal, zbins_use.size), dtype=np.float32) + np.exp(np.max(self.lnchisqbins)) + 100.0
        refmags = np.zeros(ngal, dtype=np.float32)

        if (self.deepmode):
            zlimmag = zredstr.mstar(zbins_use + self.config.bkg_zbinsize) - 2.5 * np.log10(0.01)
        else:
            zlimmag = zredstr.mstar(zbins_use + self.config.bkg_zbinsize) - 2.5 * np.log10(0.1)

        bad, = np.where(zlimmag >= self.config.limmag_catalog)
        zlimmag[bad] = self.config.limmag_catalog - 0.01
        zlimmagpos = np.clip(((zlimmag - self.refmagrange[0]) * self.nrefmagbins / (self.refmagrange[1] - self.refmagrange[0])).astype(np.int32), 0, self.nrefmagbins - 1)
        zlimmag = self.refmagbins[zlimmagpos] + self.config.bkg_refmagbinsize

        zbinmid = np.median(np.arange(zredstr.z.size - 1))

        # And the main loop
        ctr = 0
        p = 0
        # This covers both loops
        while ((ctr < ngal) and (p < npix)):
            # Read in a section of the galaxies, or the pixel
            if not self.config.galfile_pixelized:
                lo = ctr
                hi = np.clip(ctr + self.natatime, None, ngal)

                gals = GalaxyCatalog.from_fits_file(self.config.galfile, rows=np.arange(lo, hi))
                ctr = hi + 1
            else:
                if master.ngals[subreg_indices[p]] == 0:
                    p += 1
                    continue

                gals = GalaxyCatalog.from_galfile(self.config.galfile, nside=master.nside,
                                                  hpix=master.hpix[subreg_indices[p]], border=0.0)

                lo = ctr
                hi = ctr + gals.size

                ctr += master.ngals[subreg_indices[p]]
                p += 1

            inds = np.arange(lo, hi)

            refmags[inds] = gals.refmag

            for i, zbin in enumerate(zbins_use):
                use, = np.where((gals.refmag > self.refmagrange[0]) &
                                (gals.refmag < zlimmag[i]))

                if (use.size > 0):
                    # Compute chisq at the redshift zbin
                    chisqs[inds[use], i] = zredstr.calculate_chisq(gals[use], zbin)

        binsizes = self.config.bkg_refmagbinsize  * self.config.bkg_chisqbinsize
        lnbinsizes = self.config.bkg_refmagbinsize * self.lnchisqbinsize

        sigma_g_sub = np.zeros((self.nrefmagbins, self.nchisqbins, zbins_use.size))
        sigma_lng_sub = np.zeros((self.nrefmagbins, self.nlnchisqbins, zbins_use.size))

        for i, zbin in enumerate(zbins_use):
            use, = np.where((chisqs[:, i] >= self.chisqrange[0]) &
                            (chisqs[:, i] < self.chisqrange[1]) &
                            (refmags >= self.refmagrange[0]) &
                            (refmags < self.refmagrange[1]))
            chisqpos = (chisqs[use, i] - self.chisqrange[0]) * self.nchisqbins / (self.chisqrange[1] - self.chisqrange[0])
            refmagpos = (refmags[use] - self.refmagrange[0]) * self.nrefmagbins / (self.refmagrange[1] - self.refmagrange[0])

            value = np.ones(use.size)

            field = cic(value, chisqpos, self.nchisqbins, refmagpos, self.nrefmagbins, isolated=True)
            for j in xrange(self.nchisqbins):
                sigma_g_sub[:, j, i] = field[:, j] / (self.areas * binsizes)

            lnchisqs = np.log(chisqs[:, i])

            use, = np.where((lnchisqs >= self.lnchisqrange[0]) &
                            (lnchisqs < self.lnchisqrange[1]) &
                            (refmags >= self.refmagrange[0]) &
                            (refmags < self.refmagrange[1]))
            lnchisqpos = (lnchisqs[use] - self.lnchisqrange[0]) * self.nlnchisqbins / (self.lnchisqrange[1] - self.lnchisqrange[0])
            refmagpos = (refmags[use] - self.refmagrange[0]) * self.nrefmagbins / (self.refmagrange[1] - self.refmagrange[0])

            value = np.ones(use.size)

            field2 = cic(value, lnchisqpos, self.nlnchisqbins, refmagpos, self.nrefmagbins, isolated=True)

            for j in xrange(self.nlnchisqbins):
                sigma_lng_sub[:, j, i] = field2[:, j] / (self.areas * lnbinsizes)

        self.config.logger.info("Finished %.2f < z < %.2f in %.1f seconds" % (zbins_use[0], zbins_use[-1],
                                                                              time.time() - starttime))

        return (zbinmark, sigma_g_sub, sigma_lng_sub)

class ZredBackgroundGenerator(object):
    """
    """

    def __init__(self, config):
        self.config = config

    def run(self, clobber=False, natatime=100000):
        """
        """

        if not os.path.isfile(self.config.zredfile):
            raise RuntimeError("Must run ZredBackgroundGenerator with a zred file")

        if not clobber:
            if os.path.isfile(self.config.bkgfile):
                with fitsio.FITS(self.config.bkgfile) as fits:
                    if 'ZREDBKG' in [ext.get_extname() for ext in fits[1: ]]:
                        self.config.logger.info("ZREDBKG already in %s and clobber is False" % (self.config.bkgfile))
                        return

        # Read in zred parameters
        zredstr = RedSequenceColorPar(self.config.parfile, fine=True, zrange=self.config.zrange)

        # Set ranges
        refmagrange = np.array([12.0, self.config.limmag_catalog])
        nrefmagbins = np.ceil((refmagrange[1] - refmagrange[0]) / self.config.bkg_refmagbinsize).astype(np.int32)
        refmagbins = np.arange(nrefmagbins) * self.config.bkg_refmagbinsize + refmagrange[0]

        zredrange = np.array([zredstr.z[0], zredstr.z[-2] + (zredstr.z[1] - zredstr.z[0])])
        nzredbins = np.ceil((zredrange[1] - zredrange[0]) / self.config.bkg_zredbinsize).astype(np.int32)
        zredbins = np.arange(nzredbins) * self.config.bkg_zredbinsize + zredrange[0]

        # Compute the areas...
        # This takes into account the configured sub-region
        depthstr = DepthMap(self.config)
        areas = depthstr.calc_areas(refmagbins)

        maxchisq = self.config.wcen_zred_chisq_max

        # Prepare pixels (if necessary) and count galaxies

        if not self.config.galfile_pixelized:
            raise ValueError("Only pixelized galfiles are supported at this moment.")

        master = Entry.from_fits_file(self.config.galfile)

        if self.config.d.hpix > 0:
            # We need to take a sub-region
            theta, phi = hp.pix2ang(master.nside, master.hpix)
            ipring_bin = hp.ang2pix(self.config.d.nside, theta, phi)
            subreg_indices, = np.where(ipring_bin == self.config.d.hpix)
        else:
            subreg_indices = np.arange(master.hpix.size)

        ngal = np.sum(master.ngals[subreg_indices])
        npix = subreg_indices.size

        starttime = time.time()

        nmag = self.config.nmag
        ncol = nmag - 1

        zreds = np.zeros(ngal, dtype=np.float32) - 1.0
        refmags = np.zeros(ngal, dtype=np.float32)

        zbinmid = np.median(np.arange(zredstr.z.size, dtype=np.int32))

        # Loop
        ctr = 0
        p = 0
        while ((ctr < ngal) and (p < npix)):
            if master.ngals[subreg_indices[p]] == 0:
                p += 1
                continue

            gals = GalaxyCatalog.from_galfile(self.config.galfile, nside=master.nside,
                                              hpix=master.hpix[subreg_indices[p]],
                                              border=0.0,
                                              zredfile=self.config.zredfile)

            use, = np.where(gals.chisq < maxchisq)

            if use.size > 0:
                lo = ctr
                hi = ctr + use.size

                inds = np.arange(lo, hi, dtype=np.int64)

                refmags[inds] = gals.refmag[use]
                zreds[inds] = gals.zred[use]

            ctr += master.ngals[subreg_indices[p]]
            p += 1

        # Compute cic
        sigma_g = np.zeros((nrefmagbins, nzredbins))

        binsizes = self.config.bkg_refmagbinsize * self.config.bkg_zredbinsize

        use, = np.where((zreds >= zredrange[0]) & (zreds < zredrange[1]) &
                        (refmags > refmagrange[0]) & (refmags < refmagrange[1]))

        zredpos = (zreds[use] - zredrange[0]) * nzredbins / (zredrange[1] - zredrange[0])
        refmagpos = (refmags[use] - refmagrange[0]) * nrefmagbins / (refmagrange[1] - refmagrange[0])

        value = np.ones(use.size)

        field = cic(value, zredpos, nzredbins, refmagpos, nrefmagbins, isolated=True)

        sigma_g[:, :] = field

        for j in range(nzredbins):
            sigma_g[:, j] = np.clip(field[:, j], 0.1, None) / (areas * binsizes)

        self.config.logger.info("Finished zred background in %.2f seconds" % (time.time() - starttime))

        # save it

        dtype = [('zredbins', 'f4', zredbins.size),
                 ('zredrange', 'f4', zredrange.size),
                 ('zredbinsize', 'f4'),
                 ('zred_index', 'i2'),
                 ('refmag_index', 'i2'),
                 ('refmagbins', 'f4', refmagbins.size),
                 ('refmagrange', 'f4', refmagrange.size),
                 ('refmagbinsize', 'f4'),
                 ('areas', 'f4', areas.size),
                 ('sigma_g', 'f4', sigma_g.shape)]

        zred_bkg = Entry(np.zeros(1, dtype=dtype))
        zred_bkg.zredbins[:] = zredbins
        zred_bkg.zredrange[:] = zredrange
        zred_bkg.zredbinsize = self.config.bkg_zredbinsize
        zred_bkg.zred_index = 0
        zred_bkg.refmag_index = 1
        zred_bkg.refmagbins[:] = refmagbins
        zred_bkg.refmagrange[:] = refmagrange
        zred_bkg.refmagbinsize = self.config.bkg_refmagbinsize
        zred_bkg.areas[:] = areas
        zred_bkg.sigma_g[:, :] = sigma_g

        zred_bkg.to_fits_file(self.config.bkgfile, extname='ZREDBKG', clobber=clobber)

