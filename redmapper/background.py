import fitsio
import numpy as np
from catalog import Entry
import scipy.ndimage as ndi
# from scipy.interpolate import RegularGridInterpolator
from utilities import interpol


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
