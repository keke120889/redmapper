import fitsio
import numpy as np
from catalog import Entry
import scipy.ndimage as ndi
# from scipy.interpolate import RegularGridInterpolator


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

    # def _interp(data)

    # def _interp_axis(data, axis, oldx, newx):
    #     axes = (ax for ax in range(len(data.shape)) if ax != axis) + (axis,)
    #     data, new_data = np.transpose(data, axes)
    #     for datum in np.transpose(data, axes).reshape(y*z, len(oldx)):
    #         new_data[]
    #     return np.reshape()

    # def __init__(self, filename):
    #     obkg = Entry.from_fits_file(filename)
    #     self.zbinsize, self.chisqbinsize, self.refmagbinsize = 0.001, 0.5, 0.01
    #     self.lnchisqbins = obkg.lnchisqbins
    #     self.zbins = np.arange(obkg.zrange[0], obkg.zrange[1], self.zbinsize)
    #     self.chisqbins = np.arange(obkg.chisqrange[0], obkg.chisqrange[1], 
    #                                                         self.chisqbinsize)
    #     self.refmagbins = np.arange(obkg.refmagrange[0], obkg.refmagrange[1], 
    #                                                         self.refmagbinsize)
    #     self.obkg = obkg

    def __init__(self, filename):
        #"""
        #docstring for the constructor
        #"""
        # Get the raw object background from the fits file
        obkg = Entry.from_fits_file(filename)
        
        # Set the bin size in redshift, chisq and refmag spaces
        self.zbinsize, self.chisqbinsize, self.refmagbinsize = 0.001, 0.5, 0.01

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
        ind = np.clip(np.round((z-zmin)/(self.zbins[1]-zmin)),0, self.zbins.size-1)

        badchisq  = np.where((chisq < self.chisqbins[0])| (chisq > self.chisqbins[-1]))
        badrefmag = np.where((refmag <= self.refmagbins[0]) 
                             | (refmag >= self.refmagbins[-1]))
        chisqindex[badchisq] = 0
        refmagindex[badrefmag] = 0

        zindex = np.full_like(chisqindex, ind)
        lookup_vals = self.sigma_g[refmagindex, chisqindex, zindex]
        lookup_vals[badchisq] = np.inf
        lookup_vals[badrefmag] = np.inf

        if not allow0:
            lookup_vals[np.where((lookup_vals == 0) & (chisq > 5.0))] = np.inf
        return lookup_vals

# # Alternate method of interpolation
# class AltAltBackground(object):
#     """Docstring."""

#     # @staticmethod
#     # def _interp(x, y, z, values, xnew=None, ynew=None, znew=None):
#     #     if xnew is None: xnew = x
#     #     if ynew is None: ynew = y
#     #     if znew is None: znew = z
#     #     xpts, ypts, zpts = np.meshgrid(xnew, ynew, znew)
#     #     coord = np.vstack((xpts.flatten(), ypts.flatten(), zpts.flatten())).T
#     #     interp_fn = RegularGridInterpolator((x, y, z), values, 
#     #                                         bounds_error=False, 
#     #                                         fill_value=None)
#     #     result = np.swapaxes(np.reshape(interp_fn(coord), 
#     #                                 (len(ynew), len(xnew), len(znew))), 0, 1)
#     #     return np.where(result < 0, 0, result)

#     @staticmethod
#     def _interp2(x, y, z, values, dx, dy, dz):
#         scaling, offset = np.array([dx, dy, dz]), np.array([x[0], y[0], z[0]])
#         coord = np.vstack((arr.flatten() for arr in np.meshgrid(x, y, z)))
#         idx = (coord.T - offset).T / scaling[(slice(None),) + 
#                                                 (None,)*(coord.ndim-1)]
#         new_values = ndi.map_coordinates(values, idx, mode='nearest')
#         result = np.swapaxes(np.reshape(new_values, 
#                                     (len(y), len(x), len(z))), 0, 1)
#         return np.where(result < 0, 0, result)

#     def __init__(self, filename):
#         obkg = Entry.from_fits_file(filename)
#         self.zbinsize, self.chisqbinsize, self.refmagbinsize = 0.001, 0.5, 0.01
        
#         self.lnchisqbins = obkg.lnchisqbins
#         self.zbins = np.arange(obkg.zrange[0], obkg.zrange[1], self.zbinsize)
#         self.chisqbins = np.arange(obkg.chisqrange[0], obkg.chisqrange[1], 
#                                                             self.chisqbinsize)
#         self.refmagbins = np.arange(obkg.refmagrange[0], obkg.refmagrange[1], 
#                                                             self.refmagbinsize)
#         self.obkg = obkg
        
#         # self.sigma_g = AltBackground._interp(obkg.refmagbins, obkg.chisqbins, 
#         #                                   obkg.zbins, obkg.sigma_g, 
#         #                                   xnew=self.refmagbins, 
#         #                                   ynew=self.chisqbins, 
#         #                                   znew=self.zbins)
#         # self.sigma_lng = AltBackground._interp(obkg.refmagbins, obkg.lnchisqbins, 
#         #                                     obkg.zbins, obkg.sigma_lng, 
#         #                                     xnew=self.refmagbins, 
#         #                                     znew=self.zbins)

#         self.sigma_g = AltBackground._interp2(self.refmagbins, self.chisqbins,
#                                 self.zbins, obkg.sigma_g, obkg.refmagbinsize,
#                                 obkg.chisqbinsize, obkg.zbinsize)
#         self.sigma_lng = AltBackground._interp2(self.refmagbins, self.lnchisqbins,
#                                 self.zbins, obkg.sigma_lng, obkg.refmagbinsize,
#                                 obkg.lnchisqbinsize, obkg.zbinsize)
#         self.n = np.sum(self.sigma_g, axis=1) * self.chisqbinsize

