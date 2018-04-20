from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import fitsio
import numpy as np

from .catalog import Entry
from .utilities import interpol

class ColorBackground(object):
    """
    """

    def __init__(self, filename, usehdrarea=False):

        refmagbinsize = 0.01
        colbinsize = 0.05
        area = 1.0

        fits = fitsio.FITS(filename)

        self.bkgs = {}

        started = False

        for ext in fits[1: ]:
            extname = ext.get_extname()

            parts = extname.split('_')
            iind = int(parts[0])
            jind = int(parts[1])

            key = iind * 100 + jind

            obkg = Entry.from_fits_ext(ext)

            # Create the refmag bins
            refmagbins = np.arange(obkg.refmagrange[0], obkg.refmagrange[1], refmagbinsize)
            nrefmagbins = refmagbins.size

            # Deal with area if necessary
            if not started:
                if usehdrarea:
                    if 'areas' in obkg.dtype.names:
                        areas = interpol(obkg.areas, obkg.refmagbins, refmagbins)
                    else:
                        hdr = ext.read_header()
                        areas = np.zeros(nrefmagbins) + hdr['AREA']
                else:
                    areas = np.zeros(nrefmagbins) + area

                started = True

            if iind == jind:
                # We are on a diagonal

                # First do the refmag
                ncolbins = obkg.colbins.size
                bc_new = np.zeros((nrefmagbins, ncolbins))
                for i in xrange(ncolbins):
                    bc_new[:, i] = np.clip(interpol(obkg.bc[:, i], obkg.refmagbins, refmagbins), 0.0, None)

                bc = bc_new.copy()

                # Now do the color
                colbins = np.arange(obkg.colrange[0], obkg.colrange[1], colbinsize)
                ncolbins = colbins.size

                bc_new = np.zeros((nrefmagbins, ncolbins))
                for j in xrange(nrefmagbins):
                    bc_new[j, :] = np.clip(interpol(bc[j, :], obkg.colbins, colbins), 0.0, None)

                n = np.sum(bc, axis=1) * (colbinsize / obkg.colbinsize)

                sigma_g = bc_new.copy()
                for j in xrange(ncolbins):
                    sigma_g[:, j] = bc_new[:, j] / areas

                self.bkgs[key] = {'col1': iind,
                                  'col2': jind,
                                  'refmagindex': 1,
                                  'colbins': colbins,
                                  'colrange': obkg.colrange,
                                  'colbinsize': colbinsize,
                                  'refmagbins': refmagbins,
                                  'refmagrange': obkg.refmagrange,
                                  'refmagbinsize': refmagbinsize,
                                  'bc': bc_new,
                                  'n': n,
                                  'sigma_g': sigma_g}
            else:
                # We are on an off-diagonal

                # start with the refmag
                ncol1bins = obkg.col1bins.size
                ncol2bins = obkg.col2bins.size
                bc_new = np.zeros((nrefmagbins, ncol2bins, ncol1bins))
                for i in xrange(ncol1bins):
                    for j in xrange(ncol2bins):
                        bc_new[:, j, i] = np.clip(interpol(obkg.bc[:, j, i], obkg.refmagbins, refmagbins), 0.0, None)

                bc = bc_new.copy()

                # color1
                col1bins = np.arange(obkg.col1range[0], obkg.col1range[1], colbinsize)
                ncol1bins = col1bins.size

                bc_new = np.zeros((nrefmagbins, ncol2bins, ncol1bins))
                for j in xrange(ncol2bins):
                    for k in xrange(nrefmagbins):
                        bc_new[k, j, :] = np.clip(interpol(bc[k, j, :], obkg.col1bins, col1bins), 0.0, None)

                bc = bc_new.copy()

                col2bins = np.arange(obkg.col2range[0], obkg.col2range[1], colbinsize)
                ncol2bins = col2bins.size

                bc_new = np.zeros((nrefmagbins, ncol2bins, ncol1bins))
                for i in xrange(ncol1bins):
                    for k in xrange(nrefmagbins):
                        bc_new[k, :, i] = np.clip(interpol(bc[k, :, i], obkg.col2bins, col2bins), 0.0, None)

                temp = np.sum(bc_new, axis=1) * colbinsize
                n = np.sum(temp, axis=1) * colbinsize

                sigma_g = bc_new.copy()

                for j in xrange(ncol1bins):
                    for k in xrange(ncol2bins):
                        sigma_g[:, k, j] = bc_new[:, k, j] / areas

                self.bkgs[key] = {'col1': iind,
                                  'col2': jind,
                                  'refmag_index': 2,
                                  'col1bins': col1bins,
                                  'col1range': obkg.col1range,
                                  'col1binsize': colbinsize,
                                  'col2bins': col2bins,
                                  'col2range': obkg.col2range,
                                  'col2binsize': colbinsize,
                                  'refmagbins': refmagbins,
                                  'refmagrange': obkg.refmagrange,
                                  'refmagbinsize': refmagbinsize,
                                  'bc': bc_new,
                                  'n': n,
                                  'sigma_g': sigma_g}

    def lookup_diagonal(self, bkg_index, colors, refmags):
        """
        """
        bkg = self.bkgs[bkg_index * 100 + bkg_index]

        refmagindex = np.clip(np.searchsorted(bkg['refmagbins'], refmags - bkg['refmagbinsize']), 0, bkg['refmagbins'].size - 1)
        col_index = np.clip(np.searchsorted(bkg['colbins'], colors - bkg['colbinsize']), 0, bkg['colbins'].size - 1)

        return bkg['bc'][refmagindex, col_index] / bkg['n'][refmagindex]

    def lookup_offdiag(self, bkg_index1, bkg_index2, colors1, colors2, refmags):
        """
        """
        key = bkg_index1 * 100 + bkg_index2
        if key not in self.bkgs:
            key = bkg_index2 * 100 + bkg_index1

        bkg = self.bkgs[key]

        refmagindex = np.clip(np.searchsorted(bkg['refmagbins'], refmags - bkg['refmagbinsize']), 0, bkg['refmagbins'].size - 1)
        col_index1 = np.clip(np.searchsorted(bkg['col1bins'], colors1 - bkg['col1binsize']), 0, bkg['col1bins'].size - 1)
        col_index2 = np.clip(np.searchsorted(bkg['col2bins'], colors2 - bkg['col2binsize']), 0, bkg['col2bins'].size - 1)

        return bkg['bc'][refmagindex, col_index2, col_index1] / bkg['n'][refmagindex]

class ColorBackgroundGenerator(object):
    """
    """

    def __init__(self, config):
        self.config = config

    def run(self, clobber=False):
        """
        """

        # Check if it's already there...
        if not clobber and os.path.isfile(self.config.bkgfile_color):
            print("Found %s and clobber is False" % (self.config.bkgfile_color))

        # read in the galaxies
        gals = GalaxyCatalog.from_galfile(self.config.galfile,
                                          nside=self.config.nside,
                                          hpix=self.config.hpix,
                                          border=self.config.border)

        # Generate ranges based on the data
        refmagbinsize = 0.1

        refmagrange = np.array([12.0, self.config.limmag])

        nmag = self.config.nmag
        ncol = nmag - 1

        col = gals.galcol

        colrange_default = np.array([-2.0, 5.0])

        colranges = np.zeros((2, ncol))
        colbinsize = 0.1
        for i in xrange(ncol):
            use, = np.where((col[:, i] > colrange_default[0]) &
                            (col[:, i] < colrange_default[1]) &
                            (gals.refmag < (self.config.limmag - 0.5)))

            h = esutil.stat.histogram(col[use, i], min=colrange_default[0],
                                      max=colrange_default[1], binsize=colbinsize)
            bins = np.arange(h.size) * colbinsize + colrange_default[0]

            good, = np.where(h > 1000)

            colranges[0, i] = np.min(bins[good])
            colranges[1, i] = np.max(bins[good]) + colbinsize

        nrefmag = np.ceil((refmagrange[1] - refmagrange[0]) / refmagbinsize).astype(np.int32)
        refmagbins = np.arange(nrefmag) * refmagbinsize + refmagrange[0]

        


