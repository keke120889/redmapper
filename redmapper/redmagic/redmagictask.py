"""Class to run redmagic, scanning over all pixels"""

from __future__ import division, absolute_import, print_function

import os
import numpy as np
import glob

from ..configuration import Configuration
from .redmagic_selector import RedmagicSelector
from ..catalog import Entry
from ..galaxy import GalaxyCatalog

class RunRedmagicTask(object):
    """
    Class to run redmagic on a full catalog.
    """

    def __init__(self, configfile, path=None):
        """
        Instantiate a RunRedmagicTask

        Parameters
        ----------
        configfile: `str`
           Configuration yaml filename
        path: `str`, optional
           Output path.  Default is None, use same absolute
           path as configfile.
        """
        if path is None:
            outpath = os.path.dirname(os.path.abspath(configfile))
        else:
            outpath = path

        self.config = Configuration(configfile, outpath=path)

    def run(self, modes=None, clobber=False):
        """
        Run redMaGiC selection over a full catalog.

        The modes are optional, if not specified all the modes
        will be run.

        Parameters
        ----------
        modes: `list`, optional
           List of strings of modes to run
        clobber: `bool`, optional
           Overwrite any existing files.  Default is False.
        """

        if not self.config.galfile_pixelized:
            raise ValueError("Code only runs with pixelized galfile.")

        # Prepare the redMaGiC selector
        selector = RedmagicSelector(self.config)

        if modes is None:
            modes = selector.modes

        n_modes = len(modes)

        # Check if files exist, clobber is False
        filenames = [''] * n_modes
        for i, mode in enumerate(modes):
            filenames[i] = self.config.redmapper_filename('redmagic_%s' % (mode))

            if os.path.isfile(filenames[i]) and not clobber:
                raise RuntimeError("redMaGiC file %s already exists, and clobber is False" % (filenames[i]))

        # Loop over all pixels in the galaxy table
        tab = Entry.from_fits_file(self.config.galfile)

        started = [False] * n_modes

        for i, pix in tab.hpix:
            gals = GalaxyCatalog.from_galfile(self.config.galfile,
                                              zredfile=self.config.zredfile,
                                              nside=tab.nside,
                                              hpix=pix,
                                              border=0.0,
                                              truth=True)

            # Loop over all modes
            for j, mode in modes:
                # Select the red galaxies
                red_gals = selector.select_redmagic_galaxies(gals, mode)

                # Spool out redMaGiC galaxies
                if not started[j]:
                    # write a new file (and overwrite if necessary, since we
                    # already did the clobber check)
                    red_gals.to_fits_file(filenames[j], clobber=True)
                else:
                    with fitsio.FITS(filenames[j], mode='rw') as fits:
                        fits[1].append(red_gals._ndarray)

        # All done!  (Except for randoms, tbd...)
