from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import os
import numpy as np
import fitsio
import time
import scipy.optimize
import esutil
import healpy as hp

from ..configuration import Configuration
from ..utilities import CubicSpline
from ..volumelimit import VolumeLimitMask, VolumeLimitMaskFixed
from ..plotting import SpecPlot, NzPlot
from ..catalog import Catalog, Entry
from ..galaxy import GalaxyCatalog
from ..redsequence import RedSequenceColorPar

class RedmagicRunner(object):
    """
    """

    def __init__(self, conf):
        """
        """
        if not isinstance(conf, Configuration):
            self.config = Configuration(conf)
        else:
            self.config = conf

    def run(self):
        """
        """

        import matplotlib.pyplot as plt

        # Read in parameter files

        calstrs = []
        with fitsio.FITS(self.config.redmagicfile) as fits:
            for ext in fits[1: ]:
                calstrs.append(Entry.from_fits_ext(ext))

        zredstr = RedSequenceColorPar(self.config.parfile, fine=True)

        vlim_masks = []
        vlim_areas = []

        for calstr in calstrs:
            vlim_masks.append()


        # Read in one pixel at a time ...

        # Spool out to a set of *temporary* files

        # While selecting each one

        # And then move temporary files

        # Read in each one, and do plots and statistics

        
