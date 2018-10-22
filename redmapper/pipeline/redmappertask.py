from __future__ import division, absolute_import, print_function

import os
import numpy as np
import glob

from ..configuration import Configuration
from ..utilities import make_lockfile

class RunRedmapperPixelTask(object):
    """
    """

    def __init__(self, configfile, pixel, nside, path=None):
        if path is None:
            outpath = os.path.dirname(os.path.abspath(configfile))
        else:
            outpath = path

        self.config = Configuration(configfile, outpath=path)
        self.pixel = pixel
        self.nside = nside

    def run(self):
        """
        """

        # need to think about outpath

        # Make sure all files are here and okay...

        if not self.config.galfile_pixelized:
            raise ValueError("Code only runs with pixelized galfile.")

        self.config.check_files(check_zredfile=True, check_bkgfile=True, check_bkgfile_components=True, check_parfile=True, check_zlambdafile=True)

        # Compute the border size

        maxdist = 1.05 * self.config.percolation_rmask_0 * (300. / 100.)**self.config.percolation_rmask_beta
        radius = maxdist / (np.radians(1.) * self.config.cosmo.Da(0, self.config.zrange[0]))

        self.config.border = 3.0 * radius

        self.config.d.hpix = self.pixel
        self.config.d.nside = self.nside
        self.config.d.outbase = '%s_%05d' % (self.config.outbase, self.pixel)
        self.config.outpath = self.path

        # Do the run

        firstpass = RunFirstPass(self.config)

        if not os.path.isfile(firstpass.filename):
            firstpass.run()
            firstpass.output(savemembers=False, withversion=False)
        else:
            print("Firstpass file %s already present.  Skipping..." % (firstpass.filename))

        self.config.catfile = firstpass.filename

        like = RunLikelihoods(config)

        if not os.path.isfile(like.filename):
            like.run()
            like.output(savemembers=False, withversion=False)
        else:
            print("Likelihood file %s already present.  Skipping..." % (like.filename))

        self.config.catfile = like.filename

        perc = RunPercolation(config)

        if not os.path.isfile(perc.filename):
            perc.run()
            perc.output(savemembers=True, withversion=False)
        else:
            print("Percolation file %s already present.  Skipping..." % (perc.filename))

