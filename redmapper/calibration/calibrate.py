from __future__ import division, absolute_import, print_function

import os
import numpy as np
import fitsio

from ..utilities import redmapper_filename
from ..color_background import ColorBackgroundGenerator
from ..catalog import Entry, Catalog
from ..galaxy import GalaxyCatalog
from .selectspecred import SelectSpecRedGalaxies
from .selectspecseeds import SelectSpecSeeds
from .redsequencecal import RedSequenceCalibrator
from .centeringcal import WcenCalibrator
from .zlambdacal import ZLambdaCalibrator
from ..zred_runner import ZredRunCatalog, ZredRunPixels
from ..background import BackgroundGenerator, ZredBackgroundGenerator
from ..redmapper_run import RedmapperRun


class RedmapperCalibrator(object):
    """
    """

    def __init__(self, conf):
        if not isinstance(conf, Configuration):
            self.config = Configuration(conf)
        else:
            self.config = conf

    def run(self):
        """
        """

        # Select the red galaxies to start
        self.config.redgalfile = self.config.redmapper_filename('zspec_redgals')
        self.config.redgalmodelfile = self.config.redmapper_filename('zspec_redgals_model')

        if os.path.isfile(self.config.redgalfile):
            print("%s already there.  Skipping..." % (self.config.redgalfile))
        else:
            print("Selecting red galaxies from spectra...")
            selred = SelectSpecRedGalaxies(self.config)
            selred.run()

        # Make a color background
        self.config.bkgfile_color = self.config.redmapper_filename('bkg_color')

        if os.path.isfile(self.config.bkgfile_color):
            print("%s already there.  Skipping..." % (self.config.bkgfile_color))
        else:
            print("Construction color background...")
            cbg = ColorBackgroundGenerator(config)
            cbg.run()

        # Generate maskgals
        self.config.maskgalfile = self.config.redmapper_filename('maskgals')

        if os.path.isfile(self.config.maskgalfile):
            print("%s already there.  Skipping..." % (self.config.maskgalfile))
        else:
            print("Constructing maskgals...")
            # This will generate the maskgalfile if it isn't found
            mask = get_mask(self.config)

        # Do the color-lambda training.
        self.config.zmemfile = self.config.redmapper_filename('iter0_colormem_pgt%4.2f_lamgt%02d' % (self.config.calib_pcut, self.config.calib_colormem_minlambda))

        if os.path.isfile(self.config.zmemfile):
            print("%s already there.  Skipping..." % (self.config.zmemfile))
        else:
            print("Doing color-lambda training...")
            rcm = RunColormem(self.config)
            rcm.run()
            rcm.output_training()

        # Generate the spec seed file
        self.config.seedfile = self.config.redmapper_filename('specseeds_train')

        if os.path.isfile(self.config.seedfile):
            print("%s already there.  Skipping..." % (self.config.seedfile))
        else:
            print("Generating spectroscopic seeds...")
            sss = SelectSpecSeeds(self.config)
            sss.run()

        calib_iteration = RedmapperCalibIteration(self.config)

        for iteration in range(1, self.config.calib_niter):
            # Run the calibration iteration
            calib_iteration.run(iteration)

            # Clean out the members

            # Generate a new seedfile if this is the first iteration

        # Run the final final iteration

        # Output a configuration file 



class RedmapperCalibrationIteration(object):
    """
    """

    def __init__(self, config):
        self.config = config

    def run(self, iteration):

        # Generate the name of the parfile
        self.config.d.outbase = '%s_iter%d' % (self.config.outbase, iteration)

        self.config.parfile = self.config.redmapper_filename('pars')

        # Run the red sequence calibration
        if os.path.isfile(self.config.parfile):
            print("%s already there.  Skipping..." % (self.config.parfile))
        else:
            redsequencecal = RedSequenceCalibrator(self.config, self.config.zmemfile)
            redsequencecal.run()

        # Make the "sz" file (I think maybe I can skip this)

        # Compute zreds based on the type of galaxy file
        if self.config.galfile_pixelized:
            self.config.zredfile = self.config.redmapper_filename('zreds_master_table', paths=self.config.d.outbase)
        else:
            self.config.zredfile = self.config.redmapper_filename('zreds')

        if os.path.isfile(self.config.zredfile):
            print("%s already there.  Skipping..." % (self.config.zredfile))
        else:
            if self.config.galfile_pixelized:
                zredRunpix = ZredRunPixels(self.config)
                zredRunpix.run()
            else:
                zredRuncat = ZredRunCatalog(self.config)
                zredRuncat.run(self.config.galfile, self.config.zredfile)

        # Compute the chisq background
        self.config.bkgfile = self.config.redmapper_filename('bkg')

        # There are two extensions here, so we need to be careful
        calc_bkg = False
        calc_zred_bkg = False
        if not os.path.isfile(self.config.bkgfile):
            calc_bkg = True
            calc_zred_bkg = True
        else:
            with fitsio.FITS(self.config.bkgfile) as fits:
                if 'CHISQBKG' not in [ext.get_extname() for ext in fits[1: ]]:
                    calc_bkg = True
                else:
                    print("Found CHISQBKG in %s.  Skipping..." % (self.config.bkgfile))
                if 'ZREDBKG' not in [ext.get_extname() for ext in fits[1: ]]:
                    calc_zred_bkg = True
                else:
                    print("Found ZREDBKG in %s.  Skipping..." % (self.config.bkgfile))

        if calc_bkg:
            print("Generating chisq background...")
            bkg_gen = BackgroundGenerator(self.config)
            bkg_gen.run()

        if calc_zred_bkg:
            print("Generating zred background...")
            zbkg_gen = ZredBackgroundGenerator(self.config)
            zbkg_gen.run()

        # Set the centering function
        centerclass = copy.deepcopy(self.config.centerclass)

        if iteration == 1:
            self.config.centerclass = self.config.firstpass_centerclass
        else:
            self.config.centerclass = 'CenteringWcenZred'

        # Generate the zreds for the specseeds
        # This is the iteration seedfile
        iter_seedfile = self.config.redmapper_filename('specseeds')

        if os.path.isfile(iter_seedfile):
            print('%s already there.  Skipping...' % (iter_seedfile))
        else:
            seedzredfile = self.config.redmapper_filename('specseeds_zreds')
            zredRuncat = ZredRunCatalog(self.config)
            zredRuncat.run(self.config.seedfile, seedzredfile)

            # Now combine seeds with zreds
            seeds = Catalog.from_fits_file(self.config.seedfile, ext=1)
            zreds = Catalog.from_fits_file(seedzredfile, ext=1)

            seeds.zred = zreds.zred
            seeds.zred_e = zreds.zred_e
            seeds.zred_chisq = zreds.chisq

            seeds.to_fits_file(iter_seedfile)


        # Run the cluster finder in specmode (And consolidate likelihoods)
        finalfile = self.config.redmapper_filename('final')

        if os.path.isfile(finalfile):
            print('%s already there.  Skipping...' % (finalfile))
        else:
            self.config.zlambdafile = None

            redmapper_run = RedmapperRun(self.config)
            redmapper_run.run(specmode=True, consolidate_like=True, seedfile=iter_seedfile)

        # If it's the first iteration, calibrate random and satellite w functions
        if iteration == 1:
            sublikefile = self.config.redmapper_filename('sub_like')

            if os.path.isfile(sublikefile):
                print('%s already there.  Skipping...' % (sublikefile))
            else:
                # Generate a subset of the likelihood file...
                # Read these as GalaxyCatalogs to do matching
                lcat = GalaxyCatalog.from_fits_file(self.config.redmapper_filename('like'))
                pcat = GalaxyCatalog.from_fits_file(finalfile)

                use, = np.where(pcat.Lambda > self.config.percolation_minlambda)

                # matching...
                lcat.match_many(pcat.ra[use], pcat.dec[use], 0.5/3600., maxmatch=1)

                sublcat = lcat[m2]

                sublcat.to_fits_file(sublikefile)

            catfile_for_rand_calib = self.config.redmapper_filename('rand_final')
            if os.path.isfile(catfile_for_rand_calib):
                print('%s already there.  Skipping...' % (catfile_for_rand_calib))
            else:
                # FIXME: how does rand_final get down there?  FIXME
                self.config.catfile = sublikefile
                self.config.centerclass = 'CenteringRandom'

                redmapper_run = RedmapperRun(self.config)
                ## FIXME: need "keepz" option to be passed
                redmapper_run.run(check=True, percolation_only=True)

                # redmapper_run needs to return a filename, I think???
                ## FIXME
                # Make sure it's actually writing it out, and trace filename!

            catfile_for_randsat_calib = self.config.redmapper_filename('randsat_final')
            if os.path.isfile(catfile_for_randsat_calib):
                print('%s already there.  Skipping...' % (catfile_for_randsat_calib))
            else:
                self.config.catfile = sublikefile
                self.config.centerclass = 'CenteringRandomSatellite'

                redmapper_run = RedmapperRun(self.config)
                redmapper_run.run(check=True, percolation_only=True) ## FIXME

        # Calibrate wcen
        self.config.centerclass = centerclass
        self.config.catfile = finalfile
        self.config.wcenfile = self.config.redmapper_filename('wcen')

        if os.path.isfile(self.config.wcenfile):
            print('%s already there.  Skipping...' % (self.config.wcenfile))
        else:
            wc = WcenCalibrator(self.config, iteration,
                                randcatfile=catfile_for_rand_calib,
                                randsatcatfile=catfile_for_randsat_calib)
            wc.run()

        self.config.set_wcen_vals()

        # Calibrate zlambda correction
        self.config.zlambdafile = self.config.redmapper_filename('zlambda')

        if os.path.isfile(self.config.zlambdafile):
            print('%s already there.  Skipping...' % (self.config.zlambdafile))
        else:
            zlambdacal = ZLambdaCalibrator(self.config, corrslope=False)
            zlambdacal.run()

        # Make pretty plots showing performance

