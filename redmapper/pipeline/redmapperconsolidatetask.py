from __future__ import division, absolute_import, print_function

import os
import numpy as np
import glob
import fitsio
import re
import healpy as hp
import esutil

from ..configuration import Configuration
from ..volumelimit import VolumeLimitMask, VolumeLimitMaskFixed
from ..catalog import Catalog
from ..utilities import read_members, astro_to_sphere
from ..plotting import SpecPlot, NzPlot
from ..galaxy import GalaxyCatalog

class RedmapperConsolidateTask(object):
    """
    """

    def __init__(self, configfile, lambda_cuts=None, vlim_lstars=None, path=None):
        if path is None:
            outpath = os.path.dirname(os.path.abspath(configfile))
        else:
            outpath = path

        self.config = Configuration(configfile, outpath=path)

        if lambda_cuts is None:
            self.lambda_cuts = self.config.consolidate_lambda_cuts
        else:
            self.lambda_cuts = lambda_cuts

        if vlim_lstars is None:
            if self.config.consolidate_vlim_lstars[0] is None:
                self.vlim_lstars = None
            else:
                self.vlim_lstars = self.config.consolidate_vlim_lstars
        else:
            self.vlim_lstars = vlim_lstars

    def run(self, do_plots=True, match_spec=True):
        """
        """

        # find the files
        catfiles = sorted(glob.glob(os.path.join(self.config.outpath, '%s_*_?????_final.fit' % (self.config.outbase))))

        self.config.logger.info("Found %d catalog files in %s" % (len(catfiles), self.config.outpath))

        # Extract the nside that was run
        m = re.search('_(\d+)_(\d\d\d\d\d)_', catfiles[0])
        if m is None:
            raise RuntimeError("Could not understand filename for %s" % (catfiles[0]))

        nside = int(m.groups()[0])

        if match_spec:
            spec = GalaxyCatalog.from_fits_file(self.config.specfile)
            use, = np.where(spec.z_err < 0.001)
            spec = spec[use]

        # Add check for the number of files that are here

        # FIXME: add check for the number of files that are here!

        # Set the "outbase" name
        # Note that the vlim will not be per lambda!

        # If we have vlim set, we need to make the vlims...
        if self.vlim_lstars is not None:
            vlim_masks = []
            vlim_areas = []
            for vlim_lstar in self.vlim_lstars:
                vlim_masks.append(VolumeLimitMask(self.config, vlim_lstar))
                vlim_areas.append(vlim_masks[-1].get_areas())
        else:
            vlim_masks = [VolumeLimitMaskFixed(self.config)]
            vlim_areas = [vlim_masks[0].get_areas()]

        started = np.zeros((len(self.lambda_cuts), len(vlim_masks)), dtype=np.bool)
        cat_filename_dict = {}

        # Unique counter for temporary ids
        ctr = 0

        all_ids = np.zeros(0, dtype=np.int32)
        all_likelihoods = np.zeros(0, dtype=np.float64)

        for catfile in catfiles:
            # Read in catalog
            self.config.logger.info("Reading %s" % (os.path.basename(catfile)))
            cat = Catalog.from_fits_file(catfile, ext=1)

            # and read in members
            mem = read_members(catfile)

            if match_spec:
                # match spec to cat and mem
                cat.cg_spec_z[:] = -1.0

                i0, i1, dists = spec.match_many(cat.ra, cat.dec, 3./3600., maxmatch=1)
                cat.cg_spec_z[i0] = spec.z[i1]

                mem.zspec[:] = -1.0
                i0, i1, dists = spec.match_many(mem.ra, mem.dec, 3./3600., maxmatch=1)
                mem.zspec[i0] = spec.z[i1]

            # Extract pixnum from name

            m = re.search('_(\d+)_(\d\d\d\d\d)_', catfile)
            if m is None:
                raise RuntimeError("Could not understand filename for %s" % (catfile))

            hpix = int(m.groups()[1])

            # Note: when adding mock support, read in true galaxies to get members

            scaleval = cat.scaleval if self.config.select_scaleval else 1.0

            # Figure out which clusters are in the pixel, less than
            # max_maskfrac, and above percolation_minlambda (with or without
            # scale)

            theta, phi = astro_to_sphere(cat.ra, cat.dec)
            ipring = hp.ang2pix(nside, theta, phi)

            use, = np.where((ipring == hpix) &
                            (cat.maskfrac < self.config.max_maskfrac) &
                            (cat.Lambda > self.config.percolation_minlambda))

            if use.size == 0:
                self.config.logger.info('Warning: no good clusters in pixel %d' % (hpix))
                continue

            cat = cat[use]

            # match catalog with members via mem_match_id
            a, b = esutil.numpy_util.match(cat.mem_match_id, mem.mem_match_id)

            # put in new, temporary IDs
            cat.mem_match_id = np.arange(use.size) + ctr
            mem.mem_match_id[b] = cat.mem_match_id[a]
            ctr += use.size

            # Cut down members
            mem = mem[b]

            # Append ALL IDs and likelihoods for sorting here
            all_ids = np.append(all_ids, cat.mem_match_id)
            all_likelihoods = np.append(all_likelihoods, cat.lnlamlike)

            # loop over minlambda
            for i, minlambda in enumerate(self.lambda_cuts):
                # loop over vlim_masks
                # (note that if we have no mask we have the "fixed" mask placeholder)

                for j, vlim_mask in enumerate(vlim_masks):
                    cat_use, = np.where((vlim_mask.calc_zmax(cat.ra, cat.dec) > cat.z_lambda) &
                                        ((cat.Lambda / scaleval) > minlambda))

                    if cat_use.size == 0:
                        continue

                    _, mem_use = esutil.numpy_util.match(cat.mem_match_id[cat_use], mem.mem_match_id)

                    if not started[i, j]:
                        # Figure out filename
                        self.config.d.outbase = '%s_redmapper_v%s_lgt%02d' % (self.config.outbase, self.config.version, minlambda)
                        if self.vlim_lstars is not None:
                            self.config.d.outbase += '_vl%02d' % (int(self.vlim_lstars[j] * 10))

                        cat_fname = self.config.redmapper_filename('catalog')
                        mem_fname = self.config.redmapper_filename('catalog_members')

                        cat_filename_dict[(i, j)] = (self.config.d.outbase, cat_fname, mem_fname)

                        # Write out new fits files...
                        cat.to_fits_file(cat_fname, clobber=True, indices=cat_use)
                        mem.to_fits_file(mem_fname, clobber=True, indices=mem_use)

                        started[i, j] = True
                    else:
                        # Append to existing fits files
                        fits = fitsio.FITS(cat_filename_dict[(i, j)][1], mode='rw')
                        fits[1].append(cat._ndarray[cat_use])
                        fits.close()

                        fits = fitsio.FITS(cat_filename_dict[(i, j)][2], mode='rw')
                        fits[1].append(mem._ndarray[mem_use])
                        fits.close()

        # Sort and renumber ...
        st = np.argsort(all_likelihoods)[::-1]

        all_ids_sorted = all_ids[st]
        new_ids = np.arange(all_ids.size) + 1

        # Now we have the index of all_ids_sorted to new_ids

        # Read in specified columns and overwrite in each case

        for i, minlambda in enumerate(self.lambda_cuts):
            for j, vlim_mask in enumerate(vlim_masks):
                catfits = fitsio.FITS(cat_filename_dict[(i, j)][1], mode='rw')
                memfits = fitsio.FITS(cat_filename_dict[(i, j)][2], mode='rw')

                cat_ids = catfits[1].read_column('mem_match_id')
                mem_ids = memfits[1].read_column('mem_match_id')

                # Every time we see an id we replace it
                aa, bb = esutil.numpy_util.match(all_ids_sorted, cat_ids)
                cat_ids[bb] = new_ids[aa]
                aa, bb = esutil.numpy_util.match(all_ids_sorted, mem_ids)
                mem_ids[bb] = new_ids[aa]

                catfits[1].write_column('mem_match_id', cat_ids)
                memfits[1].write_column('mem_match_id', mem_ids)

                catfits.close()
                memfits.close()

                if do_plots:
                    # We want to plot the zspec plot and the n(z) plot
                    cat = Catalog.from_fits_file(cat_filename_dict[(i, j)][1])

                    self.config.d.outbase = cat_filename_dict[(i, j)][0]
                    specplot = SpecPlot(self.config)
                    specplot.plot_cluster_catalog(cat, title=self.config.d.outbase)

                    nzplot = NzPlot(self.config)
                    nzplot.plot_cluster_catalog(cat, areastr=vlim_areas[j])
