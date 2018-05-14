from __future__ import division, absolute_import, print_function

import os
import numpy as np
import fitsio
from pkg_resources import resource_filename
from pkg_resources import resource_exists

from ..configuration import Configuration
from ..fitters import MedZFitter, RedSequenceFitter, EcgmmFitter
from ..galaxy import GalaxyCatalog
from ..catalog import Catalog
from ..utilities import make_nodes, CubicSpline, interpol

class SelectSpecRedGalaxies(object):
    """
    """

    def __init__(self, conf):
        if not isinstance(conf, Configuration):
            self.config = Configuration(conf)
        else:
            self.config = conf

        self._template_file = None
        module = __name__.split('.')[0]
        if resource_exists(module, 'data/initcolors/%s' % (self.config.calib_redgal_template)):
            self._template_file = resource_filename(module, 'data/initcolors/%s' % (self.config.calib_redgal_template))
        elif os.path.isfile(self.config.calib_redgal_template):
            self._template_file = os.path.abspath(self.config.calib_redgal_template)
        else:
            raise IOError("Could not find calib_redgal_template file %s in resource or path" % (self.config.calib_redgal_template))


    def run(self):
        """
        """

        # first, we read the galaxies
        gals = GalaxyCatalog.from_galfile(self.config.galfile,
                                          nside=self.config.nside,
                                          hpix=self.config.hpix,
                                          border=self.config.border)

        # and we need the spectra
        spec = Catalog.from_fits_file(self.config.specfile_train)

        # select good spectra
        use, = np.where(spec.z_err < 0.001)
        spec = spec[use]

        # Match spectra to galaxies
        i0, i1, dists = gals.match_many(spec.ra, spec.dec, 3./3600.0, maxmatch=1)

        # Make a specific galaxy table
        gals = gals[i1]
        gals.add_fields([('z', 'f4')])
        gals.z = spec[i0].z

        # Set the redshift range
        zrange = self.config.zrange.copy()
        zrange[0] = np.clip(zrange[0] - self.config.calib_zrange_cushion, 0.05, None)
        zrange[1] += self.config.calib_zrange_cushion

        use, = np.where((gals.z > zrange[0]) & (gals.z < zrange[1]) &
                        (gals.refmag < self.config.limmag_ref))
        gals = gals[use]

        galcolor = gals.galcol
        galcolor_err = gals.galcol_err

        ncol = self.config.nmag - 1

        # Make nodes
        nodes = make_nodes(zrange, self.config.calib_pivotmag_nodesize)

        # Temporary storage of output values
        medcol = np.zeros((nodes.size, ncol))
        medcol_width = np.zeros_like(medcol)
        meancol = np.zeros_like(medcol)
        meancol_scatter = np.zeros_like(meancol)

        # Read in the template
        template = Catalog.from_fits_file(self._template_file, ext=1)
        template_hdr = fitsio.read_header(self._template_file, ext=1)
        template_bands = list(template_hdr['BANDS'].rstrip())

        # Loop over modes
        nmodes = self.config.calib_colormem_colormodes.size

        for m in xrange(nmodes):
            j = self.config.calib_colormem_colormodes[m]

            print("Working on color %d" % (j))

            # Get the template index...
            try:
                template_index = template_bands.index(self.config.bands[j])
            except ValueError:
                raise ValueError("Calibration color %s-%s not in template file %s" % (self.config.bands[j], self.config.bands[j + 1], self.config.calib_redgal_template))

            # Compute global template offset
            c = interpol(template.color[:, template_index], template.z, gals.z)
            delta = galcolor[:, j] - c

            st = np.argsort(delta)
            delta5 = delta[st[int(0.05 * delta.size)]]
            delta99 = delta[st[int(0.99 * delta.size)]]

            u, = np.where((delta > delta5) & (delta < delta99))

            ecfitter = EcgmmFitter(delta[u], galcolor_err[u, j])
            wt, mu, sigma = ecfitter.fit([0.2], [-0.5, 0.0], [0.2, 0.05], offset=0.5)

            mvals = interpol(template.color[:, template_index], template.z, nodes) + mu[1]
            scvals = np.zeros(nodes.size) + sigma[1]

            # Fit median and median-width

            spl = CubicSpline(nodes, scvals)
            width = spl(gals.z)
            spl = CubicSpline(nodes, mvals)
            med = spl(gals.z)

            # Start by constraining away from the outliers...
            u, = np.where((galcolor[:, j] > (med - 2. * width)) &
                          (galcolor[:, j] < (med + 2. * width)))

            medfitter = MedZFitter(nodes, gals.z[u], galcolor[u, j])
            p0 = mvals
            mvals = medfitter.fit(p0)

            # for the median width we make a broader cut
            spl = CubicSpline(nodes, mvals)
            med = spl(gals.z)

            u, = np.where((galcolor[:, j] > (med - 2. * width)) &
                          (galcolor[:, j] < (med + 2. * width)))

            medfitter = MedZFitter(nodes, gals.z[u], np.abs(galcolor[u, j] - med[u]))
            p0 = scvals
            scvals = medfitter.fit(p0)

            # and record these
            medcol[:, j] = mvals
            medcol_width[:, j] = 1.4826 * scvals

            # Fit mean and scatter (with truncation)
            spl = CubicSpline(nodes, medcol[:, j])
            med = spl(gals.z)
            spl = CubicSpline(nodes, medcol_width[:, j])
            width = spl(gals.z)

            nsig = 1.5
            u, = np.where(np.abs(galcolor[:, j] - med) < nsig * width)

            trunc = nsig * width[u]
            rsfitter = RedSequenceFitter(nodes, gals.z[u], galcolor[u, j], galcolor_err[u, j],
                                         trunc=trunc)

            # fit just the mean...
            p0_mean = medcol[:, j]
            p0_slope = np.zeros_like(p0_mean)
            p0_scatter = medcol_width[:, j]
            mvals, = rsfitter.fit(p0_mean, p0_slope, p0_scatter, fit_mean=True)

            # And just the scatter...
            p0_mean = mvals
            scvals, = rsfitter.fit(p0_mean, p0_slope, p0_scatter, fit_scatter=True)

            # And both
            p0_scatter = scvals
            mvals, scvals = rsfitter.fit(p0_mean, p0_slope, p0_scatter, fit_mean=True, fit_scatter=True)

            meancol[:, j] = mvals
            meancol_scatter[:, j] = scvals

        # Select red galaxies, according to the redshift ranges of the modes

        zbounds = np.concatenate([np.array([zrange[0] - 0.011]),
                                  self.config.calib_colormem_zbounds,
                                  np.array([zrange[1] + 0.011])])

        mark = np.zeros(gals.size, dtype=np.bool)

        for m in xrange(nmodes):
            u, = np.where((gals.z > zbounds[m]) &
                          (gals.z < zbounds[m + 1]))
            j = self.config.calib_colormem_colormodes[m]

            if u.size > 0:
                spl = CubicSpline(nodes, meancol[:, j])
                mn = spl(gals.z[u])
                spl = CubicSpline(nodes, meancol_scatter[:, j])
                sc = spl(gals.z[u])

                gd, = np.where((np.abs(galcolor[u, j] - mn) <
                                self.config.calib_redspec_nsig * np.sqrt(galcolor_err[u, j]**2. + sc**2.)))
                mark[u[gd]] = True

        # Output the red galaxies
        use, = np.where(mark)

        fitsio.write(self.config.redgalfile, gals[use]._ndarray, clobber=True)

        # Output the model
        model = np.zeros(1, dtype=[('NODES', 'f4', nodes.size),
                                   ('MEANCOL', 'f4', meancol.shape),
                                   ('MEANCOL_SCATTER', 'f4', meancol_scatter.shape),
                                   ('MEDCOL', 'f4', medcol.shape),
                                   ('MEDCOL_WIDTH', 'f4', medcol_width.shape)])
        model['NODES'] = nodes
        model['MEANCOL'] = meancol
        model['MEANCOL_SCATTER'] = meancol_scatter
        model['MEDCOL'] = medcol
        model['MEDCOL_WIDTH'] = medcol_width

        fitsio.write(self.config.redgalmodelfile, model, clobber=True)

        # And make some plots!
        import matplotlib.pyplot as plt

        for m in xrange(nmodes):
            j = self.config.calib_colormem_colormodes[m]

            fig = plt.figure(1, figsize=(8, 6))
            fig.clf()
            ax = fig.add_subplot(111)

            ax.hexbin(gals.z[use], galcolor[use, j], bins='log')
            xvals = np.arange(zrange[0], zrange[1], 0.01)
            spl = CubicSpline(nodes, meancol[:, j])
            ax.plot(xvals, spl(xvals), 'r-')
            ax.plot(nodes, meancol[:, j], 'ro')
            ax.set_xlabel('z_spec')
            ax.set_ylabel(self.config.bands[j] + ' - ' + self.config.bands[j + 1])
            ax.set_title('Red Training Galaxies')
            ax.set_xlim(self.config.zrange)

            fig.savefig(os.path.join(self.config.outpath, self.config.plotpath,
                                     '%s_redgals_%s-%s.png' % (self.config.outbase,
                                                               self.config.bands[j],
                                                               self.config.bands[j + 1])))
            plt.close(fig)
