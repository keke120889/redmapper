from __future__ import division, absolute_import, print_function

import os
import numpy as np
import fitsio
import esutil
import scipy.optimize
import scipy.ndimage

from .configuration import Configuration
from .utilities import gaussFunction, CubicSpline

class SpecPlot(object):
    """
    """

    def __init__(self, conf, binsize=0.02, nsig=4.0):
        if not isinstance(conf, Configuration):
            self.config = Configuration(conf)
        else:
            self.config = conf

        self.binsize = binsize
        self.nsig = nsig

    def plot_cluster_catalog(self, cat):
        """
        """

        self.run_values(cat.bcg_spec_z, cat.z_lambda, cat.z_lambda_e)

    def plot_values(self, z_spec, z_phot, z_phot_e, name='z_\lambda', title=None):
        """
        """

        # We will need to remove any underscores in name for some usage...
        name_clean = name.replace('_', '')
        name_clean = name_clean.replace('^', '')

        use, = np.where(z_spec > 0.0)

        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator

        plot_xrange = np.array([0.0, self.config.zrange[1] + 0.1])

        fig = plt.figure(1, figsize=(8, 6))
        fig.clf()

        ax = fig.add_subplot(211)

        # Need to do the map, then need to get the levels and make a contour
        z_bins, z_map = self._make_photoz_map(z_spec[use], z_phot[use])

        nlevs = 5
        levbinsize = (1.0 - 0.1) / nlevs
        levs = np.arange(nlevs) * levbinsize + 0.1
        levs = np.append(levs, 10)
        levs = -levs[::-1]

        colors = ['#000000', '#333333', '#666666', '#9A9A9A', '#CECECE', '#FFFFFF']

        ax.contourf(z_bins, z_bins, -z_map.T, levs, colors=colors)
        ax.plot(plot_xrange, plot_xrange, 'b--', linewidth=3)
        ax.set_xlim(plot_xrange)
        ax.set_ylim(plot_xrange)
        ax.tick_params(axis='y', which='major', labelsize=14, length=5, left=True, right=True, direction='in')
        ax.tick_params(axis='y', which='minor', left=True, right=True, direction='in')
        ax.tick_params(axis='x', labelbottom='off', which='major', length=5, direction='in', bottom=True, top=True)
        ax.tick_params(axis='x', which='minor', bottom=True, top=True, direction='in')
        minorLocator = MultipleLocator(0.05)
        ax.yaxis.set_minor_locator(minorLocator)
        minorLocator = MultipleLocator(0.02)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.set_ylabel(r'$z_\mathrm{spec}$', fontsize=16)

        # Plot the outliers
        bad, = np.where(np.abs(z_phot[use] - z_spec[use]) / z_phot_e[use] > self.nsig)
        if bad.size > 0:
            ax.plot(z_phot[use[bad]], z_spec[use[bad]], 'r.')

        fracout = float(bad.size) / float(use.size)

        fout_label = r'$f_\mathrm{out} = %7.4f$' % (fracout)
        ax.annotate(fout_label, (plot_xrange[0] + 0.1, self.config.zrange[1]),
                    xycoords='data', ha='left', va='top', fontsize=16)

        ax.set_title(title)

        # Compute the bias / scatter
        h, rev = esutil.stat.histogram(z_phot[use], min=0.0, max=self.config.zrange[1]-0.001, rev=True, binsize=self.binsize)
        bins = np.arange(h.size) * self.binsize + 0.0 + self.binsize/2.

        bias = np.zeros(h.size)
        scatter = np.zeros_like(bias)
        errs = np.zeros_like(bias)

        gd, = np.where(h >= 3)
        for ind in gd:
            i1a = rev[rev[ind]: rev[ind + 1]]
            bias[ind] = np.median(z_spec[use[i1a]] - z_phot[use[i1a]])
            scatter[ind] = 1.4862 * np.median(np.abs((z_spec[use[i1a]] - z_phot[use[i1a]]) - bias[ind]))
            errs[ind] = np.median(z_phot_e[use[i1a]])

        # Make the bottom plot
        ax2 = fig.add_subplot(212, sharex=ax)

        ax2.plot(plot_xrange, [0.0, 0.0], 'b--', linewidth=2)
        ax2.plot(bins, bias, 'm-.', label=r'$\mathrm{Bias}$', linewidth=3)
        ax2.plot(bins, scatter/(1. + bins), 'r:', label=r'$\sigma_z / (1 + z)$', linewidth=3)
        ax2.plot(bins, errs/(1. + bins), 'c--', label=r'$\sigma_{%s} / (1 + z)$' % (name_clean), linewidth=3)

        ax2.legend(loc=4, fontsize=14)

        ax2.set_xlim(plot_xrange)
        ax2.set_ylim([-0.03, 0.024])
        ax2.tick_params(axis='both', which='major', labelsize=14, length=5, left=True, right=True, top=True, bottom=True, direction='in')
        ax2.tick_params(axis='y', which='minor', left=True, right=True, direction='in')
        ax2.tick_params(axis='x', which='minor', bottom=True, top=True, direction='in')
        ax2.set_xlabel(r'$%s$' % (name), fontsize=16)
        ax2.set_ylabel(r'$z_\mathrm{spec} - %s$' % (name), fontsize=16)
        minorLocator = MultipleLocator(0.02)
        ax2.xaxis.set_minor_locator(minorLocator)
        minorLocator = MultipleLocator(0.002)
        ax2.yaxis.set_minor_locator(minorLocator)

        fig.subplots_adjust(hspace=0.0)

        fig.savefig(self.filename)
        plt.close(fig)

    @property
    def filename(self):
        return self.config.redmapper_filename('zspec', paths=(self.config.plotpath,), filetype='png')

    def _make_photoz_map(self, z_spec, z_photo,
                         dzmax=0.1, nbins_coarse=20, nbins_fine=200, zrange=[0.0, 1.2]):

        zmin = z_photo.min()
        zmax = z_photo.max()

        zbinsize = (zmax - zmin) / nbins_coarse
        zbins = np.arange(nbins_coarse) * zbinsize + zmin
        dzbinsize = (2. * dzmax)/(nbins_coarse)
        dzbins = np.arange(nbins_coarse) * dzbinsize - dzmax

        z = zbins + 0.5 * zbinsize
        dz = dzbins + 0.5 * dzbinsize

        spl = np.zeros((nbins_coarse, nbins_coarse))

        for i in range(nbins_coarse):
            use, = np.where((z_photo >= zbins[i]) & (z_photo < (zbins[i] + zbinsize)))

            if use.size == 0:
                continue

            dzvec = z_spec[use] - z_photo[use]

            for j in range(nbins_coarse):
                use2, = np.where((dzvec >= dzbins[j]) & (dzvec < (dzbins[j] + dzbinsize)))
                spl[i, j] = use2.size

            p0 = [use.size, np.median(dzvec), np.std(dzvec)]
            try:
                coeff, varMatrix = scipy.optimize.curve_fit(gaussFunction, dz, spl[i, :], p0=p0)
            except RuntimeError:
                # set to the default?
                coeff = p0
            spl[i, :] = spl[i, :] / coeff[0]

        maxdz = 0.5 * (z[1] - z[0])

        zbinsize = (zrange[1] - zrange[0]) / nbins_fine
        z_bins = np.arange(nbins_fine) * zbinsize + zrange[0]

        z_map_values = np.zeros((nbins_fine, nbins_fine))

        for i in range(nbins_fine):
            delta = np.abs(z_bins[i] - z)
            index = np.argmin(delta)
            val = delta[index]

            if (val < maxdz):
                dz1 = z_bins - z_bins[i]
                s = CubicSpline(dz, spl[index, :])
                y = s(dz1)
                bad, = np.where((np.abs(dz1) > 0.1) | (y < 0.0))
                y[bad] = 0.0

                z_map_values[i, :] = y / y.max()

        bad = np.where(z_map_values < 1e-3)
        z_map_values[bad] = 0.0

        z_map_smooth = scipy.ndimage.uniform_filter(z_map_values, size=5, mode='nearest')

        return z_bins, z_map_smooth
