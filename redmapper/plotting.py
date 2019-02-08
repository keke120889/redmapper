"""Classes for making pretty diagnostic plots for redmapper catalogs
"""

from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import os
import numpy as np
import fitsio
import esutil
import scipy.optimize
import scipy.ndimage
import copy

from .configuration import Configuration
from .utilities import gaussFunction, CubicSpline, interpol

class SpecPlot(object):
    """
    Class to make plots comparing the cluster spec-z with cluster z_lambda (photo-z).

    In the case of real data, the most likely central is used as a proxy for
    the cluster redshift (so miscenters look like wrong redshifts).  In the
    case of mock data, the mean redshift of the members can be used instead.
    """

    def __init__(self, conf, binsize=0.02, nsig=4.0):
        """
        Instantiate a SpecPlot.

        Parameters
        ----------
        conf: `redmapper.Configuration` or `str`
           Configuration object or filename
        binsize: `float`, optional
           Redshift smoothing bin size.  Default is 0.02.
        nsig: `float`, optional
           Number of sigma to be considered a redshift outlier.  Default is 4.0.
        """
        if not isinstance(conf, Configuration):
            self.config = Configuration(conf)
        else:
            self.config = conf

        self.binsize = binsize
        self.nsig = nsig

    def plot_cluster_catalog(self, cat, title=None, figure_return=False):
        """
        Plot the spectroscopic comparison for a cluster catalog, using the
        default catalog values.

        This plot will compare the catalog cg_spec_z (most likely central
        galaxy redshift) with the cluster z_lambda.

        Parameters
        ----------
        cat: `redmapper.ClusterCatalog`
           Cluster catalog to plot.
        title: `str`, optional
           Title string for plot.  Default is None.
        figure_return: `bool`, optional
           Return the figure instead of saving a png.  Default is False.

        Returns
        -------
        fig: `matplotlib.Figure` or `None`
           Figure to show, if figure_return is True.
        """

        return self.plot_values(cat.cg_spec_z, cat.z_lambda, cat.z_lambda_e, title=title,
                                figure_return=figure_return)

    def plot_cluster_catalog_from_members(self, cat, mem, title=None, figure_return=False):
        """
        Plot the spectroscopic comparison for a cluster catalog, using the
        average member redshift.

        This plot will compare the catalog median redshift with the cluster
        z_lambda.  Generally only useful for simulated catalogs where you have
        complete coverage.

        Parameters
        ----------
        cat: `redmapper.ClusterCatalog`
           Cluster catalog to plot.
        mem: `redmapper.Catalog`
           Member catalog associated with cat.
        title: `str`, optional
           Title string for plot.  Default is None.
        figure_return: `bool`, optional
           Return the figure instead of saving a png.  Default is False.

        Returns
        -------
        fig: `matplotlib.Figure` or `None`
           Figure to show, if figure_return is True.
        """

        # Not sure why this is necessary on the mocks, but there are some -1s...
        ok, = np.where(mem.zspec > 0.0)
        mem = mem[ok]

        a, b = esutil.numpy_util.match(cat.mem_match_id, mem.mem_match_id)

        h, rev = esutil.stat.histogram(a, rev=True)
        mem_zmed = np.zeros(cat.size)
        for i in xrange(h.size):
            if h[i] == 0:
                continue
            i1a = rev[rev[i]: rev[i + 1]]
            mem_zmed[i] = np.median(mem.zspec[i1a])

        return self.plot_values(mem_zmed, cat.z_lambda, cat.z_lambda_e, title=title,
                                figure_return=figure_return)

    def plot_values(self, z_spec, z_phot, z_phot_e, name='z_\lambda', specname='z_{\mathrm{spec}}',
                    title=None, figure_return=False):
        """
        Make a pretty spectrscopic plot from an arbitrary list of values.

        Parameters
        ----------
        z_spec: `np.array`
           Float array of spectroscopic redshifts
        z_phot: `np.array`
           Float array of photometric redshifts
        z_phot_e: `np.array`
           Float array of photometric redshift errors
        name: `str`, optional
           Name of photo-z field for label.  Default is 'z_\lambda'.
        title: `str`, optional
           Title string.  Default is None
        figure_return: `bool`, optional
           Return the figure instead of saving a png.  Default is False.

        Returns
        -------
        fig: `matplotlib.Figure` or `None`
           Figure to show, if figure_return is True.
        """

        # We will need to remove any underscores in name for some usage...
        name_clean = name.replace('_', '')
        name_clean = name_clean.replace('^', '')

        use, = np.where(z_spec > 0.0)

        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator

        plot_xrange = np.array([0.0, self.config.zrange[1] + 0.1])

        fig = plt.figure(figsize=(8, 6))
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
        ax.set_ylabel(r'$%s$' % (specname), fontsize=16)
 
        # Plot the outliers
        bad, = np.where(np.abs(z_phot[use] - z_spec[use]) / z_phot_e[use] > self.nsig)
        if bad.size > 0:
            ax.plot(z_phot[use[bad]], z_spec[use[bad]], 'r*')

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
        ax2.set_ylabel(r'$%s - %s$' % (specname, name), fontsize=16)
        minorLocator = MultipleLocator(0.02)
        ax2.xaxis.set_minor_locator(minorLocator)
        minorLocator = MultipleLocator(0.002)
        ax2.yaxis.set_minor_locator(minorLocator)

        fig.subplots_adjust(hspace=0.0)

        fig.tight_layout()

        if not figure_return:
            fig.savefig(self.filename)
            plt.close(fig)
        else:
            return fig

    @property
    def filename(self):
        """
        Get the filename that should be used to save the figure.

        Returns
        -------
        filename: `str`
           Formatted filename to save figure.
        """
        return self.config.redmapper_filename('zspec', paths=(self.config.plotpath,), filetype='png')

    def _make_photoz_map(self, z_spec, z_photo,
                         dzmax=0.1, nbins_coarse=20, nbins_fine=200, zrange=[0.0, 1.2]):
        """
        Internal method to make a photo-z map for contour plotting.

        Parameters
        ----------
        z_spec: `np.array`
           Float array of z_spec values.
        z_photo: `np.array`
           Float array of photometric redshift values.
        dzmax: `float`, optional
           Maximum delta-z when fitting in bin.  Default is 0.1
        nbins_coarse: `int`, optional
           Number of bins for initial coarse binning.  Default is 20
        nbins_fine: `int`, optional
           Number of bins for smooth fine binning for plot.  Default is 200
        zrange: `list`, optional
           Redshift range to build map.  Default is [0.0, 1.2]

        Returns
        -------
        z_bins: `np.array`
           Float array of redshift bins for map
        z_map_smooth: `np.array`
           Float array of image of smoothed map.
        """

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
                with np.warnings.catch_warnings():
                    np.warnings.simplefilter('error')
                    coeff, varMatrix = scipy.optimize.curve_fit(gaussFunction, dz, spl[i, :], p0=p0)
            except (RuntimeError, RuntimeWarning, scipy.optimize.OptimizeWarning):
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
                ss = CubicSpline(dz, spl[index, :])
                y = ss(dz1)
                bad, = np.where((np.abs(dz1) > 0.1) | (y < 0.0))
                y[bad] = 0.0
                if y.max() == 0:
                    z_map_values[i, :] = 0.0
                else:
                    z_map_values[i, :] = y / y.max()

        bad = np.where(z_map_values < 1e-3)
        z_map_values[bad] = 0.0

        z_map_smooth = scipy.ndimage.uniform_filter(z_map_values, size=5, mode='nearest')

        return z_bins, z_map_smooth

class NzPlot(object):
    """
    Class to make a plot with cluster catalog n(z) with comoving coordinates.
    """

    def __init__(self, conf, binsize=0.02):
        """
        Instantiate a NzPlot.

        Parameters
        ----------
        conf: `redmapper.Configuration` or `str`
           Configuration object or filename
        binsize: `float`, optional
           Redshift smoothing bin size.  Default is 0.02.
        """
        if not isinstance(conf, Configuration):
            self.config = Configuration(conf)
        else:
            self.config = conf

        self.binsize = binsize
        self._redmapper_name = 'nz'

    def plot_cluster_catalog(self, cat, areastr, nosamp=False):
        """
        Plot the n(z) for a cluster catalog, using the default catalog values.

        This plot will sample redshifts from z_lambda +/- z_lambda_e.

        Parameters
        ----------
        cat: `redmapper.ClusterCatalog`
           Cluster catalog to plot.
        areastr: `redmapper.Catalog`
           Area structure, with .z and .area
        nosamp: `bool`, optional
           Do not sample from z_lambda.  Default is False.
        """

        if nosamp:
            zsamp = cat.z_lambda
        else:
            zsamp = np.zeros(cat.size)

            nsampbin = 100
            npzbins = self.config.npzbins

            for i, cluster in enumerate(cat):
                xvals = ((np.arange(nsampbin, dtype=np.float64) / nsampbin) *
                         (cluster.pzbins[-1] - cluster.pzbins[0]) + cluster.pzbins[0])
                yvals = interpol(cluster.pz, cluster.pzbins, xvals)
                pdf = yvals / np.sum(yvals)
                cdf = np.cumsum(pdf)
                cdfi = (cdf * xvals.size).astype(np.int32)
                rand = (np.random.uniform(size=1)*nsampbin).astype(np.int32)
                test, = np.where(cdfi >= rand)
                zsamp[i] = xvals[test[0]]

        self.plot_nz(zsamp, areastr, self.config.zrange,
                     xlabel=r'$z_{\lambda}$',
                     ylabel=r'$n\,(1e4\,\mathrm{clusters} / \mathrm{Mpc}^{3})$',
                     redmapper_name='nz')

    def plot_nz(self, z, areastr, zrange, xlabel=None, ylabel=None,
                title=None, redmapper_name='nz'):
        """
        """

        import matplotlib.pyplot as plt

        corr_zrange = copy.copy(zrange)
        nbin = int(np.ceil((corr_zrange[1] - corr_zrange[0]) / self.binsize))
        corr_zrange[1] = nbin * self.binsize + corr_zrange[0]

        hist = esutil.stat.histogram(z, min=corr_zrange[0], max=corr_zrange[1]-0.0001, binsize=self.binsize, more=True)
        h = hist['hist']
        zbins = hist['center']

        indices = np.clip(np.searchsorted(areastr.z, zbins), 0, areastr.size - 1)

        vol = np.zeros(zbins.size)
        for i in xrange(zbins.size):
            vol[i] = (self.config.cosmo.V(zbins[i] - self.binsize/2.,
                                          zbins[i] + self.binsize/2.) *
                      (areastr.area[indices[i]] / 41252.961))

        dens = h.astype(np.float32) / vol
        err = np.sqrt(dens * vol) / vol

        fig = plt.figure(1, figsize=(8, 6))
        fig.clf()

        ax = fig.add_subplot(111)

        ax.errorbar(zbins, dens*1e4, yerr=err*1e4, fmt='r.', markersize=8)
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=16)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=16)
        if title is not None:
            ax.set_title(title, fontsize=16)
        ax.tick_params(axis='both',which='major',labelsize=14)
        fig.tight_layout()

        self._redmapper_name = redmapper_name
        fig.savefig(self.filename)
        plt.close(fig)

    def plot_redmagic_catalog(self, cat, name, eta, n0, areastr, sample=True, zrange=None):
        """
        """

        if sample:
            zsamp = np.random.normal(size=cat.size) * cat.zredmagic_e + cat.zredmagic
        else:
            zsamp = cat.zredmagic

        if zrange is None:
            zrange = self.config.redmagic_zrange

        redmapper_name = 'redmagic_%s_%3.1f-%02d_nz' % (name, eta, int(n0))
        self.plot_nz(zsamp, areastr, zrange,
                     xlabel=r'$z_{\mathrm{redmagic}}$',
                     ylabel=r'$n\,(1e4\,\mathrm{galaxies} / \mathrm{Mpc}^{3})$',
                     title='%s: %3.1f-%02d' % (name, eta, int(n0)),
                     redmapper_name=redmapper_name)

    @property
    def filename(self):
        """
        Get the filename that should be used to save the figure.

        Returns
        -------
        filename: `str`
           Formatted filename to save figure.
        """
        return self.config.redmapper_filename(self._redmapper_name, paths=(self.config.plotpath,), filetype='png')
