from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import yaml
import fitsio
import copy
from esutil.cosmology import Cosmo
import numpy as np
import re
import os

from .cluster import cluster_dtype_base, member_dtype_base
from ._version import __version__

class ConfigField(object):
    """
    A validatable field with a default
    """

    def __init__(self, value=None, default=None, isArray=False, required=False, array_length=None):
        self._value = value
        self._isArray = isArray
        self._required = required
        self._array_length = array_length

        self._default = default
        if isArray:
            if default is not None:
                self._default = np.atleast_1d(default)
            if self._value is not None:
                self._value = np.atleast_1d(self._value)

        if self._value is None:
            self._value = self._default

    def __get__(self, obj, type=None):
        return self._value

    def __set__(self, obj, value):
        if self._isArray:
            self._value = np.atleast_1d(value)
        else:
            try:
                if len(value) > 1:
                    raise ValueError("ConfigField cannot be length > 1")
            except:
                pass
            self._value = value

    def reset(self):
        self._value = self._default

    def set_length(self, length):
        self._array_length = length

    def validate(self, name):
        if self._required:
            if self._value is None:
                raise ValueError("Required ConfigField %s is not set" % (name))

        if self._value is not None and self._isArray:
            if self._array_length is not None:
                if self._value.size != self._array_length:
                    raise ValueError("ConfigField %s has the wrong length (%d != %d)" %
                                     (name, self._value.size, self._array_length))

        return True

def read_yaml(filename):

    """
    Name:
        read_yaml
    Purpose:
        Read in a YAML file with key/value pairs and put into dict.
    Calling Sequence:
        outdict = read_yaml(filename, defaults=None)
    Inputs:
        filename: YAML file name
    Outputs:
        outdict: a dictionary of the key/value pairs in the YAML file.
    """

    # The output dictionary
    outdict = {}

    # Open the yaml file and find key/value pairs
    with open(filename) as f: yaml_data = yaml.load(f)
    for tag in outdict:
        if outdict[tag] is None:
            raise ValueError('A value for the required tag \"' 
                                + tag + '\" must be specified.')

    # Add the pairs to the dictionary
    for tag in yaml_data: outdict[tag] = yaml_data[tag]

    return outdict

class Configuration(object):
    version = ConfigField(default=__version__, required=True)

    galfile = ConfigField(required=True)
    zredfile = ConfigField()
    halofile = ConfigField()
    randfile = ConfigField()
    catfile = ConfigField()
    specfile = ConfigField()
    specfile_train = ConfigField()
    outbase = ConfigField(required=True)
    parfile = ConfigField()
    bkgfile = ConfigField()
    bkgfile_color = ConfigField()
    zlambdafile = ConfigField()
    maskfile = ConfigField()
    depthfile = ConfigField()
    wcenfile = ConfigField()
    redgalfile = ConfigField()

    calib_nproc = ConfigField(default=1, required=True)

    outpath = ConfigField(default='./', required=True)
    plotpath = ConfigField(default='', required=True)

    border = ConfigField(default=0.0, required=True)
    hpix = ConfigField(default=0, required=True)
    nside = ConfigField(default=0, required=True)
    galfile_pixelized = ConfigField(required=True)

    nmag = ConfigField(required=True)
    area = ConfigField(required=True)
    limmag_catalog = ConfigField(required=True)
    limmag_ref = ConfigField(required=True)
    refmag = ConfigField(required=True)
    ref_ind = ConfigField(required=True)
    zeropoint = ConfigField(required=True)
    survey_mode = ConfigField(required=True)
    b = ConfigField(isArray=True)
    galfile_nside = ConfigField(required=True)
    bands = ConfigField(required=True)

    zrange = ConfigField(isArray=True, array_length=2, required=True)
    lval_reference = ConfigField(default=0.2, required=True)

    maskgalfile = ConfigField(default='maskgals.fit', required=True)
    mask_mode = ConfigField(default=0, required=True)
    max_maskfrac = ConfigField(default=0.2, required=True)

    dldr_gamma = ConfigField(default=0.6, required=True)
    rsig = ConfigField(default=0.05, required=True)
    chisq_max = ConfigField(default=20.0, required=True)
    npzbins = ConfigField(default=21, required=True)

    mstar_survey = ConfigField(default='sdss')
    mstar_band = ConfigField(default='i03')

    calib_niter = ConfigField(default=3)
    calib_zrange_cushion = ConfigField(default=0.05)

    calib_use_pcol = ConfigField(default=True)
    calib_redgal_template = ConfigField()
    calib_pivotmag_nodesize = ConfigField(default=0.1)
    calib_color_nodesizes = ConfigField(isArray=True, default=np.array([0.05]))
    calib_slope_nodesizes = ConfigField(isArray=True, default=np.array([0.1]))
    calib_color_maxnodes = ConfigField(isArray=True, default=np.array([-1.0]))
    calib_covmat_maxnodes = ConfigField(isArray=True, default=np.array([-1.0]))
    calib_covmat_nodesize = ConfigField(default=0.15)
    calib_covmat_min_eigenvalue = ConfigField(default=0.0001)
    calib_covmat_prior = ConfigField(default=0.45)
    calib_corr_nodesize = ConfigField(default=0.05)
    calib_corr_slope_nodesize = ConfigField(default=0.1)
    calib_corr_nocorrslope = ConfigField(default=True)
    calib_corr_pcut = ConfigField(default=0.9)
    calib_color_order = ConfigField(isArray=True, default=np.array([-1]))

    calib_color_nsig = ConfigField(default=1.5)
    calib_redspec_nsig = ConfigField(default=2.0)

    calib_colormem_r0 = ConfigField(default=0.5)
    calib_colormem_beta = ConfigField(default=0.0)
    calib_colormem_smooth = ConfigField(default=0.003)
    calib_colormem_minlambda = ConfigField(default=10.0)
    calib_colormem_zbounds = ConfigField(isArray=True, default=np.array([0.4]))
    calib_colormem_colormodes = ConfigField(isArray=True, default=np.array([1]))
    calib_colormem_sigint = ConfigField(isArray=True, default=np.array([0.05]))
    calib_pcut = ConfigField(default=0.3)
    calib_color_pcut = ConfigField(default=0.7)

    calib_lumfunc_alpha = ConfigField(default=-1.0, required=True)

    zredc_binsize_fine = ConfigField(default=0.0001)
    zredc_binsize_coarse = ConfigField(default=0.005)

    bkg_chisqbinsize = ConfigField(default=0.5)
    bkg_refmagbinsize = ConfigField(default=0.2)
    bkg_zbinsize = ConfigField(default=0.02)
    bkg_deepmode = ConfigField(default=False)

    zlambda_pivot = ConfigField(default=30.0, required=True)
    zlambda_binsize = ConfigField(default=0.002, required=True)
    zlambda_maxiter = ConfigField(default=20, required=True)
    zlambda_tol = ConfigField(default=0.0002, required=True)
    zlambda_topfrac = ConfigField(default=0.7, required=True)
    zlambda_epsilon = ConfigField(default=0.005, required=True)
    zlambda_parab_step = ConfigField(default=0.002, required=True)

    centerclass = ConfigField(default='CenteringBCG', required=True)
    wcen_rsoft = ConfigField(default=0.05, required=True)
    wcen_zred_chisq_max = ConfigField(default=100.0, required=True)
    wcen_minlambda = ConfigField(default=10.0, required=True)
    wcen_maxlambda = ConfigField(default=100.0, required=True)
    wcen_cal_zrange = ConfigField(isArray=True, default=np.array([0.0,1.0]))
    wcen_pivot = ConfigField(default=30.0, required=True)
    wcen_uselum = ConfigField(default=True, required=True)
    wcen_Delta0 = ConfigField(required=False)
    wcen_Delta1 = ConfigField(required=False)
    wcen_sigma_m = ConfigField(required=False)
    lnw_cen_sigma = ConfigField(required=False)
    lnw_cen_mean = ConfigField(required=False)
    lnw_sat_sigma = ConfigField(required=False)
    lnw_sat_mean = ConfigField(required=False)
    lnw_fg_sigma = ConfigField(required=False)
    lnw_fg_mean = ConfigField(required=False)

    firstpass_r0 = ConfigField(default=0.5, required=True)
    firstpass_beta = ConfigField(default=0.0, required=True)
    firstpass_niter = ConfigField(default=2, required=True)
    firstpass_minlambda = ConfigField(default=3.0, required=True)

    likelihoods_r0 = ConfigField(default=1.0, required=True)
    likelihoods_beta = ConfigField(default=0.2, required=True)
    likelihoods_use_zred = ConfigField(default=True, required=True)
    likelihoods_minlambda = ConfigField(default=3.0, required=True)

    percolation_r0 = ConfigField(default=1.0, required=True)
    percolation_beta = ConfigField(default=0.2, required=True)
    percolation_rmask_0 = ConfigField(default=1.5, required=True)
    percolation_rmask_beta = ConfigField(default=0.2, required=True)
    percolation_rmask_gamma = ConfigField(default=0.0, required=True)
    percolation_rmask_zpivot = ConfigField(default=0.3, required=True)
    percolation_lmask = ConfigField()
    percolation_niter = ConfigField(default=2, required=True)
    percolation_minlambda = ConfigField(default=3.0, required=True)
    percolation_pbcg_cut = ConfigField(default=0.5, required=True)
    percolation_maxcen = ConfigField(default=5, required=True)
    percolation_memradius = ConfigField()
    percolation_memlum = ConfigField()

    def __init__(self, configfile):

        self._reset_vars()

        # First, read in the yaml file
        confdict = read_yaml(configfile)

        # And now set the config variables
        #for key in confdict:
        #    setattr(self, key, confdict[key])
        self._set_vars_from_dict(confdict)

        # validate the galfile and refmag
        type(self).__dict__['galfile'].validate('galfile')
        type(self).__dict__['refmag'].validate('refmag')

        # get galaxy file stats
        gal_stats = self._galfile_stats()

        self._set_vars_from_dict(gal_stats, check_none=True)

        # Record the cluster dtype for convenience
        self.cluster_dtype = copy.copy(cluster_dtype_base)
        self.cluster_dtype.extend([('MAG', 'f4', self.nmag),
                                   ('MAG_ERR', 'f4', self.nmag),
                                   ('PZBINS', 'f4', self.npzbins),
                                   ('PZ', 'f4', self.npzbins),
                                   ('RA_CENT', 'f8', self.percolation_maxcen),
                                   ('DEC_CENT', 'f8', self.percolation_maxcen),
                                   ('ID_CENT', 'i4', self.percolation_maxcen),
                                   ('LAMBDA_CENT', 'f4', self.percolation_maxcen),
                                   ('ZLAMBDA_CENT', 'f4', self.percolation_maxcen),
                                   ('P_CEN', 'f4', self.percolation_maxcen),
                                   ('Q_CEN', 'f4', self.percolation_maxcen),
                                   ('P_FG', 'f4', self.percolation_maxcen),
                                   ('Q_MISS', 'f4'),
                                   ('P_SAT', 'f4', self.percolation_maxcen),
                                   ('P_C', 'f4', self.percolation_maxcen)])
        self.member_dtype = copy.copy(member_dtype_base)
        self.member_dtype.extend([('MAG', 'f4', self.nmag),
                                  ('MAG_ERR', 'f4', self.nmag)])
        # also need pz stuff, etc, etc.  Will need to deal with defaults

        # Calibration size checking
        self._set_lengths(['calib_colormem_zbounds', 'calib_colormem_colormodes'],
                          len(self.calib_colormem_zbounds))
        self._set_lengths(['calib_color_nodesizes', 'calib_slope_nodesizes',
                           'calib_color_maxnodes', 'calib_covmat_maxnodes',
                           'calib_color_order'], self.nmag - 1)

        # will want to set defaults here...
        self.cosmo = Cosmo()

        # Finally, we can validate...
        self.validate()

    def validate(self):
        """
        """

        for var in type(self).__dict__:
            try:
                type(self).__dict__[var].validate(var)
            except AttributeError:
                pass

    def _reset_vars(self):
        for var in type(self).__dict__:
            try:
                type(self).__dict__[var].reset()
            except AttributeError:
                pass

    def _set_vars_from_dict(self, d, check_none=False):
        for key in d:
            if check_none and d[key] is None:
                continue
            if key not in type(self).__dict__:
                raise AttributeError("Unknown config variable: %s" % (key))
            setattr(self, key, d[key])

    def _set_lengths(self, l, length):
        for arr in l:
            if arr not in type(self).__dict__:
                raise AttributeError("Unknown config variable: %s" % (arr))
            type(self).__dict__[arr].set_length(length)


    def _galfile_stats(self):
        """
        """

        hdr = fitsio.read_header(self.galfile, ext=1)
        pixelated = hdr.get("PIXELS", 0)
        fitsformat = hdr.get("FITS", 0)

        if not fitsformat:
            raise ValueError("Input galfile must describe fits files.")

        gal_stats = {}
        if not pixelated:
            # statistics are from the header
            gal_stats['galfile_pixelized'] = False

            hdrmode = hdr.get("MODE", "").rstrip()

            if hdrmode == 'SDSS':
                gal_stats['survey_mode'] = 0
            elif hdrmode == 'DES':
                gal_stats['survey_mode'] = 1
            else:
                raise ValueError("Input galaxy file with unknown mode: %s" % (hdrmode))

            ## FIXME: check that these are all in the header
            gal_stats['area'] = hdr.get('AREA', -100.0)
            gal_stats['limmag_ref'] = hdr.get('LIM_REF')
            gal_stats['nmag'] = hdr.get('NMAG')
            gal_stats['zeropoint'] = hdr.get('ZP')
            gal_stats['ref_ind'] = hdr.get(self.refmag.upper()+'_IND')
            gal_stats['b'] = None
            gal_stats['galfile_nside'] = 0
            gal_stats['bands'] = [None]*gal_stats['nmag']

            # figure out the bands...
            for name in hdr:
                m = re.search('(.*)_IND', name)
                if m is None:
                    continue
                if m.groups()[0] == 'REF':
                    continue
                band = m.groups()[0].lower()
                gal_stats['bands'][hdr[name]] = band

        else:
            # statistics are from the master table file
            gal_stats['galfile_pixelized'] = True

            master=fitsio.read(self.galfile, ext=1, upper=True)

            mode = master['MODE'][0].rstrip().decode()
            if (mode == 'SDSS'):
                gal_stats['survey_mode'] = 0
            elif (mode == 'DES'):
                gal_stats['survey_mode'] = 1
            else:
                raise ValueError("Input galaxy file with unknown mode: %s" % (mode))

            gal_stats['area'] = master['AREA'][0]
            gal_stats['limmag_ref'] = master['LIM_REF'][0]
            gal_stats['nmag'] = master['NMAG'][0]
            if ('B' in master.dtype.names):
                gal_stats['b'] = master['B'][0]
            else:
                gal_stats['b'] = None
            gal_stats['zeropoint'] = master['ZEROPOINT'][0]
            gal_stats['ref_ind'] = master[self.refmag.upper()+'_IND'][0]
            gal_stats['galfile_nside'] = master['NSIDE'][0]
            gal_stats['bands'] = [None]*gal_stats['nmag']

            # Figure out the bands...
            for name in master.dtype.names:
                m = re.search('(.*)_IND', name)
                if m is None:
                    continue
                if m.groups()[0] == 'REF':
                    continue
                band = m.groups()[0].lower()
                gal_stats['bands'][master[name][0]] = band

        if any(x is None for x in gal_stats['bands']):
            # Remove this, and use config values
            gal_stats.pop('bands', None)

        return gal_stats


