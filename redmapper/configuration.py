import yaml
import fitsio
import copy
from esutil.cosmology import Cosmo
import numpy as np

from cluster import cluster_dtype_base, member_dtype_base
from _version import __version__

class ConfigField(object):
    """
    A validatable field with a default
    """

    def __init__(self, value=None, default=None, isArray=False, required=False, array_length=None):
        self._value = value
        self._isArray = isArray
        self._required = required
        self._array_length = array_length

        _default = default
        if isArray:
            if default is not None:
                _default = np.atleast_1d(default)
            if self._value is not None:
                self._value = np.atleast_1d(self._value)

        if self._value is None:
            self._value = _default

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
    outbase = ConfigField(required=True)
    parfile = ConfigField()
    bkgfile = ConfigField()
    zlambdafile = ConfigField()
    maskfile = ConfigField()
    depthfile = ConfigField()

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

    zrange = ConfigField(isArray=True, array_length=2, required=True)
    lval_reference = ConfigField(default=0.2, required=True)

    maskgalfile = ConfigField(default='maskgals.fit', required=True)
    mask_mode = ConfigField(default=0, required=True)
    max_maskfrac = ConfigField(default=0.2, required=True)

    dldr_gamma = ConfigField(default=0.6, required=True)
    rsig = ConfigField(default=0.05, required=True)
    chisq_max = ConfigField(default=20.0, required=True)
    npzbins = ConfigField(default=21, required=True)

    zlambda_pivot = ConfigField(default=30.0, required=True)
    zlambda_binsize = ConfigField(default=0.002, required=True)
    zlambda_maxiter = ConfigField(default=20, required=True)
    zlambda_tol = ConfigField(default=0.0002, required=True)
    zlambda_topfrac = ConfigField(default=0.7, required=True)
    zlambda_epsilon = ConfigField(default=0.005, required=True)
    zlambda_parab_step = ConfigField(default=0.002, required=True)

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

        #for key in gal_stats:
        #    if gal_stats[key] is not None:
        #        setattr(self, key, gal_stats[key])
        self._set_vars_from_dict(gal_stats, check_none=True)

        # Record the cluster dtype for convenience
        self.cluster_dtype = copy.copy(cluster_dtype_base)
        self.cluster_dtype.extend([('MAG', 'f4', self.nmag),
                                   ('MAG_ERR', 'f4', self.nmag),
                                   ('PZBINS', 'f4', self.npzbins),
                                   ('PZ', 'f4', self.npzbins)])
        self.member_dtype = copy.copy(member_dtype_base)
        self.member_dtype.extend([('MAG', 'f4', self.nmag),
                                  ('MAG_ERR', 'f4', self.nmag)])
        # also need pz stuff, etc, etc.  Will need to deal with defaults

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

    def _set_vars_from_dict(self, d, check_none=False):
        for key in d:
            if check_none and d[key] is None:
                continue
            if key not in type(self).__dict__:
                raise AttributeError("Unknown config variable: %s" % (key))
            setattr(self, key, d[key])


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

        else:
            # statistics are from the master table file
            gal_stats['galfile_pixelized'] = True

            master=fitsio.read(self.galfile,ext=1)

            mode = master['MODE'][0].rstrip()
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

        return gal_stats


