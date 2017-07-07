import yaml
import fitsio

def read_yaml(filename, defaults=None):
    
    """
    Name:
        read_yaml
    Purpose:
        Read in a YAML file with key/value pairs and put into dict.
    Calling Sequence:
        outdict = read_yaml(filename, defaults=None)
    Inputs:
        filename: YAML file name
    Optional Inputs:
        defaults: Dictionary of required/default fields/values. All fields in 
                  defaults are required to be specified. If they are set to 
                  None, they must be specified by filename.
    Outputs:
        outdict: a dictionary of the key/value pairs in the YAML file.
    """
    
    # The output dictionary
    outdict = defaults if defaults is not None else {}
    
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

    def __init__(self, conffile):
        defaults = {'hpix':0,
                    'nside':0}

        confdict = read_yaml(conffile,defaults=defaults)

        ## FIXME: check for required inputs

        for key in confdict:
            setattr(self, key, confdict[key])

        # get galaxy file stats
        gal_stats = self.galfile_stats(confdict['galfile'], confdict['refmag'])

        for key in gal_stats:
            setattr(self, key, gal_stats[key])

    def galfile_stats(self, galfile, refmag):
        """
        """

        hdr = fitsio.read_header(galfile, ext=1)
        pixelated = hdr.get("PIXELS", 0)
        fitsformat = hdr.get("FITS", 0)

        if not fitsformat:
            raise ValueError("Input galfile must describe fits files.")

        gal_stats = {}

        if not pixelated:
            # statistics are from the header
            gal_stats['pixelized'] = False

            hdrmode = hdr.get("MODE", "").rstrip()

            if hdrmode == 'SDSS':
                gal_stats['mode'] = 0
            elif hdrmode == 'DES':
                gal_stats['mode'] = 1
            else:
                raise ValueError("Input galaxy file with unknown mode: %s" % (hdrmode))

            ## FIXME: check that these are all in the header
            gal_stats['area'] = hdr.get('AREA', -100.0)
            gal_stats['limmag_ref'] = hdr.get('LIM_REF')
            gal_stats['nmag'] = hdr.get('NMAG')
            gal_stats['zeropoint'] = hdr.get('ZP')
            gal_stats['ref_ind'] = hdr.get(refmag.upper()+'_IND')
            gal_stats['b'] = 0

        else:
            # statistics are from the master table file
            gal_stats['pixelized'] = True

            master=fitsio.read(galfile,ext=1)

            mode = master['MODE'][0].rstrip()
            if (mode == 'SDSS'):
                gal_stats['mode'] = 0
            elif (mode == 'DES'):
                gal_stats['mode'] = 1
            else:
                raise ValueError("Input galaxy file with unknown mode: %s" % (mode))

            gal_stats['area'] = master['AREA'][0]
            gal_stats['limmag_ref'] = master['LIM_REF'][0]
            gal_stats['nmag'] = master['NMAG'][0]
            if ('B' in master.dtype.names):
                gal_stats['b'] = master['B'][0]
            else:
                gal_stats['b'] = 0
            gal_stats['zeropoint'] = master['ZEROPOINT'][0]
            gal_stats['ref_ind'] = master[refmag.upper()+'_IND'][0]

        return gal_stats
