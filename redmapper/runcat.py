import fitsio
import esutil

import config
from cluster import ClusterCatalog


def run(confdict=None, conffile=None, outbase=None, 
                        savemembers=False, mask=False):
    
    """
    Name:
        run
    Purpose:
        Run the redmapper cluster finding algorithm.
    Calling sequence:
        TBD
    Inputs:
        confdict: A configuration dictionary containing information
                  about how this run of RM works. Note that
                  one of confdict or conffile is required.
        conffile: A configuration file containing information
                  aboud how this run of RM works. Note that
                  one of confdict or conffile is required.
    Optional Inputs:
        outbase: Directory location of where to put outputs.
        savemembers: TBD
        mask = TBD
    """

    # Read configurations from either explicit dict or YAML file
    if (confdict is None) and (conffile is None):
        raise ValueError("Must have one of confdict or conffile")
    if (confdict is not None) and (conffile is not None):
        raise ValueError("Must have only one of confdict or conffile")
    if conffile is not None: confdict = config.read_config(conffile)

    r0, beta = confdict['percolation_r0'], confdict['percolation_beta']

    # This allows us to override outbase on the call line
    if outbase is None: outbase = confdict['outbase']

    # Read in the background from the bkgfile
    bkg = None # To be implemented

    # Read in the pars from the parfile
    pars = None # To be implemented

    # Read in masked galaxies
    maskgals = None #fitsio.read(confdict['maskgalfile'], ext=1) # To be implemented

    # Read in the mask
    maskfile = confdict['maskfile'] #To be implemented

    # Read in the input catalog from the catfile
    #NOTE from Tom - I think that this is the old "galfile" in IDL
    clusters = ClusterCatalog.from_fits_file(confdict['catfile'])

    return clusters



