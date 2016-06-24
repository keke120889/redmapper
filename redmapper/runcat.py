import fitsio
import esutil

import config


def run(confdict=None, conffile=None, outbase=None, 
                        savemembers=False, mask=False):
    '''
    docstring
    '''

    if (confdict is None) and (conffile is None):
        raise ValueError("Must have one of confdict or conffile")

    if (confdict is not None) and (conffile is not None):
        raise ValueError("Must have only one of confdict or conffile")

    # read in config file
    if conffile is not None:
        confdict = config.read_config(conffile)

    # this allows us to override outbase on the call line
    if outbase is None:
        outbase = confdict['outbase']

    # read in the input catalog
    incat = fitsio.read(confdict['catfile'], ext=1)

    # etc, etc.

    
    

    
    
        
    
