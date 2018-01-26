import fitsio
import esutil

import config as rmconfig
from cluster import ClusterCatalog
from background import Background
from mask import HPMask
from galaxy import GalaxyCatalog
from cluster import Cluster
from cluster import ClusterCatalog
from depthmap import DepthMap
from zlambda import Zlambda
from zlambda import ZlambdaCorrectionPar
#from runcat_base import RunCatalogBase
from cluster_runner import ClusterRunner

class RunCatalog(ClusterRunner):
    """
    """

    def _more_setup(self):
        # read in catalog, etc

        pass

    def _process_cluster(self, *args):
        # here is where the work on an individual cluster is done

        pass

    #def run(self, mask=False):
    #    """
    #    """

        # The "runcat" mode does masking based on the "percolation" settings.
    #    self._setup('percolation')

        # read in input catalog
    #    incat = fitsio.read(self.config['catfile'], ext=1)

        # cut down the input catalog to those that are within the pixel


        # loop over clusters

        # compute limiting mag stuff if depth not available
        #  (needs to be tested)


        # iterate here!

        # compute richness

        # and z_lambda

        # and +/-

        # apply any zlambda corrections

        # copy selection of members

        # and record pfree values...


    #def output(self, savemembers=True):
    #    pass

    

def run(confdict=None, conffile=None, outbase=None, 
                        savemembers=False, mask=False):
    
    """
    Name:
        run
    Purpose:
        Run the redmapper cluster finding algorithm.
    Calling sequence:
        TODO
    Inputs:
        confdict: A configuration dictionary containing information
                  about how this run of RM works. Note that
                  one of confdict or conffile is required.
        conffile: A configuration file containing information
                  aboud how this run of RM works. Note that
                  one of confdict or conffile is required.
    Optional Inputs:
        outbase: Directory location of where to put outputs.
        savemembers: TODO
        mask = TODO
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
    bkg = None # TODO

    # Read in the pars from the parfile
    pars = None # TODO

    # Read in masked galaxies
    maskgals = None #fitsio.read(confdict['maskgalfile'], ext=1) # TODO

    # Read in the mask
    maskfile = confdict['maskfile'] #TODO

    # Read in the input catalog from the catfile
    #NOTE from Tom - I think that this is the old "galfile" in IDL
    clusters = ClusterCatalog.from_fits_file(confdict['catfile'])

    return clusters



