#
# This is for making the redMaPPer spectroscopic input galaxy catalog (2000 galaxies in the 100 deg^2 region).
# 20250127 IHsuan starts
<<<<<<< HEAD
# 20250129        finishes
=======
# 20250212        finishes
>>>>>>> 9780bef47f20be6af2f2cb478a3115d0ea79139e
#
########################################
import h5py
import numpy as np
import random
from astropy.table import Table

# Read in the data
cat       = h5py.File('/project/chihway/chto/Roman/combine_v2.h5', 'r')
catselect = np.load('/project/chihway/chto/Roman/batchrun/Rd_select.npy', mmap_mode='r')

# choose the 100 deg^2 region
ra           = np.array(cat['catalog/gold/ra'][:][catselect],  dtype='f8')
dec          = np.array(cat['catalog/gold/dec'][:][catselect], dtype='f8')
FilterRA     = np.logical_and(130.0 <= ra, ra <= 140.0)
FilterDEC    = np.logical_and(20.0 <= dec, dec <= 30.0)
FilterRegion = np.logical_and(FilterRA, FilterDEC)

# Randomly select 2000 galaxies
RandomSample = np.sort(random.sample(range(0, np.sum(FilterRegion)), 200000))

np.save('/project/chihway/chto/Create_redMaPPer_Spectroscopic_Input_Catalog_Test100deg2.npy', RandomSample)
