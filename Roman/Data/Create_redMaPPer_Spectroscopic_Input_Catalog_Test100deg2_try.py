#
# This is for making the redMaPPer spectroscopic input galaxy catalog (2000 galaxies in the 100 deg^2 region).
# 20250127 IHsuan starts
# 20250218        finishes
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

# Randomly select 200000 galaxies
# RandomSample = np.sort(random.sample(range(0, np.sum(FilterRegion)), 200000))

# np.save('/project/chihway/ihsuan/Roman/Data/Create_redMaPPer_Spectroscopic_Input_Catalog_Test100deg2_try.npy', RandomSample)
ztrue     = np.array(cat['catalog/gold/z'][:][catselect][FilterRegion],   dtype='f4')

RedshiftBin = np.arange(0.0, np.max(ztrue), 0.05)

FinalSample = []

for i in range(len(RedshiftBin) - 2):
    Filter = np.logical_and(RedshiftBin[i] <= ztrue, ztrue < RedshiftBin[i + 1])
    FilterIndex = np.where(Filter)[0]
    # These need not be complete, but for adequate performance you want at least 40 clusters per 0.05 redshift bin.
    RandomSample = random.sample(list(FilterIndex), 50)
    FinalSample.append(RandomSample)

FinalSample = np.reshape(FinalSample, (len(FinalSample * 50), ))
FinalSample = np.sort(FinalSample)

ra        = np.array(cat['catalog/gold/ra'][:][catselect][FilterRegion][FinalSample],  dtype='f8')
dec       = np.array(cat['catalog/gold/dec'][:][catselect][FilterRegion][FinalSample], dtype='f8')
z_err     = np.ones(np.shape(dec)) * 0.00001

np.save('/project/chihway/ihsuan/Roman/Data/Create_redMaPPer_Spectroscopic_Input_Catalog_Test100deg2_try.npy', FinalSample)

# Save the data
table = Table()
table['ra'] = ra
table['dec'] = dec
table['z'] = ztrue[FinalSample]
table['z_err'] = z_err

table.write('/project/chihway/ihsuan/Roman/Data/Create_redMaPPer_Spectroscopic_Input_Catalog_Test100deg2_try.fits', format='fits', overwrite=True)
