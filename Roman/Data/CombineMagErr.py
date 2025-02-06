#
# This is for computing and combining the mag_err of Roman and LSST.
#
# 20250127 IHsuan starts
# 20250204        finishes
#
###############################
import numpy as np

FluxRoman    = np.load('/project/chihway/chto/Roman/batchrun/Rd_Roman.npy',    mmap_mode='r')
FluxerrRoman = np.load('/project/chihway/chto/Roman/batchrun/Rd_Romanerr.npy', mmap_mode='r')

mag_errRoman = 2.5 * (FluxerrRoman/(FluxRoman * np.log(10)))

del FluxRoman, FluxerrRoman

FluxLSST    = np.load('/project/chihway/chto/Roman/batchrun/Rd_LSST.npy',    mmap_mode='r')
FluxerrLSST = np.load('/project/chihway/chto/Roman/batchrun/Rd_LSSTerr.npy', mmap_mode='r')

mag_errLSST = 2.5 * (FluxerrLSST/(FluxLSST * np.log(10)))[:, 0:-1]

del FluxLSST, FluxerrLSST

mag_err = np.concatenate((mag_errLSST, mag_errRoman),
                         axis=1)

del mag_errRoman, mag_errLSST

np.save('/project/chihway/ihsuan/Roman/Data/CombineMagErr.npy', mag_err)

del mag_err
