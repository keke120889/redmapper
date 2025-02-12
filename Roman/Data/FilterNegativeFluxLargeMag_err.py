import numpy as np

Mag     = np.load('/project/chihway/ihsuan/Roman/Data/CombineMagnitude.npy', mmap_mode='r')
Mag_err = np.load('/project/chihway/ihsuan/Roman/Data/CombineMagErr.npy',    mmap_mode='r')

Filter1 = ~np.isnan(Mag)
Filter2 = np.isfinite(Mag)
Filter3 = Mag_err < 90.0
Filter  = np.logical_and(Filter1, Filter2)
Filter  = np.logical_and(Filter,  Filter3)

del Filter1, Filter2, Filter3

Filter = np.all(Filter, axis=1)

np.save('FilterNegativeFluxLargeMag_err.npy', Filter)
