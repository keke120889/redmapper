#
# This is for combining the Roman and LSST flux and computing the magnitude.
#
# 20250124 IHsuan starts
# 20250204        finishes
#
###############################
import numpy as np
from astropy.table import Table

magRoman    = 22.5 - 2.5 * np.log10(np.load('/project/chihway/chto/Roman/batchrun/Rd_Roman.npy', mmap_mode='r'))
magLSST     = 22.5 - 2.5 * np.log10(np.load('/project/chihway/chto/Roman/batchrun/Rd_LSST.npy',  mmap_mode='r'))[:, 0:-1]

mag = np.concatenate((magLSST, magRoman),
                     axis=1)

del magRoman
del magLSST

#FluxerrRoman = np.load('/project/chihway/chto/Roman/batchrun/Rd_Romanerr.npy')
#FluxerrLSST  = np.load('/project/chihway/chto/Roman/batchrun/Rd_LSSTerr.npy')

#Fluxerr = np.concatenate((FluxerrRoman, FluxerrLSST),
#                         axis=1)

#del FluxerrRoman
#del FluxerrLSST


# https://www.sdss4.org/dr17/algorithms/magnitudes/#Fluxunits:maggiesandnanomaggies
#mag     = 22.5 - 2.5 * np.log10(Flux)

#del Flux

#t = Table()
#t['mag']     = mag

np.save('/project/chihway/ihsuan/Roman/Data/CombineMagnitude.npy', mag)

del mag

#mag_err = 2.5 * (Fluxerr/(Flux * np.log(10)))

#del Flux
#del Fluxerr

#t['mag_err'] = mag_err

#del mag_err

#t.write('CombineMagnitude.fits', format='fits', overwrite=True)
