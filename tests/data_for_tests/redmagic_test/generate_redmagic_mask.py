#!/usr/bin/env python

import healsparse
import healpy as hp
import numpy as np
import redmapper
import esutil

nside = 512
nsideCoverage = 32

gals = redmapper.GalaxyCatalog.from_fits_file('redmagic_test_input_gals.fit')

theta = np.radians(90.0 - gals.dec)
phi = np.radians(gals.ra)

ipnest = hp.ang2pix(nside, theta, phi, nest=True)

hist = esutil.stat.histogram(ipnest, min=0, max=hp.nside2npix(nside))

gdPix, = np.where(hist > 0)

sparseMap = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nside, dtype=np.float32)
sparseMap.updateValues(gdPix, np.ones(gdPix.size, dtype=np.float32))

sparseMap.write('redmagic_test_mask_hs.fit')

