#
# This is for the test of making the redMaPPer input galaxy catalog (100 deg^2).
#
# 20250124 IHsuan starts
#
########################################
import h5py
import numpy as np
import redmapper

# Read in data
cat       = h5py.File('/project/chihway/chto/Roman/combine_v2.h5', 'r')
catselect = np.load('/project/chihway/chto/Roman/batchrun/Rd_select.npy',      mmap_mode='r')

# choose the 100 deg^2 region
ra           = np.array(cat['catalog/gold/ra'][:][catselect],  dtype='f8')
dec          = np.array(cat['catalog/gold/dec'][:][catselect], dtype='f8')
FilterRA     = np.logical_and(130.0 <= ra, ra <= 140.0)
FilterDEC    = np.logical_and(20.0 <= dec, dec <= 30.0)
FilterRegion = np.logical_and(FilterRA, FilterDEC)

del FilterRA, FilterDEC

Mag       = np.load('/project/chihway/ihsuan/Roman/Data/CombineMagnitude.npy', mmap_mode='r')[FilterRegion]
Mag_err   = np.load('/project/chihway/ihsuan/Roman/Data/CombineMagErr.npy',    mmap_mode='r')[FilterRegion]
Filter    = np.load('/home/ihsuan/redmapper/Roman/Data/FilterNegativeFluxLargeMag_err.npy', mmap_mode='r')[FilterRegion]

galaxy_id = np.array(cat['catalog/gold/id'][:][catselect][FilterRegion],  dtype='i8')
ra        = np.array(cat['catalog/gold/ra'][:][catselect][FilterRegion],  dtype='f8')
dec       = np.array(cat['catalog/gold/dec'][:][catselect][FilterRegion], dtype='f8')
ztrue     = np.array(cat['catalog/gold/z'][:][catselect][FilterRegion],   dtype='f4')

del cat, catselect

# refmag: z band
refmag     = np.array(Mag[:, 4],     dtype='f4')
refmag_err = np.array(Mag_err[:, 4], dtype='f4')
# mag, we include the refmag.
mag        = np.array(Mag,     dtype='f4')
mag_err    = np.array(Mag_err, dtype='f4')

del Mag, Mag_err

# ebv, ignore, set it to zero
ebv = np.zeros(np.shape(refmag))

# No m200, central, halo_id

# Save
zp = 30.0
bands = ['u', 'g', 'r', 'i', 'z', 'Y', 'J', 'H', 'F']
refband = 'z'
nmag = len(bands) # Include the ref. band

info_dict = {}
info_dict['LIM_REF'] = 24.65326 # limiting mag. of ref. band
info_dict['REF_IND'] = 4        # z-band
info_dict['AREA']    = 100.0    # catalog area
info_dict['NMAG']    = nmag     
info_dict['MODE']    = 'LSST'   
info_dict['ZP']      = zp       
info_dict['U_IND']   = 0        # u-band index
info_dict['G_IND']   = 1
info_dict['R_IND']   = 2
info_dict['I_IND']   = 3
info_dict['Z_IND']   = 4
info_dict['Y_IND']   = 5
info_dict['J_IND']   = 6
info_dict['H_IND']   = 7
info_dict['F_IND']   = 8
# the rest of the bands...

maker = redmapper.GalaxyCatalogMaker('/project/chihway/ihsuan/Roman/Data/InputCatalog/Create_redMaPPer_Input_Catalog_Test100deg2', info_dict)

redmapper_dtype = [('id',         'i8'),
                   ('ra',         'f8'),
                   ('dec',        'f8'),
                   ('ztrue',      'f4'),
                   ('refmag',     'f4'),
                   ('refmag_err', 'f4'),
                   ('mag',        'f4', nmag),
                   ('mag_err',    'f4', nmag),
                   ('ebv',        'f4')]

galaxies = np.zeros(np.sum(Filter), dtype=redmapper_dtype)
galaxies['id']         = galaxy_id[Filter]
galaxies['ra']         = ra[Filter]
galaxies['dec']        = dec[Filter]
galaxies['ztrue']      = ztrue[Filter]
galaxies['refmag']     = refmag[Filter]
galaxies['refmag_err'] = refmag_err[Filter]
galaxies['mag']        = mag[Filter]
galaxies['mag_err']    = mag_err[Filter]
galaxies['ebv']        = ebv[Filter]

maker.append_galaxies(galaxies)
maker.finalize_catalog()
