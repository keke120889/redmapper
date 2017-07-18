"""
This is a very small script that is made so that 
we can look into the fits files in this directory
without using print statements in the unit
tests themselves.
"""

import fitsio

filename = "test_cluster_members.fit"
#filename = "testgals_chisq_mode0.fit"
#filename = "pixelized_dr8_test/dr8_test_galaxies_master_table.fit"
#filename = "pixelized_dr8_test/dr8_test_galaxies_0008421.fit"
#filename = "test_dr8_pars.fit"
#filename = "test_dr8_mask.fit"
#filename = "maskgals.fit"
#filename = "test_bkg.fit"
#filename = "test_dr8_depth.fit"
data,h = fitsio.read(filename,header=True)

#to get more information
dat = fitsio.read(filename, ext=1)

out = data.dtype.names
print filename
for i in range(len(out)):
    print out[i]
