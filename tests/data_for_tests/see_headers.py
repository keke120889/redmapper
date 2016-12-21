"""
This is a very small script that is made so that 
we can look into the fits files in this directory
without using print statements in the unit
tests themselves.
"""

import fitsio

filename = "test_cluster_members.fit"
data,h = fitsio.read(filename,header=True)
out = data.dtype.names
for i in range(len(out)):
    print out[i]
