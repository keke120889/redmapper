#!/usr/bin/env python

from __future__ import division, absolute_import, print_function

import os
import sys
import argparse
import fitsio
import healsparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert old redmapper depth file to healsparse mask')

    parser.add_argument('-m', '--depthfile', action='store', type=str, required=True,
                        help='Input depth file (old format)')
    parser.add_argument('-h', '--healsparsefile', action='store', type=str, required=True,
                        help='Output mask healsparse file (new format)')
    parser.add_argument('-n', '--nsideCoverage', action='store', type=int, default=32,
                        help='Coverage nside for healsparse file')
    parser.add_argument('-C', '--clobber', action='store_true',
                        help='Clobber output file?')

    args = parser.parse_args()

    old_depth, hdr = fitsio.read(args.depthfile, ext=1, header=True, lower=True)

    nside = hdr['nside']

    # Need to remove the HPIX from the dtype

    dtype_new = []
    names = []
    for d in old_depth.dtype.descr:
        if d[0] != 'hpix':
            dtype_new.append(d)
            names.append(d[0])

    sparseMap = healsparse.HealSparseMap.makeEmpty(args.nsideCoverage, nside, dtype_new)
    sparseMap.updateValues(old_depth['hpix'], old_depth[names], nest=hdr['nest'])

    sparseMap.write(args.healsparsefile, clobber=args.clobber)
