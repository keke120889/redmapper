#!/usr/bin/env python

from __future__ import division, absolute_import, print_function

import os
import sys
import argparse
import fitsio
import healsparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert old redmapper geometry mask to healsparse mask')

    parser.add_argument('-m', '--maskfile', action='store', type=str, required=True,
                        help='Input mask file (old format)')
    parser.add_argument('-h', '--healsparsefile', action='store', type=str, required=True,
                        help='Output mask healsparse file (new format)')
    parser.add_argument('-n', '--nsideCoverage', action='store', type=int, default=32,
                        help='Coverage nside for healsparse file')
    parser.add_argument('-C', '--clobber', action='store_true',
                        help='Clobber output file?')

    args = parser.parse_args()

    old_mask, hdr = fitsio.read(args.maskfile, ext=1, header=True, lower=True)

    nside = hdr['nside']

    sparseMap = healsparse.HealSparseMap.makeEmpty(args.nsideCoverage, nside, old_mask['fracgood'].dtype)
    sparseMap.updateValues(old_mask['hpix'], old_mask['fracgood'], nest=hdr['nest'])

    sparseMap.write(args.healsparsefile, clobber=args.clobber)
