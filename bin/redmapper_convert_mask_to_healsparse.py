#!/usr/bin/env python

from __future__ import division, absolute_import, print_function

import os
import sys
import argparse
import fitsio
import redmapper

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert old redmapper geometry mask to healsparse mask')

    parser.add_argument('-m', '--maskfile', action='store', type=str, required=True,
                        help='Input mask file (old format)')
    parser.add_argument('-f', '--healsparsefile', action='store', type=str, required=True,
                        help='Output mask healsparse file (new format)')
    parser.add_argument('-n', '--nsideCoverage', action='store', type=int, default=32,
                        help='Coverage nside for healsparse file')
    parser.add_argument('-C', '--clobber', action='store_true',
                        help='Clobber output file?')

    args = parser.parse_args()

    redmapper.mask.convert_maskfile_to_healsparse(args.maskfile, args.healsparsefile,
                                                  args.nsideCoverage, clobber=args.clobber)
