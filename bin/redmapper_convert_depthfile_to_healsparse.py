#!/usr/bin/env python

from __future__ import division, absolute_import, print_function

import os
import sys
import argparse
import fitsio
import redmapper

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert old redmapper depth file to healsparse mask')

    parser.add_argument('-m', '--depthfile', action='store', type=str, required=True,
                        help='Input depth file (old format)')
    parser.add_argument('-f', '--healsparsefile', action='store', type=str, required=True,
                        help='Output mask healsparse file (new format)')
    parser.add_argument('-n', '--nsideCoverage', action='store', type=int, default=32,
                        help='Coverage nside for healsparse file')
    parser.add_argument('-C', '--clobber', action='store_true',
                        help='Clobber output file?')

    args = parser.parse_args()

    redmapper.depthmap.convert_depthfile_to_healsparse(args.depthfile, args.healsparsefile,
                                                       args.nsideCoverage, clobber=args.clobber)
