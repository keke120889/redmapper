#!/usr/bin/env python

from __future__ import division, absolute_import, print_function

import os
import sys
import argparse
import redmapper

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute random weights')

    parser.add_argument('-c', '--configfile', action='store', type=str, required=True,
                        help='YAML config file')
    parser.add_argument('-r', '--randfile', action='store', type=str, required=True,
                        help='Random file to compute weights')
    parser.add_argument('-l', '--lambda_cuts', action='store', type=float, nargs='+',
                        required=True, help='Minimum richness')

    args = parser.parse_args()

    config = redmapper.Configuration(args.configfile)
    weigher = redmapper.RandomWeigher(config, args.randfile)
    for lambda_cut in args.lambda_cuts:
        wt_randfile, wt_areafile = weigher.weight_randoms(lambda_cut)

        print("Made weighted random file %s" % (wt_randfile))
        print("Made weighted area file %s" % (wt_areafile))
