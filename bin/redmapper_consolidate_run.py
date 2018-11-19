#!/usr/bin/env python

from __future__ import division, absolute_import, print_function

import os
import sys
import argparse
import redmapper

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Consolidate a redMaPPer parallel run')

    parser.add_argument('-c', '--configfile', action='store', type=str, required=True,
                        help='YAML config file')
    parser.add_argument('-l', '--minlambdas', action='store', type=float,
                        nargs='+', required=False, help='Minimum lambdas')
    parser.add_argument('-v', '--vlim_lstars', action='store', type=float,
                        nargs='+', required=False, help='Volume-limit L* cuts')

    args = parser.parse_args()

    consolidate = redmapper.pipeline.RedmapperConsolidateTask(args.configfile, minlambdas=args.minlambdas, vlim_lstars=args.vlim_lstars)
    consolidate.run()
