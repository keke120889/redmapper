#!/usr/bin/env python

from __future__ import division, absolute_import, print_function

import os
import sys
import argparse
import redmapper

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calibrate the redMaGiC red galaxy selection')

    parser.add_argument('-c', '--configfile', action='store', type=str, required=True,
                        help='YAML config file')

    args = parser.parse_args()

    calib = redmapper.redmagic.RedmagicCalibrator(args.configfile)
    calib.run()
