# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='RedMaPPer',
    version='0.0.1',
    description='Public, Python implementation of RedMaPPer',
    long_description=readme,
    author='Eli Rykoff, Brett Harvey',
    author_email='erykoff@slac.stanford.edu, rbharvey@stanford.edu',
    url='https://github.com/erykoff/redmapper',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

