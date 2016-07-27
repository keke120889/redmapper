# -*- coding: utf-8 -*-

from setuptools import setup, find_packages, Extension
import numpy
import glob

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

exec(open('redmapper/_version.py').read())

include_dirs = [numpy.get_include()]

ext_modules=[]

# solver_nfw
solver_nfw_sources=['redmapper/solver_nfw/solver_nfw_pywrap.c',
                    'redmapper/solver_nfw/solver_nfw.c',
                    'redmapper/solver_nfw/nfw_weights.c']
solver_nfw_module = Extension('redmapper.solver_nfw._solver_nfw_pywrap',
                              extra_compile_args=['-std=gnu99'],
                              sources=solver_nfw_sources,
                              include_dirs=include_dirs)
ext_modules.append(solver_nfw_module)

# data files
initcolorfiles = glob.glob('data/initcolors/*.fit')
mstarfiles = glob.glob('data/mstar/*.fit')
    
setup(
    name='redmapper',
    version=__version__,
    description='Public, Python implementation of redMaPPer',
    long_description=readme,
    author='Eli Rykoff, Brett Harvey',
    author_email='erykoff@slac.stanford.edu, rbharvey@stanford.edu',
    url='https://github.com/erykoff/redmapper',
    license=license,
    ext_modules=ext_modules,
    install_requires=['numpy'],
    packages=find_packages(exclude=('tests', 'docs')),
    data_files=[('redmapper/data/initcolors', initcolorfiles),
                ('redmapper/data/mstar', mstarfiles)]
)

