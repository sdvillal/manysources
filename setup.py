#!/usr/bin/env python2
# coding=utf-8

# Authors: Santi Villalba <sdvillal@gmail.com>
# Licence: BSD 3 clause

# TODO: include flo as author

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import manysources

setup(
    name='manysources',
    license='BSD 3 clause',
    description='model<->example<->prediction interactions with a chemical twist',
    long_description=open('README.rst').read().replace('|Build Status| |Coverage Status|', ''),
    version=manysources.__version__,
    url='https://github.com/sdvillal/whatami',
    author='Santi Villalba',
    author_email='sdvillal@gmail.com',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Operating System :: Unix',
    ],
    install_requires=[
        'h5py',
        'scipy',
        'numpy',
        'pandas',
        'joblib',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'whatami',
        'argh',
        'rdkit',
        'numba',
        'cytoolz', 
        'tsne',
        'networkx',
    ],
    tests_require=['pytest'],
    platforms=['Any'],
)
