import os
from setuptools import setup
from io import open

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

requirements = [
                'toolz>=0.8',
                'distributed>=1.13.3',
                'graphviz>=0.5',
                'cairocffi>=0.7',
                'dask>=0.11.1',
                'sobol_seq>=0.1.2',
                'numpy>=1.8',
                'scipy>=0.16.1',
                'Cython>=0.25.1',
                'matplotlib>=1.1',
                'GPy>=1.0.9',
                'unqlite>=0.6.0'
                ]

setup(
    name='elfi',
    packages=['elfi'],
    version='0.3',
    author='HIIT',
    author_email='aki.vehtari@aalto.fi',
    url='https://github.com/HIIT/elfi',

    install_requires=requirements,

    extras_require={
        'doc': ['Sphinx'],
    },

    description='Modular ABC inference framework for python',
    long_description=long_description,

    license='BSD3',

    classifiers=['Programming Language :: Python',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Bio-Informatics',
                 'Programming Language :: Python :: 3'
                 'Operating System :: OS Independent',
                 'Development Status :: 2 - Pre-Alpha',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: BSD3 License'])
