from setuptools import setup
from io import open

with open('README.rst', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='abcpy',
    packages=['abcpy'],
    version='0.1',
    author='HIIT',
    author_email='aki.vehtari@aalto.fi',
    url='https://github.com/HIIT/abcpy',

    install_requires=[
        'numpy>=1.8',
        'scipy>=0.16.1',
        'toolz>=0.8',
        'distributed>=1.13',
        'graphviz>=0.5',
        'cairocffi>=0.7',
        'dask>=0.11',
        'matplotlib>=1.1',
        'sobol_seq>=0.1.2',
        'six>=1.5',
        'decorator>=3.4',
        'GPy>=1.0.9'
    ],

    extras_require={
        'doc': ['Sphinx'],
        'dev': [
            'Sphinx',
            'pytest',
            'tox',
            'pep8',
            ],
    },

    description='Modular ABC inference framework for python',
    long_description=long_description,

    license='MIT',

    classifiers=['Programming Language :: Python',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Bio-Informatics',
                 'Programming Language :: Python :: 3'
                 'Operating System :: OS Independent',
                 'Development Status :: 2 - Pre-Alpha',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License'])
