from setuptools import setup, find_packages
from codecs import open
from os import path
from gammaALPs.version import get_git_version_pypi

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='gammaALPs',
    version=get_git_version_pypi(),
    include_package_data=True,
    description='Python code to calculate the conversion probability between photons and axions / axion-like'
                'particles in astrophysical magnetic fields',
    long_description=long_description,  # this is the readme
    long_description_content_type='text/x-rst',

    # The project's main homepage.
    url='https://github.com/me-manu/gammaALPs',

    # Author details
    author='Manuel Meyer',
    author_email='me.manu@gmx.net',

    # Choose your license
    license='BSD',

    # See https://PyPI.python.org/PyPI?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],

    # What does your project relate to?
    keywords=['gamma-ray astronomy', 'particle physics', 'axions'],

    packages=find_packages(exclude=['build', 'docs', 'templates']),

    install_requires=[
        'numpy >= 1.19',
        'scipy >= 1.5',
        'numba >= 0.51',
        'astropy>=4.0',
        'ebltable >=0.2'
    ]

)
