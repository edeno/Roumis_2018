#!/usr/bin/env python3

from setuptools import find_packages, setup

INSTALL_REQUIRES = ['loren_frank_data_processing', 'ripple_detection',
                    'spectral_connectivity', 'replay_classification']
TESTS_REQUIRE = ['pytest >= 2.7.1']

setup(
    name='roumis_2018',
    version='0.1.0.dev0',
    license='GPL-3.0',
    description=('Analysis of MEC and CA1 connectivity'),
    author='Demetris Roumis, Eric Denovellis',
    author_email='demetris.roumis@ucsf.edu, edeno@bu.edu',
    url='https://github.com/edeno/Roumis_2018',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
)
