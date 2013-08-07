import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

# IMPORTANT: before release, remove the 'devN' tag from the release name
version = '0.3'
release = '0.3.0.dev1'

setup(
    name = 'dtcwt',
    version = release,
    author = "Rich Wareham",
    author_email = "rich.dtcwt@richwareham.com",
    description = ("A port of the Dual-Tree Complex Wavelet Transform MATLAB toolbox."),
    license = "Free To Use But Restricted",
    keywords = "numpy, wavelet, complex wavelet, DT-CWT",
    url = "https://github.com/rjw57/dtcwt",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    long_description=read('README.rst'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: Free To Use But Restricted",
    ],
    package_data = {
        'dtcwt': ['data/*.npz',],
    },

    setup_requires=[ 'nose>=1.0', ],

    install_requires=[ 'numpy', 'six', ],

    extras_require={
        'docs': [ 'sphinx', 'docutils', ],
    },

    tests_require=[ 'coverage', ],

    command_options = {
        'build_sphinx': {
            'version': ( 'setup.py', version ),
            'release': ( 'setup.py', release ),
        },
    },
)

# vim:sw=4:sts=4:et
