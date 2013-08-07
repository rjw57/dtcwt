import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "dtcwt",
    version = "0.1",
    author = "Rich Wareham",
    author_email = "rich.dtcwt@richwareham.com",
    description = ("A port of the Dual-Tree Complex Wavelet Transform MATLAB toolbox."),
    license = "Free To Use But Restricted",
    keywords = "numpy, wavelet, complex wavelet, DT-CWT",
    url = "http://packages.python.org/dtcwt",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    long_description=read('README.rst'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: Free To Use But Restricted",
    ],

    setup_requires=['nose>=1.0','coverage','sphinx',],

    install_requires=['numpy','scipy',],
)
