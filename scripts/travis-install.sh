#!/bin/bash
#
# Script used to prepare CI environment on travis.

# Enable tracing and exit on error
set -xe

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

pip install --upgrade pip
pip install tox coveralls

case "${TOX_ENV}" in
    *-opencl)
        # Special setup required when we're testing OpenCL.
        $DIR/download-and-install-pocl.sh
        ;;
esac

# vim:sw=4:sts=4:et:ts=4
