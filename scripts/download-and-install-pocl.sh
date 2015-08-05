#!/bin/bash
#
# Script used on CI server to:
#
# - Download pocl
# - Compile pocl
# - Install pocl to $HOME/opt/pocl
# - Set default values in ~/.aksetup-defaults.py to enable pip installation of
#   PyOpenCL. (Does not touch the file if it already exists.)

# Trace execution and exit on error
set -xe

POCL_VERSION=0.11
POCL_URL=http://portablecl.org/downloads/pocl-${POCL_VERSION}.tar.gz
POCL_PREFIX="${HOME}/opt/pocl"

TMP_DIR=$(mktemp -d --tmpdir pocl-build.XXXXXX)
cd "${TMP_DIR}"

echo "Downloading pocl..."
curl -O "${POCL_URL}"

echo "Extracting..."
tar xvf pocl-${POCL_VERSION}.tar.gz

echo "Overriding llvm-config tool."
LLVM_VERSION=3.6
LLVM_CONFIG=$(which llvm-config-${LLVM_VERSION})
if [ -z "${LLVM_CONFIG}" ]; then
	echo "No llvm-config-${LLVM_VERSION} tool found." >&2
	exit 1
fi
mkdir bin
export PATH=$PWD/bin:$PATH
ln -s "${LLVM_CONFIG}" bin/llvm-config

echo "Compiling..."
cd pocl-${POCL_VERSION}
CXX=g++-5 CC=gcc-5 ./configure --prefix=${POCL_PREFIX} --disable-icd
make
make install

echo "Removing ${TMP_DIR}..."
rm -r "${TMP_DIR}"

AKSETUP_DEFAULTS="${HOME}/.aksetup-defaults.py"
if [ ! -f "${AKSETUP_DEFAULTS}" ]; then
	echo "Adding $AKSETUP_DEFAULTS configuration for PyOpenCL."
	cat >"${AKSETUP_DEFAULTS}" <<EOL
import os
CL_INC_DIR=[os.path.expanduser('~/opt/pocl/include')]
CL_LIB_DIR=[os.path.expanduser('~/opt/pocl/lib')]
EOL
else
	echo "WARNING: not updating $AKSETUP_DEFAULTS as file exists. PyOpenCL may not build."
fi
