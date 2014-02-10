#!/bin/bash

## CONFIGURATION

# URL to download toolboxen from
TOOLBOX_URL=https://github.com/timseries/dtcwt_matlab/archive/master.zip

# Path to dtcwt toolbox
DTCWT_TOOLBOX=$HOME/Downloads/dtcwt_toolbox4_3

# Path to dtcwt keypoints toolbox
DTCWT_KEYPOINTS=$HOME/Downloads/DTCWTkeypoints

# Path to dtcwt 3D toolbox
DTCWT_3D=$HOME/Downloads/DTCWT_3D

# Path to MATLAB
MATLAB=/opt/MATLAB/R2013b/bin/matlab

## END OF CONFIGURATION

# Change to this directory
cd "`dirname "${BASH_SOURCE[0]}"`"

# Download toolboxes if necessary
if [ ! -f toolboxes.zip ]; then
    echo "Downloading toolboxes..."
    wget -O toolboxes.zip $TOOLBOX_URL
fi

# Unzip toolboxes if necessary
if [ ! -d toolboxes ]; then
    echo "Excracting toolboxes..."
    mkdir toolboxes
    cd toolboxes
    unzip ../toolboxes.zip
    cd ..
fi

if [ -f verification.mat ]; then
    rm verification.mat
fi

echo "Generating verification data in MATLAB..."
"$MATLAB" -nosplash -nodesktop -r "gen_verif; quit"

if [ ! -f verification.mat ]; then
    echo "error: no output from MATLAB"
    exit 1
fi

echo "Converting to NumPy format..."
python verif_m_to_npz.py

echo "Done"

