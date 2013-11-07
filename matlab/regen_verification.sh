#!/bin/bash

## CONFIGURATION

# Path to dtcwt toolbox
DTCWT_TOOLBOX=$HOME/Downloads/dtcwt_toolbox4_3

# Path to dtcwt keypoints toolbox
DTCWT_KEYPOINTS=$HOME/Downloads/DTCWTkeypoints

# Path to MATLAB
MATLAB=/opt/MATLAB/R2013b/bin/matlab

## END OF CONFIGURATION

# Update MATLAB path
export MATLABPATH="$MATLABPATH:$DTCWT_TOOLBOX:$DTCWT_KEYPOINTS"

# Change to this directory
cd "`dirname "${BASH_SOURCE[0]}"`"

echo "Generating verification data in MATLAB..."
"$MATLAB" -nosplash -nodesktop -r "gen_verif; quit"

echo "Converting to NumPy format..."
python verif_m_to_npz.py

echo "Done"

