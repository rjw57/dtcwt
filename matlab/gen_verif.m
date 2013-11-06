%% Generate verification arrays to compare known good DT-CWT inputs and outputs.
%%
%% This script assumes that Nick Kingsbury's DT-CWT toolbox has been installed.
%%
%% Run with something like:
%%
%% $ MATLABPATH=/path/to/dtcwt_toolbox4_3 /path/to/matlab -nosplash -nodesktop -r "run /path/to/gen_verif; quit"

%% Load Lena image
inputs = load('lena.mat');
lena = inputs.lena;

near_sym_b = load('near_sym_b');
qshift_d = load('qshift_d');

h1a = qshift_d.h1a;
h1b = flipud(h1a);

lena_coldfilt = coldfilt(lena, h1b, h1a);

[lena_Yl, lena_Yh, lena_Yscale] = dtwavexfm2(lena, 4, 'near_sym_a', 'qshift_a');

save('verification.mat', 'lena_coldfilt', 'lena_Yl', 'lena_Yh', 'lena_Yscale');

%% 
