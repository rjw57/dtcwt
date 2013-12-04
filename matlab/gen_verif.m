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

g0a = qshift_d.g0a;
g0b = flipud(g0a);

lena_colifilt = colifilt(lena, g0b, g0a);

[lena_Yl, lena_Yh, lena_Yscale] = dtwavexfm2(lena, 4, 'near_sym_a', 'qshift_a');

near_sym_b_bp = load('near_sym_b_bp');
qshift_b_bp = load('qshift_b_bp');

[lena_Ylb, lena_Yhb, lena_Yscaleb] = dtwavexfm2b(lena, 4, 'near_sym_b_bp', 'qshift_b_bp');

%% 3x interpolation of subbands
x = lena_Yh{3};
sx = size(x);
nx = sx(2)*3; ny = sx(1)*3;
scx = sx(2) / nx;
scy = sx(1) / ny;
[X, Y] = meshgrid(0.5 + ((1:nx)-0.5)*scx, 0.5 + ((1:ny)-0.5)*scy);
locs = [Y(:), X(:)];
w = [-3 -1; -3 -3; -1 -3; 1 -3; 3 -3; 3 -1]*pi/2.15;
lena_upsample = zeros(ny, nx, 6);
for sb=1:6
    tmp = zeros(ny, nx);
    tmp(:) = cpxinterp2b(x(:,:,sb),locs,w(sb,:), 'linear');
    lena_upsample(:,:,sb) = tmp;
end

save('verification.mat', 'lena_coldfilt', 'lena_colifilt', 'lena_Yl', 'lena_Yh', 'lena_Yscale', 'lena_Ylb', 'lena_Yhb', 'lena_Yscaleb', 'lena_upsample');
