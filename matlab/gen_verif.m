%% Generate verification arrays to compare known good DT-CWT inputs and outputs.
%%
%% This script assumes that Nick Kingsbury's DT-CWT toolbox has been installed.
%%
%% Run with something like:
%%
%% $ /path/to/matlab -nosplash -nodesktop -r "run /path/to/gen_verif; quit"
%%
%% There should be the DTCWT toolboxes installed in a toolboxes directory next
%% to the script. See the regen_verification.sh script for an example of setting
%% this up.

% Add the qbgn and toolbox files to the path
strFilePath=[fileparts(which(mfilename('fullpath'))) '/'];
addpath([strFilePath 'qbgn/']);
addpath(genpath([strFilePath 'toolboxes/']));

%% Load mandrill image
inputs = load('mandrill.mat');
mandrill = inputs.mandrill;

near_sym_b = load('near_sym_b');
qshift_d = load('qshift_d');

h1a = qshift_d.h1a;
h1b = flipud(h1a);

mandrill_coldfilt = coldfilt(mandrill, h1b, h1a);

g0a = qshift_d.g0a;
g0b = flipud(g0a);

mandrill_colifilt = colifilt(mandrill, g0b, g0a);

[mandrill_Yl, mandrill_Yh, mandrill_Yscale] = dtwavexfm2(mandrill, 4, 'near_sym_a', 'qshift_a');

near_sym_b_bp = load('near_sym_b_bp');
qshift_b_bp = load('qshift_b_bp');

[mandrill_Ylb, mandrill_Yhb, mandrill_Yscaleb] = dtwavexfm2b(mandrill, 4, 'near_sym_b_bp', 'qshift_b_bp');

%% 3x interpolation of highpasses
x = mandrill_Yh{3};
sx = size(x);
nx = sx(2)*3; ny = sx(1)*3;
scx = sx(2) / nx;
scy = sx(1) / ny;
[X, Y] = meshgrid(0.5 + ((1:nx)-0.5)*scx, 0.5 + ((1:ny)-0.5)*scy);
locs = [Y(:), X(:)];
w = [-3 -1; -3 -3; -1 -3; 1 -3; 3 -3; 3 -1]*pi/2.15;
mandrill_upsample = zeros(ny, nx, 6);
for sb=1:6
    tmp = zeros(ny, nx);
    tmp(:) = cpxinterp2b(x(:,:,sb),locs,w(sb,:), 'linear');
    mandrill_upsample(:,:,sb) = tmp;
end

%% Generate quantized bandlimited gaussian noise (gbgn) phantom
%generate quantized band-limited gaussian noise, and case to 8-bit to save space
qbgn = uint8(gen_qbgn(128,128));
%take the 3D wavelet transform, which defaults to near_sym_a (5,7) and qshift_b (14 taps)
[qbgn_Yl, qbgn_Yh, qbgn_Yscale] = dtwavexfm3(double(qbgn), 3);
%now re-arrange the coefficients to form complex-valued high-pass subbands instead of alternating real/imag parts
qbgn_Yh = ri2c(qbgn_Yh);

save('qbgn.mat','qbgn');

save('verification.mat', 'mandrill_coldfilt', 'mandrill_colifilt', 'mandrill_Yl', 'mandrill_Yh', 'mandrill_Yscale', ...
     'mandrill_Ylb', 'mandrill_Yhb', 'mandrill_Yscaleb', 'mandrill_upsample', 'qbgn_Yl', 'qbgn_Yh', 'qbgn_Yscale');
