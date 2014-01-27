% gen_qbgn.m
% Generate and display a 3-D fruitcake-like volume.
%
% Nick Kingsbury, Cambridge University, Nov 2011.
function bg=gen_qbgn(N,F,seed)
    if nargin<3
        seed=0;
    end    
    Q=4;
    h = gaussian(3,0.01);
    h = gaussian(1.5,0.01);
    h = gaussian(8,0.01);
    y = zeros(N,N,F);
    RandStream.setGlobalStream(RandStream('mt19937ar', 'seed', seed));
    for f=1:F,
        x=randn(N,N);
        
        y(:,:,f) = colfilter(colfilter(x,h).',h).';
    end
    for k=1:N,
        y(k,:,:) = colfilter(squeeze(y(k,:,:)).',h).';
    end
    yvar = var(y(:));
    y = y*(16/sqrt(yvar)) + 128;  
    t1=ceil(min(y(:)));
    t2=floor(max(y(:)));

    step=(t2-t1)/50;
    z = double(softquant(y,[t1 t2 step]));
    bg=z;
end