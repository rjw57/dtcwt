function h = gaussian(sigma,thresh)

% function h = gaussian(sigma,thresh)
%
% Generate a gaussian vector / impulse response, h, with
% a standard deviation of sigma samples and a total value of unity.
% The length of h is odd and is truncated at the points where
% exp(-x^2/2) < thresh. By default, thresh = 0.01.
%
% Nick Kingsbury, Cambridge University, Nov 2005.

if nargin < 2, thresh = 0.01; end

% Solve for when exp(-x^2 / 2*sigma^2) = thresh
xmax = sigma * sqrt(max(-2*log(thresh),1e-6));

% Calculate h over the range when exp(-x^2/2) >= thresh.
n = floor(xmax);
x = [-n:n]';
h = exp(x.*x/(-2*sigma*sigma));

% Normalise h so it sums to unity.
h = h / sum(h);

return;


