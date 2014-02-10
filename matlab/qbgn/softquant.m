function y = softquant(x,q)
% function y = softquant(x,q)
%
% Soft quantise x, using quantiser thresholds specified by q as follows:
%   q = [qmin  qmax  stepsize];
% As x increases linearly, y will smoothly transition
% between adjacent quantiser steps around the nominal transition point.
%
% Nick Kingsbury, Cambridge University, Nov 2011.

% Quantise x.

% Apply an offset and a scale factor to x so that y goes from zero to 2*pi*ymax
% and the stepsize is 2*pi.
ymax = round((q(2) - q(1))/q(3));  % no. of steps.
y = (x - q(1)) * (1/q(3));
y = (2*pi) * max(min((x - q(1)) * (1/q(3)),ymax),0);

% Apply the non-linear periodic function that is periodic over 2*pi, such
% that it generates smooth steps and has approx zero gradient in between
% the transition regions.
y = y - 1.4*sin(y) + 0.2*sin(2*y);

% Rescale y back to the original range of x and reinsert the offset.
y = y * (q(3)*0.5/pi) + q(1);
return;




