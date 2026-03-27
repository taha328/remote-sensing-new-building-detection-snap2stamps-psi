function w = gausswin(L, alpha)
%GAUSSWIN Minimal fallback for StaMPS when Signal Processing Toolbox is unavailable.
%   W = GAUSSWIN(L) returns an L-point Gaussian window with the MATLAB
%   default alpha of 2.5.
%
%   W = GAUSSWIN(L, ALPHA) matches the standard Gaussian window definition
%   used by MATLAB/Octave:
%       w(n) = exp(-0.5 * (alpha * n / ((L - 1) / 2)).^2)
%
%   This project-scoped fallback exists so StaMPS functions such as
%   CLAP_FILT can run in MATLAB environments where Signal Processing
%   Toolbox functions are not installed on disk.

if nargin < 1
    error('gausswin requires at least one input argument.');
end

if nargin < 2 || isempty(alpha)
    alpha = 2.5;
end

validateattributes(L, {'numeric'}, {'scalar', 'real', 'finite', 'integer', 'nonnegative'}, mfilename, 'L', 1);
validateattributes(alpha, {'numeric'}, {'scalar', 'real', 'finite', 'nonnan'}, mfilename, 'alpha', 2);

if L == 0
    w = zeros(0, 1);
    return;
end

if L == 1
    w = 1;
    return;
end

n = -(L - 1) / 2 : (L - 1) / 2;
scale = (L - 1) / 2;
w = exp(-0.5 * (alpha * n / scale) .^ 2).';
