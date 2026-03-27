function y = interp(x, q)
%INTERP Minimal fallback for legacy StaMPS calls when toolbox INTERP is unavailable.
%   Y = INTERP(X, Q) upsamples the vector or matrix X by the integer factor
%   Q using linear interpolation between samples.
%
%   StaMPS uses the legacy toolbox INTERP in PS_EST_GAMMA_QUICK via:
%       interp([1, Prand], 10)
%   and then trims the final Q-1 samples. This fallback preserves that
%   contract by returning Q * size(X, 1) samples and keeping original
%   samples at indices 1:Q:end.
%
%   The implementation intentionally relies only on core MATLAB INTERP1 so
%   the project can run in MATLAB environments where the toolbox-provided
%   INTERP function is not installed on disk.

if nargin < 2
    error('interp requires two input arguments.');
end

if nargin > 2
    error('This project fallback only supports interp(X, Q).');
end

validateattributes(q, {'numeric'}, {'scalar', 'real', 'finite', 'integer', 'positive'}, mfilename, 'Q', 2);

if isempty(x)
    y = x;
    return;
end

input_was_row = isrow(x);
if input_was_row
    x = x.';
end

validateattributes(x, {'numeric'}, {'2d', 'real', 'finite', 'nonnan'}, mfilename, 'X', 1);

n = size(x, 1);

if n == 1
    y = repmat(x, q, 1);
else
    xi = (1:n).';
    xq = (1:1 / q:n).';
    y = interp1(xi, x, xq, 'linear');
    y = [y; repmat(x(end, :), q - 1, 1)];
end

if input_was_row
    y = y.';
end
