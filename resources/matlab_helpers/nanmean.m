function y = nanmean(x, dim)
%NANMEAN Minimal fallback for legacy StaMPS calls when NANMEAN is unavailable.
%   Y = NANMEAN(X) returns the mean of X over the first non-singleton
%   dimension, omitting NaN values.
%
%   Y = NANMEAN(X, DIM) returns the mean over DIM, omitting NaN values.
%
%   This project-scoped fallback exists so StaMPS Step 7 can run in MATLAB
%   environments where legacy NANMEAN is not installed on disk.

if nargin < 1
    error('nanmean requires at least one input argument.');
end

if nargin < 2 || isempty(dim)
    y = mean(x, 'omitnan');
    return;
end

validateattributes(dim, {'numeric'}, {'scalar', 'real', 'finite', 'integer', 'positive'}, mfilename, 'DIM', 2);
y = mean(x, dim, 'omitnan');
