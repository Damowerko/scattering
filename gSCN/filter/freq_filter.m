function [ H ] = freq_filter(hf, V, E)
%GRAPH_FILTER Greates a graph filter by using the frequency space.
%   Using the frequency response hf of the filter
%   the function generates a spacial graph filter
%   by operating in the frequency space.

if ~isreal(E)
    error('Eigenvalues must be real')
end

N = length(E);
if length(hf) < N
    hf(N) = 0; % ensure frequency response is the appropriate length by zero pad
end

% ensure eigenvaleus are in matrix form
if size(E,1) ~= size(E, 2)
    lambdas = E;
    E = diag(lambdas);
end

% ensure the eigenvectors and eigenvalues are ordered
[~, I] = sort(diag(E), 'ascend');
V = V(:,I);

% use the GFT to implement the filter
H = V * diag(hf(:)) * V';
end
