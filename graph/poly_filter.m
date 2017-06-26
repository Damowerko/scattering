function [ H ] = poly_filter(hf, V, E)
%GRAPH_FILTER Greates a graph polynomial filter matrix.
%   Using the frequency response f of the filter
%   the function generates a graph filter using the powers of the
%   graph shift operator with eigenvectors V with eigenvalue matrix E.

if ~isreal(E)
    error('Eigenvalues must be real')
end

N = length(E);
if length(hf) < N
    hf(N) = 0;
end

if size(E,1) == size(E, 2)
    lambdas = diag(E);
else
    lambdas = E;
    E = diag(lambdas);
end

[E, I] = sort(diag(E), 'ascend');
V = V(:,I);
E = diag(E);

S = V*E*V';

W = fliplr(vander(lambdas)); %vandermode matrix
h = W\hf; %filter coefficients
H = zeros(N,N);
for n = 1:N
    H = H + h(n) * S^n;
end
end