function [ f ] = morlet_1d_graph(frequencies, sigma, j, flip)
% Generates a morlet wavelet in frequency space between 0 and 2pi of length
% N.
N = length(frequencies);
w = map_lambdas(frequencies, 0, 2*pi*(N-1)/N, flip);
w = w .* 2^j;
nu = sqrt(1 + exp(-sigma^2) - 2*exp(-3/4*sigma^2));
f = nu * pi^(-1/4)*( exp(-1/2*(sigma-w).^2) - exp(-1/2*(w.^2+sigma^2)) );
end
