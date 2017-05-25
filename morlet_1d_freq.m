function [ f ] = morlet_1d_freq(N, sigma, j)
% Generates a morlet wavelet in frequency space between 0 and 2pi of length
% N.
step = 2*pi / N;
w = 0:step:2*pi - step;
w = w .* 2^j;
nu = sqrt(1 + exp(-sigma^2) - 2*exp(-3/4*sigma^2));
f = nu * pi^(-1/4)*( exp(-1/2*(sigma-w).^2) - exp(-1/2*(w.^2+sigma^2)) );
end
