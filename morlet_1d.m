function [ x ] = morlet_1d(N, sigma, xi)
beta = exp(-xi^2*sigma^2/2);
n = -N/2+1:N/2;
x = exp(-(n).^2/(2*sigma^2)) .* (exp(-i*(n)*xi)-beta);
x = x / norm(x);
end