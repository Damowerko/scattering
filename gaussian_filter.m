function [ x ] = gaussian_filter(N, sigma)
n = -N/2+1:N/2;
x = exp(-(n).^2/(2*sigma^2));
x = x / norm(x);
end