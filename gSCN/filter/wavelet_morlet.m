function [ f ] = wavelet_morlet(w, sigma, xi, j)
w = w * 2^(j); % rescale: 2^j w

a = pi^(-0.25) * (1+exp(-(sigma*xi)^2)-2*exp(-0.75*(sigma*xi)^2))^(-0.5);
b = exp(-0.5*xi^2*sigma^2);

f = a * (exp(-0.5*(sigma*xi-w).^2) - b*exp(-0.5*w.^2));
end
