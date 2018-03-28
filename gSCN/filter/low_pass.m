function [ f ] = low_pass(frequencies, sigma, J)
w = 2^J * w;
f = double(w < 1/sigma);
end