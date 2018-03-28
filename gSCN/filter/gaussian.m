function [ f ] = gaussian(w, sigma, J)
w = 2^J * w;
f = normpdf(w, 0, 1/sigma);
f = f / norm(f,1);
end