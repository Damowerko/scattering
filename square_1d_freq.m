function [ f ] = square_1d_freq(N, sigma, j)
xi = 3/4 * pi;
max = xi + sigma;
min = xi - sigma;



step = 2*pi / N;
w = 0:step:2*pi - step;
w = w .* 2^j;

disp(w)

f = zeros(N,1);
f((min<w)&(w<max)) = 1;
f = f/norm(f);
end
