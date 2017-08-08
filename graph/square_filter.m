function [ f ] = square_filter(freq, j)
% Generates a scaled haar filter.
thres = max(freq) * 2^(-j-1);
f = freq < thres;
end
