function [ f ] = haar_filter(N, M)
% Generates a scaled haar filter.
N = ceil(N);
M = ceil(M);
f = [ones(1,ceil(M/2)) -1*ones(1, floor(M/2)) zeros(1,N-M)];
f = fft(f);
end
