function [ F ] = fourier( N )
%FOURIER Create an NxN fourier basis matrix.
F = zeros(N);
for n = 1:N
    [~,v] = cexp(n-1,N);
    F(:,n) = v / norm(v);
end
end

