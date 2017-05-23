function [ f ] = morlet_1d_freq(N, xi, sigma)
% Generates a morlet wavelet in frequency space between 0 and 2pi of length
% N.
    f = gabor(N, xi, sigma);
    f = morletify(f, sigma);
end

function f = gabor(N,xi,sigma)
	extent = 1;         % extent of periodization - the higher, the better
	sigma = 1/sigma;
	f = zeros(N,1);
	% Calculate the 2*pi-periodization of the filter over 0 to 2*pi*(N-1)/N
	for k = -extent:1+extent
		f = f+exp(-( ( [0:N-1].'-k*N )/N*2*pi-xi ).^2./(2*sigma^2));
	end
end

function f = morletify(f,sigma)
	f0 = f(1);
	f = f-f0*gabor(length(f),0,sigma);
end
