function [ f ] = gaussian_filter_graph(frequencies, sigma)
extent = 1;         % extent of periodization - the higher, the better
sigma = 1/sigma;
f = zeros(length(frequencies),1);
% Calculate the 2*pi-periodization of the filter over 0 to 2*pi*(N-1)/N
N = length(frequencies);
n = map_lambdas(frequencies, 0, N-1);
for k = -extent:1+extent
	f = f+exp(-(([0:N-1].'-k*N)/N*2*pi).^2./(2*sigma^2));
end
end
