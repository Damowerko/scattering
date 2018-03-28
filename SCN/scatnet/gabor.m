function f = gabor(N,xi,sigma,precision)
	extent = 1;         % extent of periodization - the higher, the better
	
	sigma = 1/sigma;
	
	f = zeros(N,1,precision);
	
	% Calculate the 2*pi-periodization of the filter over 0 to 2*pi*(N-1)/N
	for k = -extent:1+extent
		f = f+exp(-(([0:N-1].'-k*N)/N*2*pi-xi).^2./(2*sigma^2));
	end
end
