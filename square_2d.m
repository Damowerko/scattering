function filt = square_2d(M, N, sigma, xi)

	[x , y] = meshgrid(1:N, 1:M);
	
	x = x - ceil(N/2) - 1;
	y = y - ceil(M/2) - 1;
	r = sqrt(x.^2 + y.^2);
    
    filt = zeros(M,N); 
    filt((xi-sigma<r)&(r<xi+sigma)) = 1;
    
    
	
end