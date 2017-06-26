function [ output_args ] = morlet_2d( M, N, sigma )
   
    [x , y] = meshgrid(1:M, 1:N);
	
	x = x - ceil(M/2) - 1;
	y = y - ceil(N/2) - 1;
	
	Rth = rotation_matrix_2d(theta);
	A = Rth\ [1/sigma^2, 0 ; 0 slant^2/sigma^2] * Rth ;
	s = x.* ( A(1,1)*x + A(1,2)*y) + y.*(A(2,1)*x + A(2,2)*y ) ;
	
	%normalize sucht that the maximum of fourier modulus is 1
	
	gaussian_envelope = exp( - s/2);
	oscilating_part = gaussian_envelope .* exp(1i*(x*xi*cos(theta) + y*xi*sin(theta)));
	K = sum(oscilating_part(:)) ./ sum(gaussian_envelope(:));
	gabc = oscilating_part - K.*gaussian_envelope;
	
	gab=1/(2*pi*sigma^2/slant^2)*fftshift(gabc);

end

