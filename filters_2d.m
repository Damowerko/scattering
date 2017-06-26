function [ filters ] = filters_2d( M, N, options )

step = 2*pi/options.L;
thetas = 0:step:2*pi-step;

%psi
psi = zeros(M,N,options.J,options.L);
for j = 0:options.J-1
    sigma_psi = options.sigma_psi * 2^j;
    xi_psi = options.xi_psi / 2^j;
    
    for l = 1:options.L
        % generate spacial filter and take the fft
        temp = morlet_2d2(M, N, sigma_psi, 4/options.L, xi_psi, thetas(l));
        temp = real(fft2(temp));
        psi(:,:, j+1, l) = temp; 
    end
end
filters.psi = real(psi);

%phi
sigma_phi = options.sigma_phi * 2^(options.J-1);
filters.phi = real(fft2(gaussian_2d(M,N,sigma_phi)));
end

