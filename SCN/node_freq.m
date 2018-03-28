function [ S, U ] = node( x, options )
% defaults
if ~isfield(options.psi, 'sigma')
    options.psi.sigma = 0.85;
end
if ~isfield(options.psi, 'xi')
    options.psi.xi = 3/4 * pi;
end
if ~isfield(options.phi, 'sigma')
    options.phi.sigma = 0.85;
end

N = length(x);
N_filt = 2^ceil(log2(N));
if(N < N_filt)
        x(N_filt) = 0;
end

% filters psi
psi = zeros(N_filt, options.J);
for j = 0:options.J-1
    filt = morlet_1d_freq(N_filt, options.psi.sigma, j);
    psi(:,j+1) = filt;
end

% apply filters and modulus
U = zeros(N, size(psi,2));
for i = 1:size(psi,2)
    X = fft(x);
    Y = psi(:,i) .* X;
    y = ifft(Y);
    U(:,i) = abs(y(1:N));
end

% low-pass filter phi
sigma_phi = options.phi.sigma * 2^(options.J-1);
phi = gaussian_filter_freq(N_filt, sigma_phi);

% apply low pass filter
S = zeros(N, size(U,2));
for i = 1:size(U,2)
    x = U(:,i);
    if(N < N_filt)
        x(N_filt) = 0;
    end
    X = fft(x);
    Y = X .* phi;
    %Y = downsample(Y, 2^options.J);
    y = ifft(Y);
    S(:,i) = real(y(1:N));
end
