function [ S, U ] = node( x, options )
% defaults
if ~isfield(options.psi, 'sigma')
    options.psi.sigma = 0.85;
end
if ~isfield(options.psi, 'xi')
    options.phi.xi = 3/4 * pi;
end
if ~isfield(options.phi, 'sigma')
    options.phi.sigma = 0.85;
end

N = length(x);
N_filt = 2^ceil(log2(N));

% filters psi
psi = zeros(N_filt, options.J);
for j = 0:options.J-1
    xi_psi = options.psi.xi * 2^(-j);
    sigma_psi = options.psi.sigma * 2^j;
    filt = 2^(-2*options.J)*morlet_1d_freq(N_filt, sigma_psi, xi_psi);
    if length(filt) < N_filt
        filt(N_filt) = 0;
    end
    psi(:,j+1) = filt;
end

% apply filters and modulus
U = zeros(N, size(psi,2));
for i = 1:size(psi,2)
    % fft of signal
    X = fft(x);
    % pad frequency signal to the length of filter
    X = pad_signal(X, N_filt);
    Y = psi(:,i) .* X; 
    ds = round(log2(2*sigma_psi/options.psi.sigma));
    ds = max(ds, 0);
    Y = unpad_signal(Y, ds, N);
    U(:,i) = abs(ifft(Y));
end

% low-pass filter phi
sigma_phi = options.phi.sigma * 2^(options.J-1);
phi = 2^(-2*options.J)*gaussian_filter_freq(N_filt, sigma_phi);
ds = round(log2(4*sigma_phi/options.phi.sigma));
ds = max(ds, 0);

% apply low pass filter
S = zeros(N, size(U,2));
for i = 1:size(U,2)
    X = fft(U(:,i));
    X = pad_signal(X, N_filt);
    Y = X .* phi;
    Y = unpad_signal(Y, ds, N);
    S(:,i) = real(ifft(Y));
end
