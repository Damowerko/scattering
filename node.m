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

% filters psi
psi = zeros(N, options.J);
for j = 0:options.J-1
    xi = options.psi.xi;
    sigma = options.psi.sigma*2^(j);
    psi(:,j+1) = 2^(-2*options.J)*morlet_1d(N, sigma, xi);
end

% perform convolutions
U = zeros(N, size(psi,2));
for i = 1:size(psi,2)
    U(:,i) = abs(conv(x, psi(:,i),'same'));
end

% low-pass filters phi
phi = zeros(N,options.J);
for j = 0:options.J-1
    sigma = options.phi.sigma * 2^options.J;
    phi(:,j+1) = 2^(-2*options.J)*gaussian_filter(N, sigma);
end

% perform convolution
S = zeros(N, size(U,2));
for i = 1:size(U,2)
    S(:,i) = conv(U(:,i),phi(:,i),'same');
end
