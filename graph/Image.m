%% Image generation
addpath '..'

x = zeros(64);
x(1:32, 1:32) = ones;
[M, N] = size(x);
x = x(:);

G = graph(image_adj(M,N));
L = laplacian(G);

[V, E] = eigs(L, length(L));
[E, I] = sort(diag(E), 'ascend');
V = V(:,I);
E = diag(E);

%% Filters

options.J = 3;
options.M = 2;
options.psi.sigma = 0.85;
options.psi.xi = 3/4*pi;
options.phi.sigma = 0.85;


filters = struct();
filters.psi = {options.J};
filters.phi = [];

for j = 0:options.J-1
    temp = morlet_1d_freq(N*M, options.psi.sigma, j)';
    filters.psi{j+1} = poly_filter(temp, V, E);
end
temp = gaussian_filter_freq(N*M, options.phi.sigma)';
filters.phi = poly_filter(temp, V, E);
clear temp;

%% Apply the filter

scat{1} = struct();
scat{1}.data{1} = layer_data_graph([], x, []);

for m = 1:options.M+1
    scat = layer_graph(scat, filters);
end
scat = scat(1:options.M);









