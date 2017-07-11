function [filters] = filter_graph(A, options)

G = graph(A);
L = laplacian(G);

[V, E] = eig(full(L));
%[E, I] = sort(diag(E), 'ascend');
%V = V(:,I);
%E = diag(E);
emax = max(real(diag(E)));
[E,I] = sort(abs(emax-diag(E)),'ascend');
E = diag(E);
V = V(:,I);

% filters
filters = struct();
filters.psi = {options.J};
filters.phi = [];

disp('Generating psi');
for j = 0:options.J-1
    hf = morlet_1d_freq(length(A), options.psi.sigma, j)';
    filters.psi{j+1} = freq_filter(hf, V, E);
end
disp('Generating phi');
hf = gaussian_filter_freq(length(A), options.phi.sigma)';
filters.phi = freq_filter(hf, V, E);

end
