function [filters] = filter_graph(A, options)

G = graph(A);

switch options.graph_shift
    case 'adjacency'
        S = adjacency(G);
    case 'laplacian'
        S = laplacian(G);
    case 'covariance'
        load('cov');
        S = C;
    otherwise
        error('Invalid graph_shift option.')
end

% get eigenvectors + eigenvalues and sort them lowest frequency first
[V, E] = eig(full(S));
if strcmpi(options.graph_shift, 'laplacian') | strcmpi(options.graph_shift, 'covariance')
    [E, I] = sort(diag(E), 'ascend');
    frequencies = E;
    E = diag(E);
elseif strcmpi(options.graph_shift, 'adjacency')
    % sort complex eigenvalues
    emax = max(real(diag(E)));
    [frequencies,I] = sort(abs(emax-diag(E)),'ascend');
    E = diag(E);
    E = diag(E(I));
end
V = V(:,I); % sort eigenvectors

% filters
filters = struct();
filters.psi = {options.J};
filters.phi = [];

for j = 0:options.J-1
    if options.lambda_scale
        hf = morlet_1d_freq(length(A), options.psi.sigma, j)';
    else
        hf = morlet_1d_graph(frequencies, options.psi.sigma, j)';
    end
    filters.psi{j+1} = freq_filter(hf, V, E);
end

if options.lambda_scale
    hf = gaussian_filter_freq(length(A), options.phi.sigma)';
else
    hf = gaussian_filter_graph(frequencies, options.psi.sigma)
end
filters.phi = freq_filter(hf, V, E);

end
