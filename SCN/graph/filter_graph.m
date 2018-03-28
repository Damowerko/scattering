function [filters] = filter_graph(S, options)
N = length(S);
% get eigenvectors + eigenvalues and sort them lowest frequency first
[V, E] = eig(full(S));
if strcmpi(options.graph_shift, 'laplacian')
    [E, I] = sort(diag(E), 'ascend');
    frequencies = E;
    E = diag(E);
elseif strcmpi(options.graph_shift, 'covariance')
    max_e = max(diag(E));
    [frequencies, I] = sort(max_e-diag(E), 'ascend');
    E = diag(max_e - frequencies);
%    frequencies = log(frequencies+0.1);
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


freq_num = ceil(length(E) * (2*ones(1,options.J)).^(-[1:options.J]));
bw = frequencies(freq_num);

S = zeros(N,1); % use to calculate normalization factor
hf = zeros(N,options.J);

for j = 0:options.J-1
    if options.lambda_scale
        hf(:,j+1) = morlet_1d_graph(frequencies, options.psi.sigma, j)';
	%hf(:,j+1) = haar_filter(N, bw(j+1));
    else
        hf(:,j+1) = morlet_1d_freq(N, options.psi.sigma, j)';
    end

    S = S + abs(hf(:,j+1)).^2;
    % normalize filter
    %hf = hf / norm(hf,1);
    %filters.psi{j+1} = freq_filter(hf, V, E);
end

psi_factor = sqrt(2/max(S));
hf = hf * psi_factor;

for j = 0:options.J-1
    filters.psi{j+1} = freq_filter(hf(:,j+1), V, E);
end

if options.lambda_scale
    hf = gaussian_filter_graph(frequencies, options.psi.sigma);
else
    hf = gaussian_filter_freq(N, options.phi.sigma)';
end
%hf = hf/norm(hf,1); % normalize

filters.phi = freq_filter(hf, V, E);
end
