function [filters, h] = filter_graph(A, psi, phi, options, varargin)
if nargin > 4
    k_matrix = varargin{1}; % matrix for vector signals
else
    k_matrix = 1;
end
N = length(A);
h = struct();
h.psi = {};
h.phi = [];
h.frequencies = [];
% get eigenvectors + eigenvalues and sort them lowest frequency first

if strcmpi(options.filter.operator, 'L')
    S = laplacian(graph(A));
    [V, E] = eig(full(S));
    [E, I] = sort(diag(E), 'ascend');
    frequencies = E;
    E = diag(E);
    V = V(:,I); % sort eigenvectors
elseif strcmpi(options.filter.operator, 'L_norm')
    L = laplacian(graph(A));
    D = diag(degree(graph(A)));
    S = D^(-1/2)*L*D^(-1/2);
    [V, E] = eig(full(S));
    [E, I] = sort(diag(E), 'ascend');
    frequencies = E;
    E = diag(E);
    V = V(:,I); % sort eigenvectors
elseif strcmpi(options.filter.operator, 'A')
    % sort complex eigenvalues
    S = A;  
    [V, E] = eig(full(S));
    emax = max(real(diag(E)));
    [~,I] = sort(abs(emax-diag(E)),'ascend');
    E = diag(E);
    E = diag(E(I));
    V = V(:,I); % sort eigenvectors
    frequencies = diag(E);
end

if options.filter.lambda_scale
    h.frequencies = frequencies;
else
    h.frequencies = 1:N;
end

if options.filter.log_scale > 0 % TODO add error for negative values
    h.frequencies = map_lambdas(h.frequencies, 0, 2*pi);
    h.frequencies = log(h.frequencies*options.filter.log_scale+0.00000001);
end
if ~isnan(options.filter.exp_scale)
    h.frequencies = exp(h.frequencies*options.filter.exp_scale);
end
if options.filter.map2pi
    h.frequencies = map_lambdas(h.frequencies, 0, 2*pi);
end

% filter creation
filters = struct();
filters.psi = {}; % wavelet
filters.phi = []; % averaging filter

norm_factor_S = zeros(N,1); % normalization factor used in max2 norm
%% PSI
% build the frequency response of psi
psi_hf = zeros(N,options.J);
for j = 0:options.J-1
    psi_hf(:,j+1) = psi(h.frequencies, options.filter.psi_sigma, options.filter.psi_xi, j)';

    norm_factor_S = norm_factor_S + abs(psi_hf(:,j+1)).^2;
end

if strcmpi(options.filter.normalization, 'l1')
    for j = 0:options.J-1
        %normalize filter
        psi_hf(:,j+1) = psi_hf(:,j+1) / norm(psi_hf(:,j+1),1);
    end
elseif strcmpi(options.filter.normalization, 'max2')
    psi_factor = sqrt(2/max(norm_factor_S)); % Frequency normalization done
        % in code, but not in paper. Keep a contractive transform.
    psi_hf = psi_hf * psi_factor;
elseif strcmpi(options.filter.normalization, 'none')
    
elseif ~strcmpi(options.filter.normalization, 'default')
    error('No such parameter: filter.%s', options.filter.normalization)
end

% convert frequency response to graph filter
for j = 0:options.J-1
    h.psi{j+1} = psi_hf(:,j+1);
    filters.psi{j+1} = freq_filter(psi_hf(:,j+1), V, E);
    filters.psi{j+1} = kron(k_matrix, filters.psi{j+1});
end

%% PHI
phi_hf = phi(h.frequencies, options.filter.phi_sigma, j);
h.phi = phi_hf;
filters.phi = freq_filter(phi_hf, V, E);
filters.phi = kron(k_matrix, filters.phi);
end
