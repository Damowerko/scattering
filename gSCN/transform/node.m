function [ node ] = node(x, j_history, filters, options)
% Apply phi to obtain scattering vector
if ~isempty(x)
    s = filters.phi * x; % H_phi x (H_phi = V * phi(freq) * V')
else
    s = [];
end
    
% Apply psi to obtain U
u = {};
j_current = {};
for j = 0:length(filters.psi)-1
    if j > j_history(end) && ~isempty(x)
        u{j+1} = abs(filters.psi{j+1} * x);
        j_current{j+1} = [j_history; j];
    else
        u{j+1} = [];
        j_current{j+1} = [j_history; j];
    end
end
node = node_data(s, u, j_current);
end

