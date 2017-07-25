function [ scat ] = layer_graph(scat, filters, options)
%LAYER_GRAPH
m = length(scat);
scat{m+1} = struct();
scat{m+1}.data = {};

for n = 1:length(scat{m}.data)
    % apply phi
    x = scat{m}.data{n}.U;
    temp = filters.phi * x;
    if options.subsample
        temp = reshape(temp, [28 28]);
        temp = temp(1:2^options.J:end, 1:2^options.J:end);
        temp = temp(:);
    end
    scat{m}.data{n}.S = temp;
    % apply psi
    for j = 0:length(filters.psi)-1
        j1 = scat{m}.data{n}.j;
        if j > j1(end)
            S = [];
            U = abs(filters.psi{j+1} * x);
            j2 = [j1 j];
            data = layer_data_graph(S, U, j2);
            scat{m+1}.data = [scat{m+1}.data data];
        end
    end
end
end

