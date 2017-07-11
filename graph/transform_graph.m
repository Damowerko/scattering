function [scat] = transform_graph(x, filters, options)

addpath '..'
[M, N] = size(x);

%% Apply the filters
clear scat
scat{1} = struct();
scat{1}.data{1} = layer_data_graph([], x, [-1]);

for m = 1:options.M+2
    scat = layer_graph(scat, filters);
end
scat = scat(1:options.M+1);








