function [scat] = transform_graph(x, filters, options)

addpath '..'
%% Apply the filters
clear scat
scat{1} = struct();
scat{1}.data{1} = layer_data_graph([], x, [-1]); %#ok<NBRAK>

for m = 1:options.M+2
    scat = layer_graph(scat, filters, options);
end
scat = scat(1:options.M+1);








