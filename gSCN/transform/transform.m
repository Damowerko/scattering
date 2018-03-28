function [scat,layers] = transform(x, filters, options)
%% Apply the filters
x = reshape(x, [length(x) 1]); % ensure x is column vector
layers = {options.M+1};
% first layer
layers{1} = struct();
% This creates all the points in the tree of transforms to apply.
layers{1}.nodes{1} = node(x, -1, filters, options);

for m = 1:options.M
    layers{m+1} = layer(layers{m}, filters, options);
end
scat = get_scat(layers);
end