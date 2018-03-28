function [ nlayer ] = layer(layer, filters, options)
nlayer = struct();
nlayer.nodes = {options.J * length(layer.nodes)};
for n = 1:length(layer.nodes)
    for m = 1:length(layer.nodes{n}.U)
        x = layer.nodes{n}.U{m};
        j_history = layer.nodes{n}.j{m};
        nlayer.nodes{(n-1)*options.J + m} = node(x, j_history, filters);
    end
end
end

