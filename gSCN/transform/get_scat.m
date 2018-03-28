function [scat] = get_scat(layers)
scat = [];
for layer = layers
    for node = layer{1}.nodes
        scat = [scat; node{1}.S];
    end
end
end

