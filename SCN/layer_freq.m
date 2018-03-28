function [ Sp1, Up1 ] = layer_freq( U, options )
%size_scale = 2^ceil(log2(size(U,1))) / size(U,1);
%Sp1 = zeros(size(U) .* [size_scale/2^options.J options.J]);
%Up1 = zeros(size(U) .* [size_scale options.J]);
Sp1 = zeros(size(U) .* [1 options.J]);
Up1 = zeros(size(U) .* [1 options.J]);

for i = 1:size(U,2)
    [tempS, tempU] = node_freq(U(:,i), options);
    Sp1(:,(i-1)*options.J+1:i*options.J) = tempS;
    Up1(:,(i-1)*options.J+1:i*options.J) = tempU;
end 
end

