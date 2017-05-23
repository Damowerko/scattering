function [ Sp1, Up1 ] = layer( U, options )

Sp1 = zeros(size(U) .* [1 options.J]);
Up1 = zeros(size(U) .* [1 options.J]);

for i = 1:size(U,2)
    [tempS, tempU] = node(U(:,i), options);
    Sp1(:,(i-1)*options.J+1:i*options.J) = tempS;
    Up1(:,(i-1)*options.J+1:i*options.J) = tempU;
end 
end

