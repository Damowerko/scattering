function [ Sp1, Up1 ] = layer_2d( U, filters, options )
size_in = size(U);
size_in(3) = size(U,3);
size_in(4) = size(U,4);

Sp1 = zeros(size_in);
Up1 = zeros(size_in .* [1 1 options.J options.L]);

for m = 1:size(U,3)
    for n = 1:size(U,4)
        [tempS, tempU] = node_2d(U(:,:,m,n), filters);
        Sp1(:,:, m, n) = tempS;
        Up1(:,:, (m-1)*options.J+1:m*options.J, (n-1)*options.L+1:n*options.L) = tempU;
    end
end
end

