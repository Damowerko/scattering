function [ output ] = layer_2d_class( input, filters, options )
m = length(input);

output = input;
output{m+1} = struct();
output{m+1}.data = {};

for n = 1:length(input{m}.data)
    [S, U] = node_2d_class(input{m}.data{n}, filters, options);
    output{m}.data{n}.S = S;
    j = input{m}.data{n}.j;
    r = input{m}.data{n}.r;
    for j = 0:size(U,3)-1
        if isempty(input{m}.data{n}.j) || j > input{m}.data{n}.j(end)
            for r = 1:size(U,4)
                x = U(:,:,j+1,r);
                data = struct();
                data.U = x;
                data.S = [];
                data.j = [input{m}.data{n}.j j];
                data.r = [input{m}.data{n}.r r];
                output{m+1}.data = [output{m+1}.data data];
            end
        end 
    end
end

