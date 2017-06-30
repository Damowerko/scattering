function [ labels ] = labels_1d( J, m )
labels = [];
for m1 = 1:m
    j = 0:J^m-1;
    j = mod(j, J^(m-m1+1));
    j = j./J^(m-m1);
    j = floor(j);
    labels = [labels int2str(j')];
end
end

