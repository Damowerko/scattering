function [G,A] = gps(thres, binary, sigma)
file = 'data/parsed/distance.csv';
distance = importdata(file);
A = exp(-sigma*distance.^2); % form adjacency and normalize
A(logical(eye(size(A)))) = 0;
A(isnan(A)) = 0;
A(A <= thres) = 0; %threshold to make it sparse
if binary
    A(A>0) = 1;
end
G = graph(A);
end