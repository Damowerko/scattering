function [ A ] = image_adj( M, N )
%IMAGE_ADJ An M*N x M*N adjacency matrix corresponding to an M x N image.
diagVec1 = repmat([ones(N-1,1); 0],M,1);  %# Make the first diagonal vector
                                          %#   (for horizontal connections)
diagVec1 = diagVec1(1:end-1);             %# Remove the last value
diagVec2 = ones(N*(M-1),1);               %# Make the second diagonal vector
                                          %#   (for vertical connections)
A = diag(diagVec1,1)+...                %# Add the diagonals to a zero matrix
      diag(diagVec2,N);
A = A+A.';                          %# Add the matrix to a transposed
                                          %#   copy of itself to make it
                                          %#   symmetric
end

