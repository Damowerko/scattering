function [ accuracy ] = knn( X, y, k )
d = pdist(X);
[~,nn] = sort(d, 1, 'ascend');
nn_label = y(nn(2:(k + 1), :));
predictions = mode(nn_label, 1)';
accuracy = 1 - norm(y-predictions)/norm(y);
end

