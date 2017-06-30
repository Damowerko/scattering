%% Poly_filter

A = eye(20);
A = circshift(A,1,2);
A_sym = A + A';

D = diag(sum(A,2));
L = D - A_sym;

[V, E] = eig(L);
[~,I] = sort(diag(real(E)),'ascend');
V = V(:, I);

hf = [ones(10,1); zeros(10,1)]; % square pulse
%hf = hf / norm(hf);

H = poly_filter(hf, V, E); % square pulse filter

x = ones(20,1); % cexp with freq 0
y = H*x;