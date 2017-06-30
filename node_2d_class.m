function [ S, U ] = node_2d(data, filters )
x = data.U;
size_in = size(x);

size_filt = size(filters.psi);
size_filt(3) = size(filters.psi,3);
size_filt(4) = size(filters.psi,4);
dsize = size_filt(1:2) - size(x);
% pad signal
%x = [x zeros(size_in(1), dsize(2)); zeros(dsize(1), size_in(2)) zeros(dsize(1), dsize(2))];

x = pad_signal(x, size_filt(1:2));
X = fft2(x);

U = zeros(size_in(1), size_in(2), size_filt(3), size_filt(4));
for j = 0:size_filt(3)-1
    for l = 1:size_filt(4)
        y = ifft2(X .* filters.psi(:,:,j+1,l));
        y = unpad_signal(y, [0 0], size_in);
        U(:,:,j+1,l) = abs(y(1:size_in(1), 1:size_in(2)));
    end
end

% phi
y = real(ifft2(X .* filters.phi));
y = unpad_signal(y, [0 0], size_in);
S = y(1:size_in(1),1:size_in(2));

end

