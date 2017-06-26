function [ gauss ] = gaussian_2d( M, N, sigma )
[x,y] = meshgrid(1:N, 1:M);
y = y - ceil(M/2) - 1;
x = x - ceil(N/2) - 1;
gauss = exp(-(x.^2 + y.^2)/(2*sigma^2));
gauss = 1/(2*pi*sigma^2) * fftshift(gauss);
end