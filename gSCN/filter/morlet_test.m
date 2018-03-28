figure;
hold on;
N = 1024;
for j = 0:5
    f = wavelet_morlet(1:N, 0.85, 3/4*pi, j);
    plot((2*pi)*(0:N-1)/(N-1),f);
end
