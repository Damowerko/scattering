%% Signal
%x = sqpulse(50,100,1);
%n = 0:127;
x = cos(n');
%x = tpulse(32, 128, 1);

% parameters
M = 3;
J = 2;

options.J = J;
options.psi.sigma = 0.85;
options.psi.xi = 3/4*pi;
options.phi.sigma = 0.85;

U = x;
for m = 1:M
    figure;
    colormap gray
    [S, U] = layer_freq(U, options);
    subplot(2,1,1);
    imagesc(transpose(S));
    title(['Order ' int2str(m)])
    xlabel('Scattering Coefficients') % x-axis label
    ylabel('j index') % y-axis label
    subplot(2,1,2);
    imagesc(transpose(U));
    xlabel('Layer output') % x-axis label
    ylabel('j index') % y-axis label
end