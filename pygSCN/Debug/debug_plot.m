function [] = debug_plot(trial,J)
load(sprintf('%d_j%d',trial,J));
for j = 1:J
    figure
    plot(h.frequencies, h.psi{j})
    title(sprintf('Plot of psi trial %d with J=%d and j=%d', trial,J,j));
    xlabel('frequency (if not integer then scaled)')
    ylabel('frequency response')
end
figure
plot(h.frequencies, h.phi)
title(sprintf('Plot of phi trial %d with J=%d', trial,J));
xlabel('frequency (if not integer then scaled)')
ylabel('frequency response')
end

