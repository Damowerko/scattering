function [P] = scat_size(N, options)
P = 0;
for m = 0:options.M
    P = P + nchoosek(options.J, m);
end
P = N*P;
end

