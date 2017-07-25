function [n] = map_lambdas(n, tmin, tmax)
n = n - min(n); % align range with 0
n = n / max(n); % range is now between 0:1
n = n * (tmax-tmin); % range is now between 0:(tmax-tmin)
n = n + tmin; % range is now between tmin:tmax
end
