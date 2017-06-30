function [ x_transform ] = transform_2d( X, options )

SAMPLES = size(X, 3);

%M = 2^ceil(log2(size(x,1)));
%N = 2^ceil(log2(size(x,2)));
%M = 2^options.J * ceil (size(x,1)/2^options.J + 2)
%N = 2^options.J * ceil (size(x,2)/2^options.J + 2)
M = size(X,1);
N = size(X,2);

clear filters;
filters = filters_2d(M, N, options);

% scattering coefficients
P = 0;
for m = 0:options.M
    P = P + m^options.L * nchoosek(options.J, m);
end
P = P * N * M;

x_train_transform = zeros(P, SAMPLES);

for n = 1:SAMPLES
    x = X(:,:,n);
    
    clear result;
    result{1} = struct();
    result{1}.data{1}.U = x;
    result{1}.data{1}.S = [];
    result{1}.data{1}.j = [];
    result{1}.data{1}.r = [];
    
    for m = 1:options.M+1
        result = layer_2d_class(result, filters, options);
    end
    result(options.M+2) = [];
    
    sample = ones(1, P);
    count = 0;
    for p = 1:length(result);
        for q = 1:length(result{p}.data)
            sample(count*N*M+1:(count+1)*M*N) = result{p}.data{q}.S(:);
            count = count + 1;
        end
    end
    x_transform(:,n) = sample;
end
end

