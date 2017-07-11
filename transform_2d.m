function [ x_transform ] = transform_2d( X, options )

SAMPLES = size(X, 3);

if options.subsample
	M = 2^ceil(log2(size(X,1)));
	N = 2^ceil(log2(size(X,2)));

	M_sub = M / 2^options.J;
	N_sub = N / 2^options.J;

	%pad signal with zeros
	dsize = [M-size(X,1), N-size(X,2)];
	if dsize(1) ~= 0
		X(M,:,:) = 0;
	end
	if dsize(2) ~= 0
		X(:,N,:) = 0;
	end
else
	M = size(X,1);
	N = size(X,2);
	M_sub = M; %the subsampled length is the same as original
	N_sub = N;
end

clear filters;
filters = filters_2d(M, N, options);

% number of scattering coefficients
P = 0;
if options.subsample
    for m = 0:options.M
        P = P + options.L^m / 2^options.J * nchoosek(options.J, m);
    end
else
    for m = 0:options.M
        P = P + options.L^m * nchoosek(options.J, m);
    end
end
P = P * N * M;

disp(['Transformed signal length: ' int2str(P)])

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
    
    sample = zeros(1, P);
    count = 0;
    for p = 1:length(result);
        for q = 1:length(result{p}.data)
            sample(count*N_sub*M_sub+1:(count+1)*M_sub*N_sub) = result{p}.data{q}.S(:);
            count = count + 1;
        end
    end
    x_transform(:,n) = sample;
end
end

