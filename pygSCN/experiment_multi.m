%% Pick stations
[X,Y] = pick_greedy();
%% Chose x and y
n_stations = 25;
Ix = logical(X(:,n_stations+1));
Iy = logical(Y(:,n_stations+1));

%% Import data
fields = {'PressureSeaLevelMBar'};

data = cell(1, length(fields));
labels = cell(1, length(fields));
I = zeros(length(fields), length(Iy));

for i = 1:length(fields)
    field = fields{i};
    data_raw = importfile(['Data/parsed/' field '.csv'])';
    data{i} = data_raw(Ix,:);
    
    % Threshold by number of missing points
    missing = isnan(data{i});
    thres = 0; % missing data threshold
    I(i,:) = sum(missing,1) <= thres;
end
I = sum(I,1) == size(I,1);

for i = 1:length(fields)
    data{i} = data{i}(:,I);
    labels_raw = csvread('Data/parsed/reliability_Ny.csv', 1, 4)';
    labels = labels_raw(I);
end

n_datapoints = sum(I);
n_fields = length(fields);

%% Interpolate
% G = gps(0.9,false);
% L = full(laplacian(G));
% L = L(Ix,Ix);
% [V,D] = eig(L);
% V1 = inv(V);
% 
% for i = 1:size(data,1)
%     row = data(i,:);
%     f = V1a;
%     Aeq = eye(sum(~isnan(row)));
%     beq = row(~isnan(row));
%     v = linprog(f,[],[],Aeq,beq);
%     
% end

%% Make train and test splits
split = 0.5;
I = rand(length(labels),1) < split;

trainX = cell(1,length(fields));
testX = cell(1,length(fields));

for i = 1:length(fields)
    trainX{i} = data{i}(:,I);
    trainY = labels(:,I);
    testX{i} = data{i}(:,~I);
    testY = labels(:,~I);
end

% Train and Test
% Train and Test
[G, A] = gps(0.0000001,0,0.1);
A = A(Ix,Ix);
L = full(laplacian(graph(A)));

darr = 1:sum(Ix); % pca w number

Mmax = 2;
Jmax = 5;
true_positive = zeros(length(darr), Jmax, Mmax);
true_negative = zeros(length(darr), Jmax, Mmax);
accuracy = zeros(length(darr), Jmax, Mmax);

experiment_id = '28';
tic

for M = 1:Mmax
    for J = 1:Jmax
        if J < M
            continue;
        end
        fprintf('Working on M=%d/%d , J=%d/%d\n', M, Mmax, J, Jmax)
        
        options = options();
        options.J = J;  
        options.M = M;
        options.filter.operator = 'L'; % A L
        options.filter.normalization = 'max2'; % None max2 l1
        options.filter.lambda_scale = true; % 
        options.filter.psi_sigma = 0.85;
        options.filter.psi_xi = 3/4*pi;
        options.filter.phi_sigma = 0.005;
        options.filter.log_scale = 2;
        
        filters = filter_graph(A, @wavelet_morlet, @low_pass, options);
        
        disp 'Tranforming fields...'
        
        N = scat_size(n_stations,options);
        trainX_transformed = zeros(N*n_fields,n_stations);
        testX_transformed = zeros(N*n_fields, n_stations);
        
        progress = textprogressbar(n_fields);
        for i = 1:n_fields
            for n = 1:size(trainX{i},2)
                trainX_transformed( ((i-1)*N+1):(i*N), n) = transform(trainX{i}(:,n), filters, options);
            end
            
            for n = 1:size(testX{i},2)
                testX_transformed( ((i-1)*N+1):(i*N), n) = transform(testX{i}(:,n), filters, options);
            end
            
            progress(i)
        end
        
        % Training
        v = cell(2, 1);
        m = cell(2, 1);

        disp 'Training classifier...'

        for n = 1:2
            s = trainX_transformed(:,trainY == (n-1));
            m{n} =  mean(s,2);
            [V,E] = eig(cov((bsxfun(@minus,s,m{n}))'));
            [E,I] = sort(diag(E),'descend');
            v{n} = V(:,I);
        end

        disp 'Evaluation'
        for dn = 1:length(darr)
            % Evaluations
            d = darr(dn); % number of PCA coefficients to use

            if d > length(v{n})
                continue;
            end

            error = zeros(2,size(testX_transformed,2));
            for n = 1:2
                s = bsxfun(@minus, testX_transformed, m{n});
                pc = v{n}(:,1:d);
                err_temp = zeros(size(pc,2)+1, size(s,2));
                err_temp(1,:) = sum(abs(s).^2,1);
                err_temp(2:end,:) = -abs(pc'*s).^2;
                error(n, :) = transpose(sum(err_temp, 1));
            end

            
            [~,pred] = min(error, [], 1);
            pred = pred - 1; % matlab is 1 indexed
            
            
            positive = testY==1;
            positive_pred = pred==1;
            negative = testY==0;
            negative_pred = pred==0;
                        
            tp_rate = sum(positive_pred(positive))/sum(positive);
            tn_rate = sum(negative_pred(negative))/sum(negative);

            true_positive(dn, J, M) = tp_rate; % save results
            true_negative(dn, J, M) = tn_rate;
            accuracy(dn, J, M) = sum(pred==testY)/length(testY);

        end 
    end
end
if exist(['pygSCN/Results/' experiment_id '.mat'], 'file') ~= 2
    save(['pygSCN/Results/' experiment_id],'true_positive','true_negative','accuracy','options');
else
    disp(['Experiment with id ' experiment_id ' already exists'])
end

fprintf('Max accuracy %f\n', max(max(max(accuracy))))
toc











