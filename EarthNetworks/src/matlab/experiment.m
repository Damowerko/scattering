%% Pick stations
if exist('stations.mat', 'file') == 2
    load('stations.mat');
else
    [X,~] = pick_greedy();
    save('stations','X');
end
%X: Determines which stations to pick if a certain number of stations is
%desired. Each column tells which stations to choose for that number of
%desired stations (i.e. column 1 gives me zero stations, column 2 gives me
%the best station if I want only one, column 3 gives me two stations if I
%want the best two, etc.).
% For some reason, after selecting the best 60 stations, it doesn't select
% any more stations.
%\\\\\ TODO: Double check that, indeed, there are only 60 stations that
%have data all the time.

% Y: Not being used.
%% Chose x and y
n_stations = 25; % Number of stations (nodes in the network)
% Which are the best stations for the desired n:
Ix = logical(X(:,n_stations+1));

%% Import data
fields = {'PressureSeaLevelMBar',};
k_matrix = [
    1 0 0;
    0 1 0;
    0 0 1;
];
k_matrix = 1
n_fields = length(fields);  
[data, labels, n_samples] = import_data(fields, Ix, 0);

%% Make train and test split
split = 0.9; % proportion of training samples
I_train = rand(length(labels),1) < split;
I_test = ~I_train;

%% Generate adjacency and laplacian matrices
[G, A] = gps(0.0000001,0,0.1);
A = A(Ix,Ix);
L = full(laplacian(graph(A)));
C = cov(data(:,I_train).');

%% Scattering
experiment_id = '27';
 
options = options();
options.filter.map2pi = true;
options.filter.operator = 'L'; % A L L_norm
options.filter.normalization = 'max2'; % None max2 l1
options.filter.lambda_scale = true; % scale with respect to frequency
options.filter.psi_sigma = 0.85;
options.filter.psi_xi = 3/4*pi;
options.filter.phi_sigma = 0.005;
options.filter.log_scale = 2; % if not use set to x<=0
options.filter.exp_scale = NaN;

Mmax = 2; % Maximum number of layers
Jmax = 5; % J of scattering



% Possible number of PCA components that can be kept (From 1 to n)
darr = 1:n_stations; % pca w number

missed_detection = NaN(length(darr), Jmax, Mmax);
false_alarm = NaN(length(darr), Jmax, Mmax);
accuracy = NaN(length(darr), Jmax, Mmax);
avg_error_rate = NaN(length(darr), Jmax, Mmax);


for M = 1:Mmax
    for J = 1:Jmax
        if J < M
            continue;
        end
        fprintf('Working on M=%d/%d , J=%d/%d\n', M, Mmax, J, Jmax)
        options.J = J;  
        options.M = M;
        
        [filters,h]= filter_graph(A, @wavelet_morlet, @gaussian, options, k_matrix);
        
        % Apply transformation
        P = scat_size(n_stations,options); % calculate number of coeff per field
        transformed = zeros(P*n_fields, n_samples);
        
        for i = 1:n_samples
            transformed(:,i) = transform(data(:,i), filters,options);
        end
        
        % Make the splits
        trainX = transformed(:,I_train);
        testX = transformed(:,I_test);
        trainY = labels(I_train);
        testY = labels(I_test);
        
        % Training
        v = cell(2, 1);
        m = cell(2, 1);
        for n = 1:2
            s = trainX(:,trainY == (n-1)); % centered version of the transform.
            m{n} =  mean(s,2);
            [V,E] = eig(cov((bsxfun(@minus,s,m{n}))'));
            [E,I] = sort(diag(E),'descend');
            v{n} = V(:,I);
        end
        
        % Testing
        for dn = 1:length(darr)
            % Evaluations
            d = darr(dn); % number of PCA coefficients to use

            if d > length(v{1})
                continue;
            end

            error = zeros(2,size(testX,2));
            for n = 1:2
                s = bsxfun(@minus, testX, m{n});
                pc = v{n}(:,1:d);
                err_temp = zeros(size(pc,2)+1, size(s,2));
                err_temp(1,:) = sum(abs(s).^2,1); % WHYY??
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

            missed_detection(dn, J, M) = 1-tp_rate; % save results
            false_alarm(dn, J, M) = 1-tn_rate;
            accuracy(dn, J, M) = sum(pred==testY)/length(testY);
            avg_error_rate(dn, J, M) = 0.5*(1-tp_rate)+0.5*(1-tn_rate);

        end 
    end
end
if exist(['pygSCN/Results/' experiment_id '.mat'], 'file') ~= 2
    save(['pygSCN/Results/' experiment_id],'missed_detection', ...
        'false_alarm','accuracy','avg_error_rate','options');
else
    disp(['Experiment with id ' experiment_id ' already exists'])
end

[aer,I] = min(avg_error_rate(:));
md = missed_detection(I);
fa = false_alarm(I);
fprintf('Average error rate %f\n', aer);
fprintf('Missed detection rate %f\n', md);
fprintf('False alarm rate %f\n', fa);










