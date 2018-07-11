%% Pick stations
location = "CA";

availability_filename = strcat("stations", location, ".mat");
if exist(availability_filename, 'file') == 2
    load(availability_filename);
else
    [X,~] = pick_greedy(location);
    save(availability_filename,'X');
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
fields = {'PressureSeaLevelMBar', 'TemperatureC'};
k_matrix = [
    1 0 0;
    0 1 0;
    0 0 1;
];
k_matrix = 1;
n_fields = length(fields);  
dir = strcat("EarthNetworks/Data/parsed/", location, "/");
[data, labels, n_samples] = import_data(fields, Ix, 0, dir);

%% Make train and test split
split = 0.9; % proportion of training samples
I_train = rand(length(labels),1) < split;
I_test = ~I_train;

%% Generate adjacency and laplacian matrices
C = cov(data(:,I_train).');

%% Scattering
experiment_id = '31';

% Possible number of PCA components that can be kept (From 1 to n)
darr = 1:n_stations; % pca w number

missed_detection = NaN(length(darr), 1);
false_alarm = NaN(length(darr), 1);
accuracy = NaN(length(darr), 1);
avg_error_rate = NaN(length(darr), 1);
f1 = NaN(length(darr), 1);

% Compute and apply PCA
transformed = data;
%V = eig(C);
%transformed = V' * data;

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

    missed_detection(dn) = 1-tp_rate; % save results
    false_alarm(dn) = 1-tn_rate;
    accuracy(dn) = sum(pred==testY)/length(testY);
    avg_error_rate(dn) = 0.5*(1-tp_rate)+0.5*(1-tn_rate);
    precision = sum(positive_pred(positive)) / sum(positive_pred);
    recall = sum(positive_pred(positive)) / sum(positive);
    f1(dn) = 2 * (precision * recall)/(precision + recall);
end 

if exist(['EarthNetworks/Results/' experiment_id '.mat'], 'file') ~= 2
    save(['EarthNetworks/Results/' experiment_id],'missed_detection', ...
        'false_alarm','accuracy','avg_error_rate', 'f1', 'location');
else
    disp(['Experiment with id ' experiment_id ' already exists'])
end

[aer,I] = min(avg_error_rate(:));
md = missed_detection(I);
fa = false_alarm(I);
fprintf('\n----------------------\n')
fprintf('Max accuracy %f\n', max(accuracy));
fprintf('Min average error rate %f\n', aer);
fprintf('Missed detection rate %f\n', md);
fprintf('False alarm rate %f\n', fa);
fprintf('Max f1 %f\n', max(f1));
fprintf('----------------------\n')









