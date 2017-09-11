disp 'loading data'
addpath '..'

load('cora/data');

train_split = 0.8;
test_split = 1 - train_split;

train_samples = ceil(length(x_data)*train_split);
x_train = x_data(:,1:train_samples);
y_train = y_data(1:train_samples);

test_samples = floor(length(x_data)*test_split);
x_test = x_data(:,(1:test_samples)+train_samples);
y_test = y_data((1:test_samples)+train_samples);

darr = [10 20 30 40 50 60 70 80 90 100 120 140 160 180 200];
if exist('cora/results/temp.mat', 'file')
    load 'cora/results/temp.mat'
else
    result_acc = zeros(length(darr),7,3);
    result_loss = zeros(length(darr),7,3);
end
 
for J = 1:5
for M = 1:3

if J < M | (result_acc(length(darr),J,M) ~= 0 & result_loss(length(darr), J, M) ~= 0)
    continue;
end

disp(sprintf('j=%d  |  M=%d', J, M))

clear options;
options.J = J;
options.M = M;
options.psi.sigma = 0.85;
options.psi.xi = 3/4*pi;
options.phi.sigma = 0.85;
options.graph_shift = 'covariance'; % 'laplacian' or 'adjacency' or 'covariance'
options.lambda_scale = true;
options.subsample = false;

P = 0;
for m = 0:options.M
    P = P + nchoosek(options.J, m);
end

if options.subsample
    P = P * ceil(28 / 2^(options.J))^2;
else
    P = P * 1433;
end

filters = filter_graph(covariance, options);
disp(sprintf('Signal size: %d', P))


disp 'Transforming training samples...'
progress = textprogressbar(train_samples);

clear train_transform
train_transform = zeros(P, size(x_train,3));
for i = 1:size(x_train,2)
    progress(i);

    x = x_train(:,i);
    x = x(:);

    scat = transform_graph(x, filters, options);
    y = [];
    for m = 1:length(scat)
        for n = 1:length(scat{m}.data)
            y = [y; scat{m}.data{n}.S];
        end
    end
    train_transform(:,i) = y(:);
end

if size(train_transform, 1) ~= P
    error('Invalid signal size')
end

disp 'Transforming testing samples...'
progress = textprogressbar(test_samples);

clear test_transform
test_transform = zeros(P, size(x_test,3));
for i = 1:size(x_test,2)
    progress(i);
    
    x = x_test(:,i);
    x = x(:);
    
    scat = transform_graph(x, filters, options);
    y = [];
    for m = 1:length(scat)
        for n = 1:length(scat{m}.data)
            y = [y; scat{m}.data{n}.S];
        end
    end
    test_transform(:,i) = y(:);
end

%% Training
v = cell(7, 1);
m = cell(7, 1);

disp 'Training classifier...'
progress = textprogressbar(7);

for n = 1:7
    s = train_transform(:, y_train == n);
    m{n} =  mean(s,2);
    [V,E] = eig(cov((bsxfun(@minus,s,m{n}))'));
    [E,I] = sort(diag(E),'descend');
    v{n} = V(:,I);
    progress(n);
end

disp 'Evaluation'
progress = textprogressbar(length(darr));
for dn = 1:length(darr)
    %% Evaluations
    d = darr(dn); % number of PCA coefficients to use

    if d > length(v{n})
        continue;
    end

    E = zeros(7,size(x_test,2));
    error = zeros(test_samples, 7);
    for n = 1:7
        s = bsxfun(@minus, test_transform, m{n});
        pc = v{n}(:,1:d);
        err_temp = zeros(size(pc,2)+1, size(s,2));
	    err_temp(1,:) = sum(abs(s).^2,1);
	    err_temp(2:end,:) = -abs(pc'*s).^2;
        error(:, n) = transpose(sum(err_temp, 1));
        %err = sqrt(cumsum(err,1));
    end


    % find probability
    prob = bsxfun(@rdivide, error, sum(error, 2));
    prob = 1 - prob;

    % logical index actual classes
    Y_test = repmat(y_test(:),1,7) == repmat(1:7,length(y_test),1);
    Y_test = transpose(Y_test);

    logloss = -sum(log(prob(Y_test))) / size(y_test,1);

    % accuracy
    [~,I] = sort(error,2, 'ascend');
    correct = I(:,1) == y_test(:);
    accuracy = sum(correct) / length(correct);

    %disp(sprintf('Accuracy: %f  |  Logloss: %f', accuracy, logloss))
    result_loss(dn, J, M) = logloss; % save results
    result_acc(dn, J, M) = accuracy;
    
    save(['cora/results/temp'],'result_loss','result_acc');
    progress(dn); % update progress bar
end

    [acc_max, ai] = max(result_acc(:,J,M));
    [loss_min, li] = min(result_acc(:,J,M));
    fprintf('Accuracy: %f (%d)  |  Logloss: %f (%d)', acc_max, ai, loss_min, li)

end
end

save(['cora/results/' datestr(datetime)],'result_loss','result_acc');
delete('cora/results/temp.mat');
