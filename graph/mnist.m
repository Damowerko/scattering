disp 'loading data'
addpath '..'
x_train = loadMNISTImages('../MNIST/train-images.idx3-ubyte');
x_train = reshape(x_train, [28 28 60000]);
y_train = loadMNISTLabels('../MNIST/train-labels.idx1-ubyte');

x_test = loadMNISTImages('../MNIST/t10k-images.idx3-ubyte');
x_test = reshape(x_test, [28 28 10000]);
y_test = loadMNISTLabels('../MNIST/t10k-labels.idx1-ubyte');

train_samples = 10000;
x_train = x_train(:,:,1:train_samples);
y_train = y_train(1:train_samples);

test_samples = 1000;
x_test = x_test(:,:,1:test_samples);
y_test = y_test(1:test_samples);


clear options;
options.J = 4;
options.M = 2;
options.psi.sigma = 0.85;
options.psi.xi = 3/4*pi;
options.phi.sigma = 0.85;

P = 0;
for m = 0:options.M
    P = P + nchoosek(options.J, m);
end
P = P * 28 * 28;

disp 'generating filters'

filters = filter_graph(image_adj(28,28), options);

disp 'scattering transform'

train_transform = zeros(P, size(x_train,3));
for i = 1:size(x_train,3)
    disp(['Test sample ' int2str(i) '/' int2str(size(x_train,3))])
    x = x_train(:,:,i);
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

test_transform = zeros(P, size(x_test,3));
for i = 1:size(x_test,3)
    disp(['Test sample ' int2str(i) '/' int2str(size(x_test,3))])
    x = x_test(:,:,i);
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
v = cell(10, 1);
m = cell(10, 1);

for n = 1:10
    disp(['training: ' int2str(n) '/10'])
    s = train_transform(:, y_train == (n-1));
    m{n} =  mean(s,2);
    [V,E] = eig(cov((bsxfun(@minus,s,m{n}))'));
    [E,I] = sort(diag(E),'descend');
    v{n} = V(:,I);
end

disp 'evaluating'
%% Evaluations
d = 40; % number of PCA coefficients to use
E = zeros(10,size(x_test,2));
error = zeros(test_samples, 10);
for n = 1:10
    s = bsxfun(@minus, test_transform, m{n});
    pc = v{n}(:,1:d);
    err_temp = zeros(size(pc,2)+1, size(s,2));
	err_temp(1,:) = sum(abs(s).^2,1);
	err_temp(2:end,:) = -abs(pc'*s).^2;
    error(:, n) = transpose(sum(err_temp, 1));
    %err = sqrt(cumsum(err,1));
end

disp 'done!'
% find probability
prob = bsxfun(@rdivide, error, sum(error, 2));
prob = 1 - prob;

% logical index actual classes
Y_test = repmat(y_test(:),1,10) == repmat(1:10,length(y_test),1);
Y_test = transpose(Y_test);

logloss = -sum(log(prob(Y_test))) / size(y_test,1);
disp(['Logloss: ' num2str(logloss)])

% accuracy
[~,I] = sort(error,2, 'ascend');
correct = I(:,1) == y_test+1;
accuracy = sum(correct) / length(correct);
disp(['Accuracy: ' num2str(accuracy)]);
