% %% 1D
% x = sqpulse(50,100,1);
% %q = 0:127;
% %x = cos(q');
% %x = tpulse(32, 128, 1);
% 
% % parameters
% M = 2;
% J = 2;
% 
% options.J = J;
% options.psi.sigma = 0.85;
% options.psi.xi = 3/4*pi;
% options.phi.sigma = 0.85;
% 
% U = x;
% for m = 1:M
%     figure;
%     colormap gray
%     [S, U] = layer_freq(U, options);
%     labels = labels_1d(J,M);
%     
%     subplot(2,1,1);
%     imagesc(transpose(S));
%     title(['Order ' int2str(m)])
%     xlabel('Scattering Coefficients') % x-axis label
%     ylabel('j index') % y-axis label
%     yticks(1:length(labels));
%     yticklabels([labels]);
%     
%     subplot(2,1,2);
%     imagesc(transpose(U));
%     xlabel('Layer output') % x-axis label
%     ylabel('j index') % y-axis label
%     yticks(1:length(labels));
%     yticklabels([labels]);
% end
% 
% %% 2D
% x = rgb2gray(imread('airplane.png'));
% x = double(x);
% x = [ones(32,32) zeros(32,32) ; zeros(32,64)];
% 
% options.J = 2;
% options.L = 1;
% options.sigma_phi = 0.85;
% options.sigma_psi = 0.85;
% options.xi_psi = 3/4 * pi;
% options.M = 4;
% 
% %M = 2^ceil(log2(size(x,1)));
% %N = 2^ceil(log2(size(x,2)));
% %M = 2^options.J * ceil (size(x,1)/2^options.J + 2)
% %N = 2^options.J * ceil (size(x,2)/2^options.J + 2)
% M = size(x,1);
% N = size(x,2);
% 
% filters = filters_2d(M, N, options);
% 
% U = x;
% for m = 1:options.M+1
%     [S, U] = layer_2d(U, filters, options);    
%     
%     if m == 1
%         figure;
%         imagesc(S);  
%     else
%         img = [];
%         for p = 1:size(S,3)
%             tmp = [];
%             for q = 1:size(S,4)
%                 tmp = [tmp S(:,:,p,q)];
%             end
%             img = [img; tmp];
%         end
% 
%         figure;
%         imagesc(real(img));
%         title(['layer ' int2str(m)]);
%     end
% end

%% MNIST get data

%load data
x_train = loadMNISTImages('MNIST/train-images.idx3-ubyte');
x_train = reshape(x_train, [28 28 60000]);
y_train = loadMNISTLabels('MNIST/train-labels.idx1-ubyte');

x_test = loadMNISTImages('MNIST/t10k-images.idx3-ubyte');
x_test = reshape(x_test, [28 28 10000]);
y_test = loadMNISTLabels('MNIST/t10k-labels.idx1-ubyte');

train_samples = 13000;
x_train = x_train(:,:,1:train_samples);
y_train = y_train(1:train_samples);

test_samples = 1000;
x_test = x_test(:,:,1:test_samples);
y_test = y_test(1:test_samples);


clear options;
options.J = 2;
options.L = 3;
options.sigma_phi = 0.85;
options.sigma_psi = 0.85;
options.xi_psi = 3/4 * pi;
options.M = 2;

train_transform = transform_2d(x_train, options);
test_transform = transform_2d(x_test, options);


%% PCA matrix
d = 10; % number of PCA coefficients to use
pca_transform = transpose(pca(transpose(train_transform), 'NumComponents', d));

%% Transform dataset into PC

x_train = pca_transform(1:d, :) * train_transform;
x_test = pca_transform(1:d, :) * test_transform;

% classes
classes = zeros(d,10);
for class = 0:9
    L = y_train == class;
    classes(:, class+1) = mean(x_train(:,L),2);
end

%% Evaluation

D = zeros(10,size(x_test,2));
for class = 1:10
    d = (classes(:,class) - x_test);
    d = sqrt(sum(d.^2,1));
    D(class,:) = d;
end

% find probability
l1_norm = sum(D,1);
D = D./l1_norm;
prob = 1-D;

% logical index actual classes
Y_test = repmat(y_test(:),1,10) == repmat(1:10,length(y_test),1);
Y_test = transpose(Y_test);

logloss = -sum(log(prob(Y_test))) / size(y_test,1);
disp(['Logloss: ' num2str(logloss)])


% accuracy
[~,I] = sort(D,1, 'ascend');
correct = I(1,:)' == y_test+1;
accuracy = sum(correct) / length(correct);
disp(['Accuracy: ' num2str(accuracy)]);










