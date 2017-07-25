addpath '..'
x_train = loadMNISTImages('../MNIST/train-images.idx3-ubyte');
x_train = reshape(x_train, [28 28 60000]);

train_samples = 10000;
x_train = x_train(:,:,1:train_samples);
y_train = y_train(1:train_samples);
