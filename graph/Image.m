%% Image generation
addpath '..'

x = zeros(64);
x(1:32, 1:32) = ones;
x = x(:);
[M, N] = size(x);

%% Filters
% options
options.J = 3;
options.M = 2;
options.psi.sigma = 0.85;
options.psi.xi = 3/4*pi;
options.phi.sigma = 0.85;

% transform
scat = transform_graph(x, image_adj(M,N), options)

disp('Write images in ./images/');
if exist('./images', 'dir') ~= 7
    mkdir('images')
else
    delete('./images/*.png')
end
for m = 1:length(scat)
    for n = 1:length(scat{m}.data)
        img = scat{m}.data{n}.S;
        img = reshape(img, M, N);
        img = uint8(mat2gray(img, [0 max(max(img))]) * 255);
        file = ['./images/' int2str(scat{m}.data{n}.j) '.png'];
        imwrite(img, file);
    end
end
disp('Done!');









