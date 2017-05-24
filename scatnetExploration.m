% %% Load images and scatnet
% plane = rgb2gray(imread('airplane.png'));
% boat = imread('boat.png');
% fruit = rgb2gray(imread('fruits.png'));
% tulips = rgb2gray(imread('tulips.png'));
% run('loadScatnet');
% 
% %%
% figure
% imagesc(plane);
% colormap gray
% 
% %%
% % their functions do not work with uint8
% plane = double(tulips); 
% colormap gray
% 
% % compute wavelet operators
% filt_opt.J = 2;
% filt_opt.L = 1;
% scat_opt.oversampling = 0;
% %scat_opt.M = 2;
% 
% [Wop,filters] = wavelet_factory_2d(size(plane), filt_opt, scat_opt);
% % compute scattering
% S = scat(plane, Wop);
% 
% display_with_layer_order(S,1);
% figure;
% display_filter_bank_2d(filters);
% 
% %% filters 
% [~,filters] = wavelet_factory_2d(size(plane));
% 
% for n = 1:32
%     j = filters.psi.meta.j(n);
%     theta = filters.psi.meta.theta(n);
%     filter = filters.psi.filter{n};
%     figure
%     imagesc(filter.coefft{1})
%     title(['j = ' int2str(j) ' , theta = ' int2str(theta)])
% end
% 
% %% 1d
% N = 128;
% T = 2^12; % length of averaging window
% filt_opt.Q = [1];
% filt_opt.J = [10];
% [Wop,filters] = wavelet_factory_1d(N, filt_opt);
% 
% figure;
% for m = 1
%     hold on;
%     for k = 1:length(filters{m}.psi.filter)
%         filter = filters{m}.psi.filter{k}.coefft;
%         plot(filter);
%     end
%     hold off;
% end
% 
% figure;
% for m = 1:2
%     subplot(1,2,m);
%     hold on; 
%     for k = 1:length(filters{m}.psi.filter)
%         plot(realize_filter(filters{m}.psi.filter{k}, N)); 
%     end
%     hold off;
%     ylim([0 1.5]);
%     xlim([1 5*N/8]);
% end

%% Square Pulse Scattering
%x = sqpulse(50,100,1);
%n = 0:127;
%x = cos(n');
x = tpulse(32, 128, 1);

N = length(x);
filt_opt.Q = [1 1];
filt_opt.J = 2;
filt_opt.sigma_phi = 0.85;
filt_opt.sigma_psi = 0.85;
filt_opt.xi_psi = 3/4 * pi;
filt_opt.boundary = 'zero';
filt_opt.filter_format = 'fourier';
scat_opt.M = 2;
scat_opt.oversampling = 100000;

[Wop,filters] = wavelet_factory_1d(N, filt_opt, scat_opt);
[S,U] = scat(x, Wop);
figure;
scattergram(S{2},[],S{3},[]);
figure;
scattergram(U{2},[],U{3},[]);