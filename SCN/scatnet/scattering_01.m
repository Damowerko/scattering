%% Scattering transforms.
% 2017/02/01. Version 01.
% Try to create the scattering transform wavelets and understand them.

clear all;
close all;
penn_blue = [0 0 1]
penn_red = [1 0 0]

%% Let's start with Morlet wavelet:
% p(u)=a (e^{j u xi}-b) e^{-|u|^2/(2 s^2)}
s=0.85;
xi=3*pi/4; % BRUNA
b=exp(-xi^2*s^2/2); % This works. BRUNA
a=sqrt(1/sqrt(pi*s^2)/(1-2*b*exp(-s^2*xi^2/4)+b^2)); % This works. BRUNA

Ts=1e-3;
fs=1/Ts;

u=(-9*s:Ts:9*s).'; length_u=length(u);
p=a*(exp(1i*xi*u)-b).*exp(-abs(u).^2/(2*s^2)); % BRUNA

% Wikipedia Morlet Wavelet:
% kp_s=exp(-s^2/2);
% c_s=1/sqrt(1+exp(-s^2)-2*exp(-3*s^2/4));
% p=c_s*pi^(-1/4)*exp(-u.^2/2).*(exp(1i*s*u-kp_s));

%% Let's create the convolution network

J=floor(log2(max(u)));
j=0:J;
K=6;
k=(0:K-1).';
r=exp(1j*2*pi*k/K);

l=zeros(J+1,K);
pnet=zeros(length_u,K,J);
for it1=1:J+1
	l(it1,:)=2^(-j(it1))*r.';
	c=sqrt(abs(l));
	for it2=1:K
		pnet(:,it2,it1)=c(it1,it2)*...
			a*(exp(1i*xi*l(it1,it2)*u)-b).*...
			exp(-abs(l(it1,it2)*u).^2/(2*s^2)); % BRUNA
%		pnet(:,it2,it1)=c(it1,it2)*...
%			c_s*pi^(-1/4)*exp(-abs(u*l(it1,it2)).^2/2).*...
%			(exp(1i*s*(u*l(it1,it2))-kp_s)); % WIKIPEDIA
% 		figure(); ...
% 			subplot(2,1,1); plot(u,real(exp(1i*xi*l(it1,it2)*u)-b));
% 			subplot(2,1,2); plot(u,imag(exp(1i*xi*l(it1,it2)*u)-b));
% 		pnet(:,it2,it1)=pnet(:,it2,it1)-...
% 			sum(pnet(:,it2,it1)*Ts);
% 		sum(pnet(:,it2,it1)*Ts)
% 		pnet(:,it2,it1)=pnet(:,it2,it1)/...
% 			sqrt(sum(abs(pnet(:,it2,it1).^2*Ts)));
% 		sum(abs(pnet(:,it2,it1).^2*Ts))
	end
end

% for it1=1:J+1
% 	figure('units','normalized','outerposition',[0 0 1 1]);
% 	
% 	for it2=1:K
% 		
% 		subplot(2,K,it2)
% 		plot_morlet_real=plot(u,real(pnet(:,it2,it1)));
% 		title(['j=' num2str(j(it1)) '; k=' num2str(k(it2))]);
% 		set(plot_morlet_real,'color',penn_blue,'linewidth',2);
% 		ylabel('Re\{ \psi (u)\}'); xlabel('u');
% 		xlim([min(u) max(u)])
% 		pbaspect([1.6 1 1])
% 		grid on
% 
% 		subplot(2,K,K+it2)
% 		plot_morlet_imag=plot(u,imag(pnet(:,it2,it1)));
% 		set(plot_morlet_imag,'color',penn_red,'linewidth',2);
% 		ylabel('Im\{ \psi (u) \}'); xlabel('u');
% 		xlim([min(u) max(u)])
% 		pbaspect([1.6 1 1])
% 		grid on
% 	
% 	end
% end

%% DFT time!

[f,hp,f_c,hp_c]=dft(p,fs);
length_f=length(f);

i_p=find(abs(f_c)<fs/1000);

figure('units','normalized','outerposition',[0 0 1 1]);

subplot(2,2,1)
plot_morlet_real=plot(u,real(p));
set(plot_morlet_real,'color',penn_blue,'linewidth',2);
title('Morlet wavelet'); ylabel('Re\{ \psi (u)\}'); xlabel('u');
xlim([min(u) max(u)])
pbaspect([1.6 1 1])
grid on

subplot(2,2,3)
plot_morlet_imag=plot(u,imag(p));
set(plot_morlet_imag,'color',penn_red,'linewidth',2);
ylabel('Im\{ \psi (u) \}'); xlabel('u');
xlim([min(u) max(u)])
pbaspect([1.6 1 1])
grid on

subplot(2,2,2)
plot_hmorlet_real=plot(f_c(i_p),real(hp_c(i_p)));
set(plot_hmorlet_real,'color',penn_blue,'linewidth',2);
title('Morlet wavelet transform'); ylabel('Re\{  \psi  ( f ) \}');...
	xlabel('f');
xlim([min(f_c(i_p)) max(f_c(i_p))])
pbaspect([1.6 1 1])
grid on

subplot(2,2,4)
plot_hmorlet_imag=plot(f_c(i_p),imag(hp_c(i_p)));
set(plot_hmorlet_imag,'color',penn_red,'linewidth',2);
ylabel('Im\{ \psi ( f ) \}'); xlabel('f');
xlim([min(f_c(i_p)) max(f_c(i_p))])
pbaspect([1.6 1 1])
grid on

hpnet=zeros(length_f,K,J);

for it1=1:J+1
	for it2=2:1:K
		[~,~,~,hpnet(:,it2,it1)]=dft(pnet(:,it2,it1),fs);
		
		figure('units','normalized','outerposition',[0 0 1 1]);
		
		subplot(2,2,2)
		plot_hmorlet_real=plot(f_c(i_p),real(hpnet(i_p,it2,it1)));
		set(plot_hmorlet_real,'color',penn_blue,'linewidth',2);
		title(['FT. j=' num2str(j(it1)) '; k=' num2str(k(it2))]);
			ylabel('Re\{  \psi  ( f ) \}');
			xlabel('f');
		xlim([min(f_c(i_p)) max(f_c(i_p))])
		pbaspect([1.6 1 1])
		grid on

		subplot(2,2,4)
		plot_hmorlet_imag=plot(f_c(i_p),imag(hpnet(i_p,it2,it1)));
		set(plot_hmorlet_imag,'color',penn_red,'linewidth',2);
		ylabel('Im\{ \psi ( f ) \}'); xlabel('f');
		xlim([min(f_c(i_p)) max(f_c(i_p))])
		pbaspect([1.6 1 1])
		grid on
		
		subplot(2,2,1)
		plot_morlet_real=plot(u,real(pnet(:,it2,it1)));
		title(['j=' num2str(j(it1)) '; k=' num2str(k(it2))]);
		set(plot_morlet_real,'color',penn_blue,'linewidth',2);
		ylabel('Re\{ \psi (u)\}'); xlabel('u');
		xlim([min(u) max(u)])
		pbaspect([1.6 1 1])
		grid on

		subplot(2,2,3)
		plot_morlet_imag=plot(u,imag(pnet(:,it2,it1)));
		set(plot_morlet_imag,'color',penn_red,'linewidth',2);
		ylabel('Im\{ \psi (u) \}'); xlabel('u');
		xlim([min(u) max(u)])
		pbaspect([1.6 1 1])
		grid on
	end
end

%% Processing

[t,x]=sqpulse(s,2*s,fs);

figure()
plot_sqpulse=plot(t,x);
set(plot_sqpulse,'color',penn_purple,'linewidth',2)
title('Square pulse'); xlabel('t'); ylabel('x(t)');
xlim([min(t) max(t)]);
printpdf('scattering01','no-save')

[v,wtx]=convolve(t,x,u,p);
[f,hwtx,f_c,hwtx_c]=dft(wtx,fs);

i_p=find(abs(f_c)<fs/1000);

figure('units','normalized','outerposition',[0 0 1 1]);

subplot(2,2,1)
plot_wtx_real=plot(v,real(wtx));
set(plot_wtx_real,'color',penn_green,'linewidth',2);
title('Wavelet Transform of Square Pulse');
ylabel('Re\{ x * \psi \}'); xlabel('v');
xlim([min(v) max(v)])
pbaspect([1.6 1 1])
grid on

subplot(2,2,3)
plot_wtx_imag=plot(v,imag(wtx));
set(plot_wtx_imag,'color',penn_orange,'linewidth',2);
title('Wavelet Transform of Square Pulse');
ylabel('Im\{ x * \psi \}'); xlabel('v');
xlim([min(v) max(v)])
pbaspect([1.6 1 1])
grid on

subplot(2,2,2)
plot_hwtx_real=plot(f_c(i_p),real(hwtx_c(i_p)));
set(plot_hwtx_real,'color',penn_green,'linewidth',2);
title('DFT Wavelet Transform of Square Pulse');
	ylabel('Re\{ x *  \psi  ( f ) \}');...
	xlabel('f');
xlim([min(f_c(i_p)) max(f_c(i_p))])
pbaspect([1.6 1 1])
grid on

subplot(2,2,4)
plot_hwtx_imag=plot(f_c(i_p),imag(hwtx_c(i_p)));
set(plot_hwtx_imag,'color',penn_orange,'linewidth',2);
ylabel('Im\{ x * \psi ( f ) \}'); xlabel('f');
xlim([min(f_c(i_p)) max(f_c(i_p))])
pbaspect([1.6 1 1])
grid on

% Each row is the value of (k,j) to select.
sel_path=[k repmat(j(1),K,1);k repmat(j(2),K,1);k repmat(j(3),K,1)];
%sel_path=[k(1) j(1); k(2) j(2); k(3) j(3)];
%sel_path=[1 j(1); 1 j(2); 1 j(3)];
length_path=size(sel_path,1);

wtxnet=cell(length_path);
hwtxnet=cell(length_path);
hwtxnet_c=cell(length_path);

sel_j=sel_path(1,2);
sel_k=sel_path(1,1);
	
it1=find(j==sel_j);
it2=find(k==sel_k);

[v,wtxnet{1}]=convolve(t,x,u,pnet(:,it2,it1));
wtxnet{1}=abs(wtxnet{1});

for it=2:length_path
	sel_j=sel_path(it,2);
	sel_k=sel_path(it,1);
	
	it1=find(j==sel_j);
	it2=find(k==sel_k);
	
	[v,wtxnet{it}]=convolve(v,wtxnet{it-1},u,pnet(:,it2,it1));
	wtxnet{it}=abs(wtxnet{it});
	%wtxnet{it}=wtxnet{it}/sqrt(wtxnet{it}'*wtxnet{it});
	[f,hwtxnet{it},f_c,hwtxnet_c{it}]=dft(wtxnet{it},fs);
		
	i_p=find(abs(f_c)<fs/1000);
		
	figure('units','normalized','outerposition',[0 0 1 1]);
		
	subplot(2,2,2)
	plot_hwtxnet_real=plot(f_c(i_p),real(hwtxnet_c{it}(i_p)));
	set(plot_hwtxnet_real,'color',penn_green,'linewidth',2);
	title(['FT. j=' num2str(sel_path(it,2)) ...
		'; k=' num2str(sel_path(it,1))]);
		xlabel('f');
	xlim([min(f_c(i_p)) max(f_c(i_p))])
	pbaspect([1.6 1 1])
	grid on

	subplot(2,2,4)
	plot_hwtxnet_imag=plot(f_c(i_p),imag(hwtxnet_c{it}(i_p)));
	set(plot_hwtxnet_imag,'color',penn_orange,'linewidth',2);
		xlabel('f');
	xlim([min(f_c(i_p)) max(f_c(i_p))])
	pbaspect([1.6 1 1])
	grid on
		
	subplot(2,2,1)
	plot_wtxnet_real=plot(v,real(wtxnet{it}));
	title(['j=' num2str(sel_path(it,2)) ...
		'; k=' num2str(sel_path(it,1))]);
	set(plot_wtxnet_real,'color',penn_green,'linewidth',2);
	xlabel('v');
	xlim([min(v) max(v)])
	pbaspect([1.6 1 1])
	grid on

	subplot(2,2,3)
	plot_wtxnet_imag=plot(v,imag(wtxnet{it}));
	set(plot_wtxnet_imag,'color',penn_orange,'linewidth',2);
	xlabel('v');
	xlim([min(v) max(v)])
	pbaspect([1.6 1 1])
	grid on
end