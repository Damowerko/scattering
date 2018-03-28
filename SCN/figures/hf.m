clear
load hf;
figure;
hold on;
for n = 1:size(hf,2)
    plot(hf(:,n));
end