%% Availability plot
[A,R] = availability();

fprintf('Number of datapoints: %d\n', length(R))
fprintf('Number of datapoints with outage: %d (%f%%)\n', sum(R), sum(R)/length(R)*100)
fprintf('Ratio: %d : %d\n', sum(R == 1) , sum(R == 0))

figure;
scatter(1:length(R),R)
title("Reliability data")

figure
imagesc(A)
title("Availability, yellow is available")

figure
plot(sum(A,1))
title("Total available stations")

%% Greedy Algorithim
[x,y,N,M] = pick_greedy();
%% Plot picking results
% total stations
figure
plot(0:N, sum(y,1))
xlabel('Numer of stations')

% label ratio
figure
r = repmat(R',[1 size(y,2)]);
positive = r .* y; % both y is available and label
positive = sum(positive,1);
negative = sum(y,1);
negative = negative - positive;
plot(0:N,positive, 0:N, negative)
legend('positive', 'negative') 
xlabel('Numer of stations')
