function [x,y,N,M] = pick_greedy(location)
[A,~] = availability(location);

N = size(A,1);
M = size(A,2);

x = zeros(N,N+1); % each column is a possible vector
y = zeros(M,N+1); % each column represents available datapoints for n stations starting with 0
y(:,1) = 1;

for n = 1:N
   max_x = zeros(N,1);
   max_stations = zeros(M,1);
       for m = 1:N
       new_x = x(:,n);
       new_x(m) = 1;
       if sum(new_x) == sum(x(:,n))+1
           stations = sum(A(logical(new_x),:),1) == n;
           if sum(stations) > sum(max_stations)
               max_stations = stations;
               max_x = new_x;
           end
       end
   end
   x(:,n+1) = max_x;
   y(:,n+1) = max_stations;
end
end

