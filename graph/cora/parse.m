features = fopen('cora.content');

x_data = []; % data output
y_data = []; % label output

labels = {'Case_Based' 'Genetic_Algorithms' 'Neural_Networks' ...
    'Probabilistic_Methods' 'Reinforcement_Learning' 'Rule_Learning' 'Theory'};

tline = fgetl(features);
while ischar(tline)
    whole = split(tline);
    x = whole(2:end-1);
    y = whole(end);
    
    x_data = [x_data str2num(char(x))]; % cast to int and append
    
    y = find(cellfun(@(s) ~isempty(strfind(s,y)),labels)); % to class index
    y_data = [y_data y];
    
    tline = fgetl(features); 
end
fclose(features);

covariance = cov(transpose(x_data));
save('data', 'x_data', 'y_data', 'covariance');