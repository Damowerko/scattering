function [data, labels, n_samples] = import_data(fields, Ix, missing_thres)
%IMPORT_DATA Import data for field.
%   [D, L, N] = IMPORT_DATA(FIELD, IX, MISSING_THRES) imports data about
%       FIELDS condirering stations IX and ignoring stations with.
%       Outputs a cell array of DATA for each field and vector of LABELS.
%       N is the number of samples.
for i = 1:length(fields)
    field = fields{i};
    data_raw = importfile(['Data/parsed/' field '.csv'])';
    data{i} = data_raw(Ix,:); % every row is a station, every column is 
        % time instant, for all those stations.
    data{i} = data{i} / max(max(data{i})); % normalize data fields
    
    missing = isnan(data{i});
    I(i,:) = sum(missing,1) <= missing_thres; % Which time instants have less than
        % a specific number of nans (those time instant with many nans should
        % be discarded)
end
I = sum(I,1) == size(I,1); % exclude all datapoints where at least one station is above the missing threshold

for i = 1:length(fields)
    data{i} = data{i}(:,I);
end
labels_raw = csvread('Data/parsed/reliability_Ny.csv', 1, 4)';
labels = labels_raw(I);
n_samples = size(data{1},2);
end

