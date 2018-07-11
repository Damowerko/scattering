function [weather,reliability] = availability()
rf = './pygSCN/Data/parsed/reliability_Ny.csv';
af = './pygSCN/Data/parsed/availability.csv';

reliability = csvread(rf, 1, 4)';
weather = csvread(af)';
end