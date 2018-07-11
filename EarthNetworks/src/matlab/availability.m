function [weather,reliability] = availability(location)
rf = strcat('EarthNetworks/Data/parsed/', location, '/reliability.csv');
af = strcat('EarthNetworks/Data/parsed/', location, '/availability.csv');

reliability = csvread(rf, 1, 4)';
weather = csvread(af)';
end