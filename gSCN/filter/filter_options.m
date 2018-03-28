function [ options ] = filter_options()
fID = fopen('filter_options.txt');
options = struct();
line = fgetl(fID);
while line ~= -1
    C = strsplit(line,',');
    options.(C{1}) = eval(C{2});
    line = fgetl(fID);
end
fclose(fID);
end

