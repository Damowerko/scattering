function [ options ] = options()
options = struct();
fID = fopen('options.txt');
line = fgetl(fID);
while line ~= -1
    C = strsplit(line,',');
    options.(C{1}) = eval(C{2});
    line = fgetl(fID);
end
fclose(fID);
options.filter = filter_options();
end

