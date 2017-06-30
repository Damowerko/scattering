function [x] = tpulse(T_0, T, f_s)

N = T*f_s;
M = T_0*f_s;

x = [(0:((M/2)-1)) (M-1)*ones(1,(M/2))-((M/2):(M-1)) zeros(1,(N-M))]';
x = x/norm(x);

end
