function [ x ] = sqpulse( T_0, T, f_s )
N = T * f_s;
M = T_0 * f_s;
x = 1/sqrt(M) * [ones(M,1); zeros(N-M,1)];
end

