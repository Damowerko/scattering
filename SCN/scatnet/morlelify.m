function f = morletify(f,sigma)
	f0 = f(1);
	
	f = f-f0*gabor(length(f),0,sigma,class(f));
end