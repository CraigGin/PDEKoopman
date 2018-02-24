function [u_0] = IC_Dirichlet_rand(n,a,b)
% Create initial conditions using random values 
% and boundary conditions u(-L/2) = a, u(L/2) = b

u_0 = [a randn(1,n-2) b];

end

