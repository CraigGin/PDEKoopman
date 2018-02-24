function [u_0] = IC_Neumann_rand(n,a,b)
% Create initial conditions using random values 
% and boundary conditions u(-L/2) = a, u(L/2) = b
 
x = linspace(-L/2,L/2,n);

 u_0 = [0 randn(1,n-2) 0];
 u_0(1)= -1/3*(u_0(3)-4*u_0(2)+2*a*(x(2)-x(1)));
 u_0(n)= -1/3*(u_0(n-2)-4*u_0(n-1)-2*b*(x(2)-x(1)));

end

