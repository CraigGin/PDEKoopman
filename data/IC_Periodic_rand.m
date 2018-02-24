function [u_0] = IC_Periodic_rand(n)
% Create initial conditions using random coeffiecients 
% of the Fourier Series

u_t = [randn(1,n)];
u_0 = ifft(u_t);

% OR 

% Random pointwise values

% u_0 = [randn(1,n-1) 0];
% u_0(n)= u_0(2)-u_0(1)+u_0(n-1);

end

