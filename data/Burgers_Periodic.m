function [U_output] = Burgers_Periodic(mu,eps,x,t_output,dt_factor,u_0)
% Solves the Burgers' equation:
%   du/dt + eps*u*(du/dx) = mu*(d^2u/dx^2)
% Uses a second order FD method
% Periodic BC
% Inputs:
% mu = diffusion coeff/ viscosity
% eps = strength of advection
% x = spatial discretization
% t_output = time discretization for output
% dt_factor = divide the dt from the above time discretization when solving
% (i.e. refine the time step for numerical stability, but only output every 
%  dt_factor steps)
% u_0 = initial conditions

n = length(x);
dx = x(2)-x(1);
dt = (t_output(2)-t_output(1))/dt_factor;
t = t_output(1):dt:t_output(end);

% Solve with FD scheme
U = zeros(length(t),n);
U(1,:) = u_0;

for ti = 1:length(t)-1
    
    F_plus = (U(ti,1)^2 + U(ti,2)^2)/4;
    F_minus = (U(ti,1)^2 + U(ti,n)^2)/4;
    U(ti+1,1) = U(ti,1) + dt*(mu*(U(ti,2)-2*U(ti,1)+U(ti,n))/dx^2-eps*(F_plus-F_minus)/dx);
    
    F_plus = (U(ti,n)^2 + U(ti,1)^2)/4;
    F_minus = (U(ti,n)^2 + U(ti,n-1)^2)/4;
    U(ti+1,n) = U(ti,n) + dt*(mu*(U(ti,1)-2*U(ti,n)+U(ti,n-1))/dx^2-eps*(F_plus-F_minus)/dx);
    
    for xi = 2:n-1
        F_plus = (U(ti,xi)^2 + U(ti,xi+1)^2)/4;
        F_minus = (U(ti,xi)^2 + U(ti,xi-1)^2)/4;
        U(ti+1,xi) = U(ti,xi) + dt*(mu*(U(ti,xi+1)-2*U(ti,xi)+U(ti,xi-1))/dx^2-eps*(F_plus-F_minus)/dx);
    end
end

U_output = U(1:dt_factor:end,:);

end

