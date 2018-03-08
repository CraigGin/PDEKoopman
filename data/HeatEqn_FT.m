function [U] = HeatEqn_FT(D,L,x,t,u_0)
% Solves the 1-D Heat equation using the Fourier Transform
% Periodic BC
% Inputs:
% D = diffusion coeff
% L = length of spatial domain
% x = spatial discretization
% t = time discretization
% u_0 = initial conditions

n = length(x);

% Solve with Fourier Transform
u_0t = fft(u_0); % IC

k = (2*pi/L)*[0:n/2-1 -n/2:-1]; % vector of wave numbers

% Evolve in time and Inverse FT at each time
for ti = 1:length(t)
    U(ti,:) = ifft(exp(-D*k.^2*t(ti)).*u_0t);
end

end

