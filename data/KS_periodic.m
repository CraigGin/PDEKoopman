function [uu] = KS_Periodic(x,tmax,ntime,u)

% Kuramoto-Sivashinsky equation (from Trefethen)
% u_t = -u*u_x - u_xx - u_xxxx,  periodic BCs 

%Spatial grid and initial condition:
N = length(x);
x = x.';
u = u.';
v = fft(u);

% % % % % %
% Precompute various ETDRK4 scalar quantities:
h = 0.025;
k = (2*pi/(x(end)-2*x(1)+x(2)))*[0:N/2-1 0 -N/2+1:-1]';
L = k.^2 - k.^4;
E = exp(h*L); E2 = exp(h*L/2);
M = 64;
r = exp(1i*pi*((1:M)-.5)/M);
LR = h*L(:,ones(M,1)) + r(ones(N,1),:);
Q = h*real(mean( (exp(LR/2)-1)./LR ,2)); 
f1 = h*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2)); 
f2 = h*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
f3 = h*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));

% Main time-stepping loop:
uu = u; tt = 0;
nmax = round(tmax/h); nplt = floor((tmax/(ntime-1))/h); g = -0.5i*k;
for n = 1:nmax
    t = n*h;
    Nv = g.*fft(real(ifft(v)).^2);
    a = E2.*v + Q.*Nv;
    Na = g.*fft(real(ifft(a)).^2);
    b = E2.*v + Q.*Na;
    Nb = g.*fft(real(ifft(b)).^2);
    c = E2.*a + Q.*(2*Nb-Nv);
    Nc = g.*fft(real(ifft(c)).^2);
    v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3; 
    if mod(n,nplt)==0
        u = real(ifft(v));
        uu = [uu,u]; tt = [tt,t]; 
    end
end


