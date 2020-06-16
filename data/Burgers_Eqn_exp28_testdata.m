clear all; close all; clc
% Create data sets for Experiment 28
% All data comes from solutions to Burgers' Equation. 
% Testing data:
%   Initial conditions:
%       White noise, sine, square wave (like training data) as well as
%       Gaussian, triangle wave
%   Solve from t = 0 to 0.1 in steps of 0.002
%   Diffusion coefficient is mu = 1
%   Strength of advection is eps = 10
%   128 spatial points in [-pi,pi)


rng(1012);

% Inputs (data)
exp_num = 'exp28';
n = 128; % Number of grid points
n_IC = 1000; % Number of initial conditions in each file (of each type)
M = n_IC; % Samples from latin hypercube

% Inputs (Burgers')
eps = 10; % strength of advection
mu = 1; % viscosity in Burgers'
L = 2*pi; % Length of domain
dt = 0.002; % Size of time step for data
n_time = 51; % Number of time steps
T = dt*(n_time-1);  % End time
dt_factor = 1000; % Divide dt by this factor for numerical stability when solving


% Discretize x
x = linspace(-L/2,L/2,n+1);
x = x(1:n);

% Discretize t
t = linspace(0,T,n_time);

%% Create vectors of random values 

% Sampling of A and phi for sines
X = lhsdesign(M,2);
A_vect = X(:,1);
phi_vect = 2*pi*X(:,2);

% Sampling of omega for sines
max_omega = 10;
cum_distrib = geocdf(0:max_omega,0.25);
cum_distrib = cum_distrib/cum_distrib(end);
numbs = rand(M,1);

omega_vect = zeros(M,1);

for k = 1:max_omega
    omega_vect = omega_vect + (numbs < cum_distrib(k));
end

omega_vect = 11 - omega_vect;


% Create vectors of random values for square waves
% Sampling of A, c, and w
X = lhsdesign(M,3);
A2_vect = X(:,1);
c2_vect = L*X(:,2) - L/2;
w2_vect = (L-4*(x(2)-x(1)))*X(:,3)+2*(x(2)-x(1)); 

% Sampling of mean and sigma for Gaussians
X = lhsdesign(M,2);
mean_vect = X(:,1);
sigma_vect = (1-(x(2)-x(1)))*X(:,2)+(x(2)-x(1));


%% Create vectors of random values for triangle waves
% Sampling of A, c, and w
X = lhsdesign(M,3);
A3_vect = X(:,1);
c3_vect = L*X(:,2) - L/2;
w3_vect = (L-4*(x(2)-x(1)))*X(:,3)+2*(x(2)-x(1));  


% White noise
Data = zeros(n_time*n_IC,n);
for k = 1:n_IC
    ut = zeros(1,n);
    ut(1) = randn;
    ut(2:n/2) = randn(1,n/2-1)+1i*randn(1,n/2-1);
    ut(n/2+1) = randn;
    ut(n/2+2:n) = conj(fliplr(ut(2:n/2)));
    u = ifft(ut);
    u_0 = u-mean(u);
    U = Burgers_Periodic(mu,eps,x,t,dt_factor,u_0);
    Data(k*n_time-(n_time-1):k*n_time,:) = U;      
end
  
data_set = 'test1_x';
filename = strcat('Burgers_Eqn_',exp_num,'_',data_set,'.csv');
dlmwrite(filename, Data, 'precision', '%.14f')

% Sines
Data = zeros(n_time*n_IC,n);
for k = 1:n_IC
    u_0 = A_vect(k)*sin(omega_vect(k)*x+phi_vect(k));
    U = Burgers_Periodic(mu,eps,x,t,dt_factor,u_0);
    Data(k*n_time-(n_time-1):k*n_time,:) = U;   
end

data_set = 'test2_x';
filename = strcat('Burgers_Eqn_',exp_num,'_',data_set,'.csv');
dlmwrite(filename, Data, 'precision', '%.14f')

% Square waves
Data = zeros(n_time*n_IC,n);
for k = 1:n_IC
    u = A2_vect(k)*(abs(x-c2_vect(k))<w2_vect(k)/2 | abs(x+2*pi-c2_vect(k)) < w2_vect(k)/2 | abs(x-2*pi-c2_vect(k)) < w2_vect(k)/2);
    u_0 = u - mean(u);
    U = Burgers_Periodic(mu,eps,x,t,dt_factor,u_0);
    Data(k*n_time-(n_time-1):k*n_time,:) = U;   
end

data_set = 'test3_x';
filename = strcat('Burgers_Eqn_',exp_num,'_',data_set,'.csv');
dlmwrite(filename, Data, 'precision', '%.14f')

% Gaussians
Data = zeros(n_time*n_IC,n);
for k = 1:n_IC
    Gmean = mean_vect(k);
    sigma = sigma_vect(k);
    u = 1/sqrt(2*pi*sigma^2)*exp(-(x-Gmean).^2/(2*sigma^2));
    u = u';
    u_0 = u-mean(u);
    U = Burgers_Periodic(mu,eps,x,t,dt_factor,u_0);
    Data(k*n_time-(n_time-1):k*n_time,:) = U; 
end

data_set = 'test4_x';
filename = strcat('Burgers_Eqn_',exp_num,'_',data_set,'.csv');
dlmwrite(filename, Data, 'precision', '%.14f')
    
% Triangle Waves
Data = zeros(n_time*n_IC,n);
for k = 1:n_IC
    u1 = 2*A3_vect(k)/w3_vect(k)*(x-c3_vect(k)+w3_vect(k)/2).*(-w3_vect(k)/2<=x-c3_vect(k) & x-c3_vect(k) <=0)...
         +2*A3_vect(k)/w3_vect(k)*(x-2*pi-c3_vect(k)+w3_vect(k)/2).*(-w3_vect(k)/2<=x-2*pi-c3_vect(k) & x-2*pi-c3_vect(k)<=0);     
    u2 = -2*A3_vect(k)/w3_vect(k)*(x-c3_vect(k)-w3_vect(k)/2).*(w3_vect(k)/2>x-c3_vect(k) & x-c3_vect(k) >0)...
         + -2*A3_vect(k)/w3_vect(k)*(x+2*pi-c3_vect(k)-w3_vect(k)/2).*(w3_vect(k)/2>x+2*pi-c3_vect(k) & x+2*pi-c3_vect(k)>0);  
    u = u1+u2;
    u_0 = u - mean(u);
    U = Burgers_Periodic(mu,eps,x,t,dt_factor,u_0);
    Data(k*n_time-(n_time-1):k*n_time,:) = U; 
end
    
data_set = 'test5_x';
filename = strcat('Burgers_Eqn_',exp_num,'_',data_set,'.csv');
dlmwrite(filename, Data, 'precision', '%.14f')





