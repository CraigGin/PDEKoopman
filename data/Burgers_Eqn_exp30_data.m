clear; close all; clc
% Create data sets for Experiment 30
% All data comes from solutions to Burgers' Equation. 
% Training data:
%   Initial conditions:
%       White noise, Sines 
%   Solve from t = 0 to .1 in steps of 0.002
%   Diffusion coefficient is mu = 1
%   Strength of advection is eps = 10
%   128 spatial points in [-pi,pi)
% Validation data:
%    same structure as training data but with 20000 random lin combos
%

rng(1);

% Inputs (data)
exp_num = 'exp30';
n = 128; % Number of grid points
n_IC = 6000; % Number of initial conditions in each file
n_train = 20; % Number of training files
M = n_IC*n_train/2; % Samples from latin hypercube

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

%% Create vectors of random values for sines

% Sampling of A and phi
X = lhsdesign(M,2);
A_vect = X(:,1);
phi_vect = 2*pi*X(:,2);

% Sampling of omega
max_omega = 10;
cum_distrib = geocdf(0:max_omega,0.25);
cum_distrib = cum_distrib/cum_distrib(end);
numbs = rand(M,1);

omega_vect = zeros(M,1);

for k = 1:max_omega
    omega_vect = omega_vect + (numbs < cum_distrib(k));
end

omega_vect = 11 - omega_vect;

%% Loop over files
sine_ind = 1;
for train_num = 1:n_train
    data_set = ['train',num2str(train_num),'_x'];
    
    % Set Initial Conditions
    u_0 = zeros(n_IC,n);
    
    % White noise
    for j = 1:2:n_IC-1
        ut = zeros(1,n);
        ut(1) = randn;
        ut(2:n/2) = randn(1,n/2-1)+1i*randn(1,n/2-1);
        ut(n/2+1) = randn;
        ut(n/2+2:n) = conj(fliplr(ut(2:n/2)));
        u = ifft(ut);
        u_0(j,:) = u-mean(u);
    end
    
    % Sines
    for j = 2:2:n_IC
        u_0(j,:) = A_vect(sine_ind)*sin(omega_vect(sine_ind)*x+phi_vect(sine_ind));
        sine_ind = sine_ind+1;
    end
    
    % Solve Burgers' Equation
    Data = zeros(n_time*n_IC,n);
    for k = 1:n_IC
        U = Burgers_Periodic(mu,eps,x,t,dt_factor,u_0(k,:));
        Data(k*n_time-(n_time-1):k*n_time,:) = U;      
    end
    
    filename = strcat('Burgers_Eqn_',exp_num,'_',data_set,'.csv');
    dlmwrite(filename, Data, 'precision', '%.14f')

end




%% Validation Data
clear; close all; clc

% Inputs (data)
exp_num = 'exp30';
data_set = 'val_x';
n = 128; % Number of grid points
n_IC = 30000; % Number of initial conditions
M = n_IC/2; % Samples from latin hypercube

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

%% Create vectors of random values for sines

% Sampling of A and phi
X = lhsdesign(M,2);
A_vect = X(:,1);
phi_vect = 2*pi*X(:,2);

% Sampling of omega
max_omega = 10;
cum_distrib = geocdf(0:max_omega,0.25);
cum_distrib = cum_distrib/cum_distrib(end);
numbs = rand(M,1);

omega_vect = zeros(M,1);

for k = 1:max_omega
    omega_vect = omega_vect + (numbs < cum_distrib(k));
end

omega_vect = 11 - omega_vect;


% Set Initial Conditions
u_0 = zeros(n_IC,n);
    
% White noise
for j = 1:2:n_IC-1
    ut = zeros(1,n);
    ut(1) = randn;
    ut(2:n/2) = randn(1,n/2-1)+1i*randn(1,n/2-1);
    ut(n/2+1) = randn;
    ut(n/2+2:n) = conj(fliplr(ut(2:n/2)));
    u = ifft(ut);
    u_0(j,:) = u-mean(u);
end
  
sine_ind = 1;
% Sines
for j = 2:2:n_IC
    u_0(j,:) = A_vect(sine_ind)*sin(omega_vect(sine_ind)*x/(L/(2*pi))+phi_vect(sine_ind));
    sine_ind = sine_ind+1;
end
    
% Solve Burgers Equation
Data = zeros(n_time*n_IC,n);
for k = 1:n_IC
    U = Burgers_Periodic(mu,eps,x,t,dt_factor,u_0(k,:));
    Data(k*n_time-(n_time-1):k*n_time,:) = U;      
end
    
filename = strcat('Burgers_Eqn_',exp_num,'_',data_set,'.csv');
dlmwrite(filename, Data, 'precision', '%.14f')




