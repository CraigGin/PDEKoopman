clear; close all; clc
% Create data sets for Experiment 32
% All data comes from solutions to Burgers' Equation. 
% Training data:
%   Initial conditions:
%       White noise only 
%   Solve from t = 0 to .1 in steps of 0.002
%   Diffusion coefficient is mu = 1
%   Strength of advection is eps = 10
%   128 spatial points in [-pi,pi)
% Validation data:
%    same structure as training data but with 30000 random lin combos
%

rng(1);

% Inputs (data)
exp_num = 'exp32';
n = 128; % Number of grid points
n_IC = 6000; % Number of initial conditions in each file
n_train = 20; % Number of training files

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


%% Loop over files
for train_num = 1:n_train
    data_set = ['train',num2str(train_num),'_x'];
    
    Data = zeros(n_time*n_IC,n);
  
    % White noise
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
    
    filename = strcat('Burgers_Eqn_',exp_num,'_',data_set,'.csv');
    dlmwrite(filename, Data, 'precision', '%.14f')

end




%% Validation Data
clear; close all; clc

% Inputs (data)
exp_num = 'exp32';
data_set = 'val_x';
n = 128; % Number of grid points
n_IC = 30000; % Number of initial conditions

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
    
% White noise
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
  

filename = strcat('Burgers_Eqn_',exp_num,'_',data_set,'.csv');
dlmwrite(filename, Data, 'precision', '%.14f')




