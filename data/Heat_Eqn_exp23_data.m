clear all; close all; clc
% Create data sets for Experiment 23
% All data comes from solutions to the Heat Equation. 
% Training data:
%   Initial conditions:
%       4000 Random lin. combos of the first 50 sine modes - normalized
%   Solve from t = 0 to 0.24 in steps of 0.01
%   Diffusion coefficient is D = 1
%   128 spatial points in [-pi,pi)
% Validation data:
%    same structure as training data but with 500 random lin combos
%
rng(1);
for train_num = 1:10


% Inputs
D = 1; % Diffusion coefficient
L = 2*pi; % Length of domain
n = 128; % Number of grid points
T = 0.24;  % End time
n_time = 25; % Number of time steps
n_IC = 400; % number of initial conditions
N = 50; % number of modes
exp_num = 'exp23';
data_set = ['train',num2str(train_num),'_x'];
% Specify training data, validation data, or testing data in file name

    
% Discretize x
x = linspace(-L/2,L/2,n+1);
x = x(1:n);
    
% Discretize t
t = linspace(0,T,n_time);


% Set Initial Conditions
u_0 = zeros(n_IC,n);

for j = 1:n_IC
    r = randn(1,N);
    for k = 1:N
        u_0(j,:) = u_0(j,:) + r(k)*sin(k*x);
    end 
    u_0(j,:) = u_0(j,:)/norm(u_0(j,:),Inf);
end


% Solve Heat Equation
Data = zeros(n_time*n_IC,n);
for i = 1:n_IC
    U = HeatEqn_FT(D,L,x,t,u_0(i,:)); % Periodic BC (FFT)
    Data(i*n_time-(n_time-1):i*n_time,:) = U;
end

filename = strcat('Heat_Eqn_',exp_num,'_',data_set,'.csv');
dlmwrite(filename, Data, 'precision', '%.14f')


end













%%
clear all; close all; clc
% Inputs
D = 1; % Diffusion coefficient
L = 2*pi; % Length of domain
n = 128; % Number of grid points
T = 0.24;  % End time
n_time = 25; % Number of time steps
n_IC = 500; % number of initial conditions
N = 50; % number of modes
exp_num = 'exp23';
data_set = 'val_x';
% Specify training data, validation data, or testing data in file name

    
% Discretize x
x = linspace(-L/2,L/2,n+1);
x = x(1:n);
    
% Discretize t
t = linspace(0,T,n_time);


% Set Initial Conditions
u_0 = zeros(n_IC,n);
rng(1000);
for j = 1:n_IC
    r = randn(1,N);
    for k = 1:N
        u_0(j,:) = u_0(j,:) + r(k)*sin(k*x);
    end 
    u_0(j,:) = u_0(j,:)/norm(u_0(j,:),Inf);
end


% Solve Heat Equation
Data = zeros(n_time*n_IC,n);
for i = 1:n_IC
    U = HeatEqn_FT(D,L,x,t,u_0(i,:)); % Periodic BC (FFT)
    Data(i*n_time-(n_time-1):i*n_time,:) = U;
end

filename = strcat('Heat_Eqn_',exp_num,'_',data_set,'.csv');
dlmwrite(filename, Data, 'precision', '%.14f')








