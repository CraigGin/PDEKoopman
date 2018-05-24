clear all; close all; clc
% Create data sets for Experiment 16
% All data comes from solutions to the Heat Equation. 
% Training data:
%   Initial conditions:
%       Normally distributed random weights in Fourier space, complex
%       conjugate pairs in order to enforce real values
%   Solve from t = 0 to 0.04 in steps of 0.01
%   Diffusion coefficient is D = 1
%   128 spatial points in [-pi,pi)
% Validation data:
%    same structure as training data but with 5 random lin combos
%

% Inputs
D = 1; % Diffusion coefficient
L = 2*pi; % Length of domain
n = 128; % Number of grid points
T = 0.04;  % End time
n_time = 5; % Number of time steps
n_IC = 400; % number of initial conditions
exp_num = 'exp19';
data_set = 'train1_x';
% Specify training data, validation data, or testing data in file name

    
% Discretize x
x = linspace(-L/2,L/2,n+1);
x = x(1:n);
    
% Discretize t
t = linspace(0,T,n_time);


% Set Initial Conditions
u_0 = zeros(n_IC,n);
rng(1);
for j = 1:n_IC
    ut = zeros(1,n);
    ut(1) = randn;
    ut(2:n/2) = randn(1,n/2-1)+i*randn(1,n/2-1);
    ut(n/2+1) = randn;
    ut(n/2+2:n) = conj(fliplr(ut(2:n/2)));
    u_0(j,:) = ifft(ut);
end

% Solve Heat Equation
Data = zeros(n_time*n_IC,n);
for i = 1:n_IC
    U = HeatEqn_FT(D,L,x,t,u_0(i,:)); % Periodic BC (FFT)
    Data(i*n_time-(n_time-1):i*n_time,:) = U;
end

filename = strcat('Heat_Eqn_',exp_num,'_',data_set,'.csv');
dlmwrite(filename, Data, 'precision', '%.14f')


clear all; close all; clc
% Inputs
D = 1; % Diffusion coefficient
L = 2*pi; % Length of domain
n = 128; % Number of grid points
T = 0.04;  % End time
n_time = 5; % Number of time steps
n_IC = 100; % number of initial conditions
exp_num = 'exp19';
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
    ut = zeros(1,n);
    ut(1) = randn;
    ut(2:n/2) = randn(1,n/2-1)+i*randn(1,n/2-1);
    ut(n/2+1) = randn;
    ut(n/2+2:n) = conj(fliplr(ut(2:n/2)));
    u_0(j,:) = ifft(ut);
end

% Solve Heat Equation
Data = zeros(n_time*n_IC,n);
for i = 1:n_IC
    U = HeatEqn_FT(D,L,x,t,u_0(i,:)); % Periodic BC (FFT)
    Data(i*n_time-(n_time-1):i*n_time,:) = U;
end

filename = strcat('Heat_Eqn_',exp_num,'_',data_set,'.csv');
dlmwrite(filename, Data, 'precision', '%.14f')








