clear all; close all; clc
% Create data sets for Experiment 2
% Input data is white noise, sines, and square waves - all shifted to have 
%   zero mean, and output data is Cole-Hopf transform of input scaled 
%   to be in [-1,1].
% Training data:
%   Inputs:
%       White noise, sines, square waves
%   Outputs:
%       Cole-Hopf transform of input
%   Diffusion coefficient is mu = 0.01
%   Strength of advection is eps = 1
%   128 spatial points in [-pi,pi)
% Validation data:
%    same structure as training data
%

% Inputs (data)
exp_num = 'exp2';
n = 128; % Number of grid points
n_train = 90000; % Number of training vectors
M = n_train/3; % Samples from latin hypercube

% Inputs (Burgers')
eps = 1; % strength of advection
mu = 0.01; % viscosity in Burgers'
L = 2*pi; % Length of domain

% Discretize x
x = linspace(-L/2,L/2,n+1);
x = x(1:n);


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

%% Create vectors of random values for square waves
% Sampling of A, c, and w
X = lhsdesign(M,3);
A2_vect = X(:,1);
c_vect = 2*pi*X(:,2) - pi;
w_vect = 2*pi*X(:,3); 


rng(1);


% Create Training Data
input = zeros(n_train,n);
output = zeros(n_train,n);

% White noise
for j = 1:3:n_train-2
 
    % Create white noise as input
    ut = zeros(1,n);
    ut(1) = randn;
    ut(2:n/2) = randn(1,n/2-1)+i*randn(1,n/2-1);
    ut(n/2+1) = randn;
    ut(n/2+2:n) = conj(fliplr(ut(2:n/2)));
    u = ifft(ut);
    u = u-mean(u);
    input(j,:) = u;
    
    % Do Cole-Hopf for output
    int_vec = cumtrapz(x,u);
    v = exp(-eps*int_vec/(2*mu));
    v = v/norm(v,inf);
    
    output(j,:) = v;
end


% Sines
sine_ind = 1;
for j = 2:3:n_train-1
 
    % Create sine as input
    u = A_vect(sine_ind)*sin(omega_vect(sine_ind)*x+phi_vect(sine_ind));
    sine_ind = sine_ind+1;
    input(j,:) = u;
    
    % Do Cole-Hopf for output
    int_vec = cumtrapz(x,u);
    v = exp(-eps*int_vec/(2*mu));
    v = v/norm(v,inf);
    
    output(j,:) = v;
end


% Square waves
square_ind = 1;
for j = 3:3:n_train
 
    % Create square wave as input
    u = A2_vect(square_ind)*(abs(x-c_vect(square_ind))<w_vect(square_ind)/2 | abs(x+2*pi-c_vect(square_ind)) < w_vect(square_ind)/2 | abs(x-2*pi-c_vect(square_ind)) < w_vect(square_ind)/2);
    u = u-mean(u);
    square_ind = square_ind+1;
    input(j,:) = u;
    
    % Do Cole-Hopf for output
    int_vec = cumtrapz(x,u);
    v = exp(-eps*int_vec/(2*mu));
    v = v/norm(v,inf);
   
    output(j,:) = v;
end

% Save data  
input_data_set = 'train1_x';
output_data_set = 'train1_y';

filename = strcat('ColeHopf_',exp_num,'_',input_data_set,'.csv');
dlmwrite(filename, input, 'precision', '%.14f')

filename = strcat('ColeHopf_',exp_num,'_',output_data_set,'.csv');
dlmwrite(filename, output, 'precision', '%.14f')





%% Validation Data
clear all; close all; clc
% Inputs (data)
exp_num = 'exp2';
n = 128; % Number of grid points
n_val = 12000; % Number of training vectors
M = n_val/3; % Samples from latin hypercube

% Inputs (Burgers')
eps = 1; % strength of advection
mu = 0.01; % viscosity in Burgers'
L = 2*pi; % Length of domain

% Discretize x
x = linspace(-L/2,L/2,n+1);
x = x(1:n);


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

%% Create vectors of random values for square waves
% Sampling of A, c, and w
X = lhsdesign(M,3);
A2_vect = X(:,1);
c_vect = 2*pi*X(:,2) - pi;
w_vect = 2*pi*X(:,3); 


% Create Training Data
input = zeros(n_val,n);
output = zeros(n_val,n);

% White noise
for j = 1:3:n_val-2
 
    % Create white noise as input
    ut = zeros(1,n);
    ut(1) = randn;
    ut(2:n/2) = randn(1,n/2-1)+i*randn(1,n/2-1);
    ut(n/2+1) = randn;
    ut(n/2+2:n) = conj(fliplr(ut(2:n/2)));
    u = ifft(ut);
    u = u-mean(u);
    input(j,:) = u;
    
    % Do Cole-Hopf for output
    int_vec = cumtrapz(x,u);
    v = exp(-eps*int_vec/(2*mu));
    v = v/norm(v,inf);
    
    output(j,:) = v;
end


% Sines
sine_ind = 1;
for j = 2:3:n_val-1
 
    % Create sine as input
    u = A_vect(sine_ind)*sin(omega_vect(sine_ind)*x+phi_vect(sine_ind));
    sine_ind = sine_ind+1;
    input(j,:) = u;
    
    % Do Cole-Hopf for output
    int_vec = cumtrapz(x,u);
    v = exp(-eps*int_vec/(2*mu));
    v = v/norm(v,inf);
    
    output(j,:) = v;
end


% Square waves
square_ind = 1;
for j = 3:3:n_val
 
    % Create square wave as input
    u = A2_vect(square_ind)*(abs(x-c_vect(square_ind))<w_vect(square_ind)/2 | abs(x+2*pi-c_vect(square_ind)) < w_vect(square_ind)/2 | abs(x-2*pi-c_vect(square_ind)) < w_vect(square_ind)/2);
    u = u-mean(u);
    square_ind = square_ind+1;
    input(j,:) = u;
    
    % Do Cole-Hopf for output
    int_vec = cumtrapz(x,u);
    v = exp(-eps*int_vec/(2*mu));
    v = v/norm(v,inf);
    
    output(j,:) = v;
end

% Save data  
input_data_set = 'val_x';
output_data_set = 'val_y';

filename = strcat('ColeHopf_',exp_num,'_',input_data_set,'.csv');
dlmwrite(filename, input, 'precision', '%.14f')

filename = strcat('ColeHopf_',exp_num,'_',output_data_set,'.csv');
dlmwrite(filename, output, 'precision', '%.14f')
































