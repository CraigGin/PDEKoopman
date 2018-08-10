clear all; close all; clc
% Create data sets for Experiment 31
% Data is white noise in pairs (for training identity network)
%
rng(1);

% Inputs
n = 128; % Number of grid points
n_IC = 10000; % number of initial conditions
exp_num = 'exp31';
data_set = 'train1_x';
    
Data = zeros(2*n_IC,n);
for j = 1:n_IC
    ut = zeros(1,n);
    ut(1) = randn;
    ut(2:n/2) = randn(1,n/2-1)+i*randn(1,n/2-1);
    ut(n/2+1) = randn;
    ut(n/2+2:n) = conj(fliplr(ut(2:n/2)));
    Data(2*j-1:2*j,:) = [ifft(ut); ifft(ut)];
end

filename = strcat('Heat_Eqn_',exp_num,'_',data_set,'.csv');
dlmwrite(filename, Data, 'precision', '%.14f')




%%
clear all; close all; clc

rng(100000);

% Inputs
n = 128; % Number of grid points
n_IC = 2500; % number of initial conditions
exp_num = 'exp31';
data_set = 'val_x';

Data = zeros(2*n_IC,n);
for j = 1:n_IC
    ut = zeros(1,n);
    ut(1) = randn;
    ut(2:n/2) = randn(1,n/2-1)+i*randn(1,n/2-1);
    ut(n/2+1) = randn;
    ut(n/2+2:n) = conj(fliplr(ut(2:n/2)));
    Data(2*j-1:2*j,:) = [ifft(ut); ifft(ut)];
end

filename = strcat('Heat_Eqn_',exp_num,'_',data_set,'.csv');
dlmwrite(filename, Data, 'precision', '%.14f')








