clear all; close all; clc
% Read and process data from Deep Learning Algorithm for Koopman

% Inputs and load .mat file
n_IC = 10; % Number of initial conditions
folder_name = 'exp2a';  % Type of boundary conditions
file_prefix = strcat('./',folder_name,'/Heat_Eqn_IC_',num2str(n_IC),'_',folder_name,'_2018_03_08_08_55_10_735488_'); % Prefix for file names
%file_prefix = strcat('./',folder_name,'/Heat_Eqn_IC_',num2str(n_IC),'_BC_periodic_2018_03_07_22_28_26_591671_'); % Prefix for file names
load(strcat(file_prefix,'model.mat')) % Load the model parameters
n_x = widths(1); % Number of spatial grid points
data_file = strcat('./data/',data_name,'_train1_x.csv'); % Path of data file


% Read in parameters (matrices E and vectors b)
[WE,bE] = read_params(file_prefix,'E',num_encoder_weights);
%[WO,bO] = read_params(file_prefix,'O',num_omega_weights);
[WD,bD] = read_params(file_prefix,'D',num_decoder_weights);

% Read in data
data = read_data(data_file,len_time,n_IC,n_x);

% Choose input of network
f = reshape(data(1,1,:),[n_x,1]);

y = network(f,WE,bE); % Encoder
f_hat = network(y,WD,bD); % Decoder

% Comparison plot
x = linspace(-pi,pi,n_x+1);
x = x(1:n_x);

figure(1)
plot(x,f)
hold on
plot(x,f_hat,'--r')
hold off

figure(2)
plot(y)

% norm(f-f_hat)/norm(f);

% y_mat = [];
% for j = 1:10
%     
%     % Choose input of network
%     f = reshape(data(1,j,:),[n_x,1]);
% 
%     y_mat(:,j) = network(f,WE,bE); % Encoder
% end
% 
% f = 0*x;
% w = [0 1 0 0 0 0 0 0 0 0];
% for j = 1:10
%     f = f + w(j)*sin(j*x);
% end
% 
% y = network(f',WE,bE);
% f_hat = network(y,WD,bD); % Decoder
% 
% y2 = zeros(10,1);
% for j = 1:10
%     y2 = y2 + w(j)*y_mat(:,j);
% end
% close all;
% 
% figure(3)
% plot(x,f)
% hold on
% plot(x,f_hat,'--r')
% hold off
% 
% figure(4)
% plot(y)
% hold on
% plot(y2,'--r')
% hold off