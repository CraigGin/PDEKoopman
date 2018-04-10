clear all; close all; clc
% Create data sets for autoencoder

% %% Gaussians
% % Inputs
% D = 3; % Diffusion coefficient
% L = 30; % Length of domain
% n = 128; % Number of grid points
% T = 1;  % End time
% n_time = 101; % Number of time steps
% n_IC = 10; % number of initial conditions
% exp_num = 'exp7a'; % Folder for eventual results
% data_set = 'train1_x';
% % Specify training data, validation data, or testing data in file name
% 
%     
% % Discretize x
% x = linspace(-L/2,L/2,n+1);
% x = x(1:n);
%     
% % Discretize t
% t = linspace(0,1,n_time);
% 
% 
% % Set Initial Conditions
% u_0 = zeros(n_IC,n);
% sigma = .2:.2:2;
% for i = 1:n_IC
%     u_0(i,:) = (1/(sigma(i)*sqrt(2*pi)))*exp(-(1/2)*(x/sigma(i)).^2);   
% end
% 
% % Solve Heat Equation
% Data = zeros(n_time*n_IC,n);
% 
% for i = 1:10
%     U = HeatEqn_FT(D,L,x,t,u_0(i,:)); 
%     Data(i*n_time-(n_time-1):i*n_time,:) = U;
% end
% 
% filename = strcat('Heat_Eqn_',exp_num,'_',data_set,'.csv');
% dlmwrite(filename, Data, 'precision', '%.14f')
% 
% 
% n_IC2 = 4; % number of initial conditions
% exp_num = 'exp7a'; % Folder for eventual results
% data_set = 'val_x';
% 
% % Set Initial Conditions
% 
% u_02 = zeros(n_IC2,n);
% 
% for i = 1:n_IC2
%     r=rng(n_IC+i);
%     r = rand(1,n_IC);
%     r = r/sum(r);
%     for j = 1:n_IC
%         u_02(i,:) = u_02(i,:) + r(j)*u_0(j,:);
%     end
% end
% 
% % Solve Heat Equation
% Data2 = zeros(n_time*n_IC2,n);
% 
% for i = 1:n_IC2
%     U = HeatEqn_FT(D,L,x,t,u_02(i,:)); 
%     Data2(i*n_time-(n_time-1):i*n_time,:) = U;
% end
% 
% filename = strcat('Heat_Eqn_',exp_num,'_',data_set,'.csv');
% dlmwrite(filename, Data2, 'precision', '%.14f')

%% Sines 

% clear all; close all; clc
% % Create data sets for autoencoder
% 
% % Inputs
% D = .01; % Diffusion coefficient
% L = 2*pi; % Length of domain
% n = 128; % Number of grid points
% T = 1;  % End time
% n_time = 101; % Number of time steps
% n_IC = 10; % number of initial conditions
% exp_num = 'exp6f';
% data_set = 'train1_x';
% % Specify training data, validation data, or testing data in file name
% 
%     
% % Discretize x
% x = linspace(-L/2,L/2,n+1);
% x = x(1:n);
%     
% % Discretize t
% t = linspace(0,1,n_time);
% 
% 
% % Set Initial Conditions
% 
% u_0 = zeros(n_IC,n);
% for i = 1:n_IC
%     u_0(i,:) = exp(D*i^2)*IC_Periodic(x,i);
% end
% 
% % for i = n_IC-4:n_IC
% %     rng(i);
% %     r = rand(1,10);
% %     r = r/sum(r);
% %     for j = 1:10
% %         u_0(i,:) = u_0(i,:) + r(j)*sin(j*x);
% %     end
% % end
% 
% % Solve Heat Equation
% Data = zeros(n_time*n_IC,n);
% for i = 1:n_IC
%     U = HeatEqn_FT(D,L,x,t,u_0(i,:)); % Periodic BC (FFT)
%     Data(i*n_time-(n_time-1):i*n_time,:) = U;
% end
% 
% filename = strcat('Heat_Eqn_',exp_num,'_',data_set,'.csv');
% dlmwrite(filename, Data, 'precision', '%.14f')
% 
% 
% clear all; close all; clc
% % Create data sets for autoencoder
% 
% % Inputs
% D = .01; % Diffusion coefficient
% L = 2*pi; % Length of domain
% n = 128; % Number of grid points
% T = 1;  % End time
% n_time = 101; % Number of time steps
% n_IC = 4; % number of initial conditions
% exp_num = 'exp6f';
% data_set = 'val_x';
% % Specify training data, validation data, or testing data in file name
% 
%     
% % Discretize x
% x = linspace(-L/2,L/2,n+1);
% x = x(1:n);
%     
% % Discretize t
% t = linspace(0,1,n_time);
% 
% 
% % Set Initial Conditions
% 
% u_0 = zeros(n_IC,n);
% 
% for i = 1:n_IC
%     r=rng(n_IC+i);
%     r = rand(1,10);
%     r = r/sum(r);
%     for j = 1:10
%         u_0(i,:) = u_0(i,:) + r(j)*sin(j*x);
%     end
% end
% 
% % Solve Heat Equation
% Data = zeros(n_time*n_IC,n);
% for i = 1:n_IC
%     U = HeatEqn_FT(D,L,x,t,u_0(i,:)); % Periodic BC (FFT)
%     Data(i*n_time-(n_time-1):i*n_time,:) = U;
% end
% 
% filename = strcat('Heat_Eqn_',exp_num,'_',data_set,'.csv');
% dlmwrite(filename, Data, 'precision', '%.14f')








%% Shifted Gaussian
% % Inputs
% D = 1; % Diffusion coefficient
% L = 30; % Length of domain
% n = 128; % Number of grid points
% T = 1;  % End time
% n_time = 101; % Number of time steps
% n_IC = 11; % number of initial conditions
% exp_num = 'exp7b'; % Folder for eventual results
% data_set = 'train1_x';
% % Specify training data, validation data, or testing data in file name
% 
%     
% % Discretize x
% x = linspace(-L/2,L/2,n+1);
% x = x(1:n);
%     
% % Discretize t
% t = linspace(0,1,n_time);
% 
% 
% % Set Initial Conditions
% u_0 = zeros(n_IC,n);
% sigma = 1;
% for i = 1:n_IC
%     x_0 = -6+i;
%     u_0(i,:) = (1/(sigma*sqrt(2*pi)))*exp(-(1/2)*((x-x_0)/sigma).^2);   
% end
% 
% % Solve Heat Equation
% Data = zeros(n_time*n_IC,n);
% 
% for i = 1:n_IC
%     U = HeatEqn_FT(D,L,x,t,u_0(i,:)); 
%     Data(i*n_time-(n_time-1):i*n_time,:) = U;
% end
% 
% filename = strcat('Heat_Eqn_',exp_num,'_',data_set,'.csv');
% dlmwrite(filename, Data, 'precision', '%.14f')
% 
% 
% n_IC2 = 4; % number of initial conditions
% exp_num = 'exp7b'; % Folder for eventual results
% data_set = 'val_x';
% 
% % Set Initial Conditions
% 
% u_02 = zeros(n_IC2,n);
% 
% for i = 1:n_IC2
%     r=rng(i);
%     sigma = rand+1;
%     x_0 = rand*10-5;
%     for j = 1:n_IC
%         u_02(i,:) = (1/(sigma*sqrt(2*pi)))*exp(-(1/2)*((x-x_0)/sigma).^2);
%     end
% end
% 
% 
% % Solve Heat Equation
% Data2 = zeros(n_time*n_IC2,n);
% 
% for i = 1:n_IC2
%     U = HeatEqn_FT(D,L,x,t,u_02(i,:)); 
%     Data2(i*n_time-(n_time-1):i*n_time,:) = U;
% end
% 
% filename = strcat('Heat_Eqn_',exp_num,'_',data_set,'.csv');
% dlmwrite(filename, Data2, 'precision', '%.14f')






%% Shifted and scaled Gaussians
% % Inputs
% D = 1; % Diffusion coefficient
% L = 30; % Length of domain
% n = 128; % Number of grid points
% T = 1;  % End time
% n_time = 101; % Number of time steps
% n_IC = 11; % number of initial conditions
% exp_num = 'exp7c'; % Folder for eventual results
% data_set = 'train1_x';
% % Specify training data, validation data, or testing data in file name
% 
%     
% % Discretize x
% x = linspace(-L/2,L/2,n+1);
% x = x(1:n);
%     
% % Discretize t
% t = linspace(0,1,n_time);
% 
% 
% % Set Initial Conditions
% u_0 = zeros(n_IC,n);
% sigma = .1:.1:1;
% for j = 1:length(sigma)
%     for i = 1:n_IC
%         x_0 = -6+i;
%         u_0(i+(j-1)*n_IC,:) = (1/(sigma(j)*sqrt(2*pi)))*exp(-(1/2)*((x-x_0)/sigma(j)).^2);   
%     end
% end
% 
% 
% 
% % Solve Heat Equation
% Data = zeros(n_time*n_IC*length(sigma),n);
% 
% for i = 1:n_IC*length(sigma)
%     U = HeatEqn_FT(D,L,x,t,u_0(i,:)); 
%     Data(i*n_time-(n_time-1):i*n_time,:) = U;
% end
% 
% 
% filename = strcat('Heat_Eqn_',exp_num,'_',data_set,'.csv');
% dlmwrite(filename, Data, 'precision', '%.14f')
% 
% 
% n_IC2 = 20; % number of initial conditions
% exp_num = 'exp7c'; % Folder for eventual results
% data_set = 'val_x';
% 
% % Set Initial Conditions
% 
% u_02 = zeros(n_IC2,n);
% 
% for i = 1:n_IC2
%     r=rng(i);
%     sigma = rand*2;
%     x_0 = rand*10-5;
%     for j = 1:n_IC
%         u_02(i,:) = (1/(sigma*sqrt(2*pi)))*exp(-(1/2)*((x-x_0)/sigma).^2);
%     end
% end
% 
% 
% % Solve Heat Equation
% Data2 = zeros(n_time*n_IC2,n);
% 
% for i = 1:n_IC2
%     U = HeatEqn_FT(D,L,x,t,u_02(i,:)); 
%     Data2(i*n_time-(n_time-1):i*n_time,:) = U;
% end
% 
% filename = strcat('Heat_Eqn_',exp_num,'_',data_set,'.csv');
% dlmwrite(filename, Data2, 'precision', '%.14f')



%% Sines AND Cosines

clear all; close all; clc
% Create data sets for autoencoder

% Inputs
D = 1; % Diffusion coefficient
L = 2*pi; % Length of domain
n = 128; % Number of grid points
T = 1;  % End time
n_time = 101; % Number of time steps
n_IC = 20; % number of initial conditions
exp_num = 'exp9';
data_set = 'train1_x';
% Specify training data, validation data, or testing data in file name

    
% Discretize x
x = linspace(-L/2,L/2,n+1);
x = x(1:n);
    
% Discretize t
t = linspace(0,1,n_time);


% Set Initial Conditions

u_0 = zeros(n_IC,n);
for k = 1:10
    u_0(k,:) = sin(k*x);
end

for k = 1:10
    u_0(10+k,:) = cos(k*x);
end

% for i = n_IC*2/3+1:n_IC
%     rng(i);
%     r = rand(1,n_IC*2/3);
%     r = r/sum(r);
%     for j = 1:n_IC*1/3
%         u_0(i,:) = u_0(i,:) + r(j)*sin(j*x) + r(j+n_IC/3)*cos(j*x);
%     end
% end

% Solve Heat Equation
Data = zeros(n_time*n_IC,n);
for i = 1:n_IC
    U = HeatEqn_FT(D,L,x,t,u_0(i,:)); % Periodic BC (FFT)
    Data(i*n_time-(n_time-1):i*n_time,:) = U;
end

filename = strcat('Heat_Eqn_',exp_num,'_',data_set,'.csv');
dlmwrite(filename, Data, 'precision', '%.14f')


clear all; close all; clc
% Create data sets for autoencoder

% Inputs
D = 1; % Diffusion coefficient
L = 2*pi; % Length of domain
n = 128; % Number of grid points
T = 1;  % End time
n_time = 101; % Number of time steps
n_IC = 10; % number of initial conditions
exp_num = 'exp9';
data_set = 'val_x';
% Specify training data, validation data, or testing data in file name

    
% Discretize x
x = linspace(-L/2,L/2,n+1);
x = x(1:n);
    
% Discretize t
t = linspace(0,1,n_time);


% Set Initial Conditions

u_0 = zeros(n_IC,n);

for i = 1:n_IC
    rng(i);
    r = rand(1,20);
    r = r/sum(r);
    for j = 1:10
        u_0(i,:) = u_0(i,:) + r(j)*sin(j*x) + r(j+10)*cos(j*x);
    end
end

% Solve Heat Equation
Data = zeros(n_time*n_IC,n);
for i = 1:n_IC
    U = HeatEqn_FT(D,L,x,t,u_0(i,:)); % Periodic BC (FFT)
    Data(i*n_time-(n_time-1):i*n_time,:) = U;
end

filename = strcat('Heat_Eqn_',exp_num,'_',data_set,'.csv');
dlmwrite(filename, Data, 'precision', '%.14f')



