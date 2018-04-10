 clear all; close all; clc
% Solve the 1-D Heat Equation 
% u_t = Du_xx

% Inputs
D = 1; % Diffusion coefficient
L = 2*pi; % Length of domain
n = 128; % Number of grid points
T = 1;  % End time
n_time = 101; % Number of time steps
n_IC = 10; % number of initial conditions
BC = 1; % Type of boundary conditions
% Use BC = 1 for periodic, BC = 2 for Dirichlet, BC = 3 for Neumann
exp_num = 'exp2a';
data_set = 'val_x';
% Specify training data, validation data, or testing data in file name

if BC == 1
    % No inputs needed
    
    BC_name = 'periodic';
    
    % Discretize x
    x = linspace(-L/2,L/2,n+1);
    x = x(1:n);
    
elseif BC == 2
    a = 0; % u(-L/2)
    b = 0; % u(L/2)
    
    BC_name = 'Dirichlet';
    
    % Discretize x
    x = linspace(-L/2,L/2,n);
    
elseif BC == 3
    a = 0; % u'(-L/2)
    b = 0; % u'(L/2)
    
    BC_name = 'Neumann';
    
    % Discretize x
    x = linspace(-L/2,L/2,n);
    
else
    display('Invalid entry for BC')
    return
end

%%%%%%%%%%%%%%%%%%%%

% Discretize t
t = linspace(0,T,n_time);


% Set Initial Conditions

u_0 = zeros(n_IC,n);
for i = 1:n_IC

    if BC == 1
        u_0(i,:) = IC_Periodic(x,i);
    elseif BC == 2
        u_0(i,:) = IC_Dirichlet_rand(n,a,b);
    elseif BC == 3
        u_0(i,:) = IC_Neumann_rand(n,a,b);
    end
end

% Solve Heat Equation
Data = zeros(n_time*n_IC,n);
for i = 1:n_IC

    % Call solver
    if BC == 1
        U = HeatEqn_FT(D,L,x,t,u_0(i,:)); % Periodic BC (FFT)
        %U = HeatEqn_FDP(D,L,x,t,u_0(i,:)); % Periodic BC (Finite Diff)
    elseif BC == 2
        U = HeatEqn_FDD(D,L,x,t,u_0(i,:));  % Dirichlet
    elseif BC == 3
        U = HeatEqn_FDN(D,L,x,t,u_0(i,:));  % Neumann
    end
   
    Data(i*n_time-(n_time-1):i*n_time,:) = U;
end

filename = strcat('Heat_Eqn_',exp_num,'_',data_set,'.csv');
dlmwrite(filename, Data, 'precision', '%.14f')

%surfl(x,t,real(U)); 
%shading interp; colormap(gray); 


