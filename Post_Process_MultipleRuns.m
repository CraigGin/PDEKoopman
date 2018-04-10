clear all; close all; clc
% Read and process data from Deep Learning Algorithm for Koopman

runs = 356;
not_found = zeros(1,11);
almost_not_found = zeros(1,11);
num_not_found = zeros(1,runs);
which_not_found = zeros(runs,10);
cum_Conf_mat = zeros(11,10);


for file_num = 0:runs-1

    % Inputs and load .mat file
    n_IC = 10; % Number of initial conditions
    folder_name = 'exp10';  % Experiment folder
    file_prefix = strcat('./',folder_name,'/Heat_Eqn_',folder_name,'_',num2str(file_num),'_'); % Prefix for file names
    load(strcat(file_prefix,'model.mat')) % Load the model parameters
    n_x = widths(1); % Number of spatial grid points
    data_file = strcat('./data/',data_name,'_train1_x.csv'); % Path of data file
    
    x = linspace(-pi,pi,n_x+1);
    x = x(1:n_x);

    % Read in parameters (matrices E and vectors b)
    [WE,bE] = read_params(file_prefix,'E',num_encoder_weights);
    %[WO,bO] = read_params(file_prefix,'O',num_omega_weights);
    [WD,bD] = read_params(file_prefix,'D',num_decoder_weights);

    % Read in data
    data = read_data(data_file,len_time,n_IC,n_x);


    classify = zeros(1,1000);
    other_count = 0;
    modes = zeros(n_x,10);
    for i = 1:1000
    
        f = 2*(rand(1,n_x)-1);
        f = f/norm(f);
        f_hat = f';
        for j = 1:100
            y = network(f_hat,WE,bE,act_type); % Encoder
            f_hat = network(y,WD,bD,act_type); % Decoder
            f_hat = f_hat/norm(f_hat);
        end

        for j = 1:10 
            modes(:,j) = reshape(data(1,j,:),[n_x,1]);
            modes(:,j) = modes(:,j)/norm(modes(:,j));
        end
    
        eigenweights = zeros(1,10);
        for k = 1:10
            eigenweights(k) = abs(sum(f_hat.*modes(:,k)));
        end
    
        [proj,classify(i)] = max(eigenweights);
    
        if abs(1-proj) > .1
            classify(i) = 11; 
        end

    
    end  

    
    edges = .5:1:11.5;
    [N, edges] = histcounts(classify,edges);

    
    not_found = not_found+(N==0);
    almost_not_found = almost_not_found+ (N<3);
    num_not_found(file_num+1) = sum(N(1:10)==0);
    which_not_found(file_num+1,:) = N(1:10)==0;

    
    
    

    noise_ampl = 0;
    Conf_Mat = zeros(11,10);

    for l = 1:10
        classify = zeros(1,1000);
        modes = zeros(n_x,10);
        for i = 1:1000
    
            f = sin(l*x)+noise_ampl*(2*rand(1,n_x)-1);
            f = f/norm(f);
            f_hat = f';
            for j = 1:100
                y = network(f_hat,WE,bE,act_type); % Encoder
                f_hat = network(y,WD,bD,act_type); % Decoder
                f_hat = f_hat/norm(f_hat);
            end

            for j = 1:10 
                modes(:,j) = reshape(data(1,j,:),[n_x,1]);
                modes(:,j) = modes(:,j)/norm(modes(:,j));
            end
    
            eigenweights = zeros(1,10);
            for k = 1:10
                eigenweights(k) = abs(sum(f_hat.*modes(:,k)));  
            end
    
            [proj classify(i)] = max(eigenweights);
    
            if abs(1-proj) > .1
                classify(i) = 11; 
            end
    
        end 
    
        for k = 1:11
            Conf_Mat(k,l) = sum(classify==k)/1000; 
        end
    
    end
    
    cum_Conf_mat = cum_Conf_mat + Conf_Mat;
    

end
%%

cd ../../Koopman_Updates/April11_2018/Figures/


figure(1)
bar(1:11,not_found/runs)
title('Not found','Fontsize',16)
xticklabels({'1','2','3','4','5','6','7','8','9','10','other'})
xlabel('Wavenumber','Fontsize',16)
ylabel('Probability','Fontsize',16)
print(gcf, '-dpdf', 'Exp10_Histo_NotFound.pdf');

figure(2)
bar(1:11,almost_not_found/runs)
title('Almost not found (<3)','Fontsize',16)
xticklabels({'1','2','3','4','5','6','7','8','9','10','other'})
xlabel('Wavenumber','Fontsize',16)
ylabel('Probability','Fontsize',16)
print(gcf, '-dpdf', 'Exp10_Histo_AlmNotFound.pdf');

M = max(num_not_found);
edges2 = -0.5:1:M+.5;
[N, edges2] = histcounts(num_not_found,edges2);
figure(3)
bar(0:M,N/runs)
xlabel('Number of modes not found','Fontsize',16)
ylabel('Probability','Fontsize',16)
print(gcf, '-dpdf', 'Exp10_Histo_NumNotFound.pdf');

figure(4)
imagesc(which_not_found)
xlabel('Wavenumber','Fontsize',16)
ylabel('Run','Fontsize',16)
print(gcf, '-dpdf', 'Exp10_WhichNotFound.pdf');

cum_Conf_mat = cum_Conf_mat/runs;
figure(5)
imagesc(cum_Conf_Mat)
xlabel('Input','Fontsize',16)
ylabel('Output','Fontsize',16)
yticklabels({'1','2','3','4','5','6','7','8','9','10','other'})
h = colorbar
print(gcf, '-dpdf', 'Exp10_CumConfusion.pdf');
