clear all; close all; clc
% Read and process data from Deep Learning Algorithm for Koopman

% Inputs and load .mat file
n_IC = 15; % Number of initial conditions
folder_name = 'exp4';  % Experiment folder
file_prefix = strcat('./',folder_name,'/Heat_Eqn_',folder_name,'_2018_03_08_16_52_15_863237_'); % Prefix for file names
%file_prefix = './exp2a/Heat_Eqn_IC_10_exp2a_2018_03_08_08_55_10_735488_';
load(strcat(file_prefix,'model.mat')) % Load the model parameters
n_x = widths(1); % Number of spatial grid points
data_file = strcat('./data/',data_name,'_train1_x.csv'); % Path of data file

plot_type = 4;
% Plot Types:
% 1 - Color plot of the matrix corresponding to initial conditions at choke
%     down point
% 2 - Error of initial conditions
% 3 - Comparison of a given function and the Fourier transform
% 4 - Histogram
% 5 - Confusion Matrix
% 6 - Histogram with cosines
% 7 - Confusion Matrix with cosines




% Read in parameters (matrices E and vectors b)
[WE,bE] = read_params(file_prefix,'E',num_encoder_weights);
%[WO,bO] = read_params(file_prefix,'O',num_omega_weights);
[WD,bD] = read_params(file_prefix,'D',num_decoder_weights);

% Read in data
data = read_data(data_file,len_time,n_IC,n_x);


% Comparison plot
x = linspace(-pi,pi,n_x+1);
x = x(1:n_x);










%% Check the data

% Choose input of network
% f = reshape(data(1,3,:),[n_x,1]);
% 
% y = network(f,WE,bE,act_type); % Encoder
% f_hat = network(y,WD,bD,act_type); % Decoder
% 
% 
% 
% figure(1)
% plot(x,f)
% hold on
% plot(x,f_hat,'--r')
% hold off




if plot_type == 1
    %% Graph matrix of choke down values
    y_mat = [];
    for j = 1:n_IC
        f = reshape(data(1,j,:),[n_x,1]);
        y_mat(:,j) = network(f,WE,bE,act_type); % Encoder
    end

    imagesc(y_mat')
    colorbar
    ylabel('Initial Condition','Fontsize',16)

    % [V,D] = eig(y_mat);
elseif plot_type == 2
    %% Errors on initial conditions
    error = [];
    for i = 1:n_IC
        f = reshape(data(1,i,:),[n_x,1]);

        y = network(f,WE,bE,act_type); % Encoder
        f_hat = network(y,WD,bD,act_type); % Decoder
        error(i) = norm(f-f_hat)/norm(f);
    end
    
%     for i = 1:len_time
%         f = reshape(data(i,5,:),[n_x,1]);
% 
%         y = network(f,WE,bE,act_type); % Encoder
%         f_hat = network(y,WD,bD,act_type); % Decoder
%         error(i) = norm(f-f_hat)/norm(f);
%     end
    

    plot(error,'-o')
    xlabel('Initial Condition','Fontsize',16)
%     xlabel('Time step','Fontsize',16)
%     title('\mu = -5','Fontsize',16)
    ylabel('Error','Fontsize',16)
elseif plot_type == 3
    %% Compare with Fourier Transform

%     f = 0*x;
%     w = [.2 0 .2 .2 .3 0 .1 0 0 0];
%     for j = 1:10
%         f = f + w(j)*sin(j*x);
%     end
% 
% 
      f = [zeros(1,50) ones(1,28) zeros(1,50)];
%      sigma = .5;
%      mu = 0;
%      f = (1/(sigma*sqrt(2*pi)))*exp(-(1/2)*((x-mu)/sigma).^2);  
%        f = x.^2;
 
    y = network(f',WE,bE,act_type); % Encoder
    f_hat = network(y,WD,bD,act_type); % Decoder
    f10 = SineFourier10(f);
    %f10 = Fourier10(f);
    f10_hat = SineFourier10(f_hat);
    %f10_hat = Fourier10(f_hat);

    figure(1)
    plot(x,f,'-b','Linewidth',2)
    hold on
    plot(x,f_hat,'--r','Linewidth',2)
    plot(x,f10,':k','Linewidth',2)
    xlabel('x','Fontsize',20)
    ylabel('f','Fontsize',20)
    h = legend('input','output','FT','Location','Northwest');
    set(h,'Fontsize',20)
%     h = legend('input','output','Location','Northwest');
%     set(h,'Fontsize',20)
    hold off

%     figure(2)
%     plot(x,f_hat,'--r','Linewidth',2)
%     hold on
%     plot(x,f10_hat,':k','Linewidth',2)
%     xlabel('x','Fontsize',20)
%     ylabel('y','Fontsize',20)
%     h = legend('output','output FT','Location','Northwest');
%     set(h,'Fontsize',20)
%     hold off
elseif plot_type == 4
    classify = zeros(1,1000);
    other_count = 0;
    modes = zeros(n_x,10);
    for i = 1:1000
    
        f = 2*(rand(1,n_x)-1);
        %f = rand(1,n_x);
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
        eigenweights2 = zeros(1,10);
        for k = 1:10
            eigenweights(k) = abs(sum(f_hat.*modes(:,k)));
            eigenweights2(k) = norm(f_hat.*modes(:,k));   
        end
    
        [proj classify(i)] = max(eigenweights);
        [dummy classify2] = max(eigenweights2);
    
%       if classify(i) ~= classify2
%           other_count = other_count+1
%         
%           plot(x,f_hat)     
%         
%           rng(15);
%           r = rand(1,10);
%           r = r/sum(r);
%         
%           u_0 = zeros(1,n_x);
%           for j = 1:10
%               u_0 = u_0 + r(j)*sin(j*x);
%           end
%           u_0 = u_0/norm(u_0);
%           hold on
%           plot(x,u_0,'--r')
%           h = legend('other','training')
%           set(h,'Fontsize',20)
%           hold off
%       end
    
        if abs(1-proj) > .1
            classify(i) = 11; 
        end

        %plot(x,f_hat)
    
    end  

    
    edges = .5:1:11.5;
    [N, edges] = histcounts(classify,edges);
    figure(1)
    histogram(classify,edges)
    axis([.5 11.5 0 max(N)*1.1])
    xticklabels({'1','2','3','4','5','6','7','8','9','10','other'})
    xlabel('Wavenumber','Fontsize',16)
    ylabel('Count','Fontsize',16)

    %% Check the training and validation data
%     r_train = zeros(5,10);
%     for i = 11:15
%         rng(i);
%         r = rand(1,10);
%         r = r/sum(r);
%         r_train(i-10,:) = r;
%     end
% 
%     r_train_histo = zeros(1,10);
% 
%     for i = 1:10
%         r_train_histo(i) = 1000*mean(r_train(:,i));
%     end
% 
%     r_val = zeros(4,10);
% 
%     for i = 5:8
%         rng(i);
%         r = rand(1,10);
%         r = r/sum(r);
%         r_val(i-4,:) = r;
%     end
% 
%     r_val_histo = zeros(1,10);
% 
%     for i = 1:10
%         r_val_histo(i) = 1000*mean(r_val(:,i));
%     end
% 
% 
% 
%     hold on
%     plot(1:10,r_train_histo,'or')
%     plot(1:10,r_val_histo,'dk')
%     hold off
%     h = legend('','train','val','Location','Northwest');
%     set(h,'Fontsize',16)
elseif plot_type == 5
    noise_ampl = 10000;
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
    
    %disp(Conf_Mat)
    imagesc(Conf_Mat)
    xlabel('Input','Fontsize',16)
    ylabel('Output','Fontsize',16)
    yticklabels({'1','2','3','4','5','6','7','8','9','10','other'})
    title(['Noise = ',num2str(noise_ampl)],'Fontsize',16)
    h = colorbar
    
    %cd ../../Koopman_Updates/April4_2018/Figures/
    %print(gcf, '-dpdf', 'Exp4_Confusion10000.pdf');
    
    
elseif plot_type == 6
    num_modes = 20;
    classify = zeros(1,1000);
    other_count = 0;
    modes = zeros(n_x,num_modes);
    for i = 1:5000
    
        f = 2*(rand(1,n_x)-1);
        %f = rand(1,n_x);
        f = f/norm(f);
        f_hat = f';
        for j = 1:100
            y = network(f_hat,WE,bE,act_type); % Encoder
            f_hat = network(y,WD,bD,act_type); % Decoder
            f_hat = f_hat/norm(f_hat);
        end

        for j = 1:num_modes 
            modes(:,j) = reshape(data(1,j,:),[n_x,1]);
            modes(:,j) = modes(:,j)/norm(modes(:,j));
        end
    
        eigenweights = zeros(1,num_modes);
        eigenweights2 = zeros(1,num_modes);
        for k = 1:num_modes
            eigenweights(k) = abs(sum(f_hat.*modes(:,k)));
            eigenweights2(k) = norm(f_hat.*modes(:,k));   
        end
    
        [proj classify(i)] = max(eigenweights);
        [dummy classify2] = max(eigenweights2);
    
%       if classify(i) ~= classify2
%           other_count = other_count+1
%         
%           plot(x,f_hat)     
%         
%           rng(15);
%           r = rand(1,10);
%           r = r/sum(r);
%         
%           u_0 = zeros(1,n_x);
%           for j = 1:10
%               u_0 = u_0 + r(j)*sin(j*x);
%           end
%           u_0 = u_0/norm(u_0);
%           hold on
%           plot(x,u_0,'--r')
%           h = legend('other','training')
%           set(h,'Fontsize',20)
%           hold off
%       end
    
        if abs(1-proj) > .1
            classify(i) = num_modes+1; 
        end

        plot(x,f_hat)
    
    end  

    
    edges = .5:1:num_modes+1.5;
    [N, edges] = histcounts(classify,edges);
    figure(1)
    histogram(classify,edges)
    axis([.5 num_modes+1.5 0 max(N)*1.1])
    %xticklabels({'1','2','3','4','5','6','7','8','9','10','other'})
    xticks([1:21])
    xticklabels({'1','2','3','4','5','6','7','8','9','10','1','2','3','4','5','6','7','8','9','10','other'})
    xlabel('Frequency (sin then cos)','Fontsize',16)
    ylabel('Count','Fontsize',16)

    %% Check the training and validation data
%     r_train = zeros(5,10);
%     for i = 11:15
%         rng(i);
%         r = rand(1,10);
%         r = r/sum(r);
%         r_train(i-10,:) = r;
%     end
% 
%     r_train_histo = zeros(1,10);
% 
%     for i = 1:10
%         r_train_histo(i) = 1000*mean(r_train(:,i));
%     end
% 
%     r_val = zeros(4,10);
% 
%     for i = 5:8
%         rng(i);
%         r = rand(1,10);
%         r = r/sum(r);
%         r_val(i-4,:) = r;
%     end
% 
%     r_val_histo = zeros(1,10);
% 
%     for i = 1:10
%         r_val_histo(i) = 1000*mean(r_val(:,i));
%     end
% 
% 
% 
%     hold on
%     plot(1:10,r_train_histo,'or')
%     plot(1:10,r_val_histo,'dk')
%     hold off
%     h = legend('','train','val','Location','Northwest');
%     set(h,'Fontsize',16)
elseif plot_type == 7
    noise_ampl = 10000;
    Conf_Mat = zeros(21,20);

    for l = 1:10
        classify = zeros(1,1000);
        modes = zeros(n_x,20);
        for i = 1:1000
    
            f = sin(l*x)+noise_ampl*(2*rand(1,n_x)-1);
            f = f/norm(f);
            f_hat = f';
            for j = 1:100
                y = network(f_hat,WE,bE,act_type); % Encoder
                f_hat = network(y,WD,bD,act_type); % Decoder
                f_hat = f_hat/norm(f_hat);
            end

            for j = 1:20 
                modes(:,j) = reshape(data(1,j,:),[n_x,1]);
                modes(:,j) = modes(:,j)/norm(modes(:,j));
            end
    
            eigenweights = zeros(1,20);
            for k = 1:20
                eigenweights(k) = abs(sum(f_hat.*modes(:,k)));  
            end
    
            [proj classify(i)] = max(eigenweights);
    
            if abs(1-proj) > .1
                classify(i) = 21; 
            end
    
        end 
    
        for k = 1:21
            Conf_Mat(k,l) = sum(classify==k)/1000; 
        end
    
    end
    
    for l = 1:10
        classify = zeros(1,1000);
        modes = zeros(n_x,20);
        for i = 1:1000
    
            f = cos(l*x)+noise_ampl*(2*rand(1,n_x)-1);
            f = f/norm(f);
            f_hat = f';
            for j = 1:100
                y = network(f_hat,WE,bE,act_type); % Encoder
                f_hat = network(y,WD,bD,act_type); % Decoder
                f_hat = f_hat/norm(f_hat);
            end

            for j = 1:20 
                modes(:,j) = reshape(data(1,j,:),[n_x,1]);
                modes(:,j) = modes(:,j)/norm(modes(:,j));
            end
    
            eigenweights = zeros(1,20);
            for k = 1:20
                eigenweights(k) = abs(sum(f_hat.*modes(:,k)));  
            end
    
            [proj classify(i)] = max(eigenweights);
    
            if abs(1-proj) > .1
                classify(i) = 21; 
            end
    
        end 
    
        for k = 1:21
            Conf_Mat(k,l+10) = sum(classify==k)/1000; 
        end
    
    end
    
    disp(Conf_Mat)
    imagesc(Conf_Mat)
    xlabel('Input','Fontsize',16)
    ylabel('Output','Fontsize',16)
    yticks([1:21])
    yticklabels({'1','2','3','4','5','6','7','8','9','10','1','2','3','4','5','6','7','8','9','10','other'})
    xticks([1:20])
    xticklabels({'1','2','3','4','5','6','7','8','9','10','1','2','3','4','5','6','7','8','9','10'})
    title(['Noise = ',num2str(noise_ampl)],'Fontsize',16)
    h = colorbar;
    
    cd ../../Koopman_Updates/April4_2018/Figures/
    print(gcf, '-dpdf', 'Exp8_Confusion10000.pdf');
    
else
    disp('error in plot_type')
end






%% Check linear combinations

% f = 0*x;
% w = [0 1 0 0 0 0 0 0 0 0];
% for j = 1:10
%     f = f + w(j)*sin(j*x);
% end
% f = f';
% y = network(f,WE,bE,act_type);
% f_hat = network(y,WD,bD,act_type); % Decoder
% 
% % norm(f-f_hat)/norm(f)
% 
% % y2 = zeros(10,1);
% % for j = 1:10
% %     y2 = y2 + w(j)*y_mat(:,j);
% % end
% close all;
% 
% figure(3)
% plot(x,f)
% hold on
% plot(x,f_hat,'--r')
% hold off
% % 
% % figure(4)
% % plot(y)
% % hold on
% % plot(y2,'--r')
% % hold off



%% Iterate the map
%rng(61)
%f = rand(1,128);
% f = sin(2*x);
% f = f/norm(f);
% f_hat = f';
% for j = 1:100
%     y = network(f_hat,WE,bE,act_type); % Encoder
%     f_hat = network(y,WD,bD,act_type); % Decoder
%     f_hat = f_hat/norm(f_hat);
% end
% 
%     
% plot(x,f,'-b','Linewidth',2)
% hold on
% plot(x,f_hat,'--r','Linewidth',2)
% xlabel('x','Fontsize',20)
% ylabel('f','Fontsize',20)
% h = legend('input','100 iterations','Location','Northeast');
% set(h,'Fontsize',20)
% hold off













%% Iterate map and see evolution
% rng(61)
% f = zeros(n_x,101);
% f(:,1) = rand(n_x,1);
% f(:,1) = f(:,1)/norm(f(:,1));
% f(:,1) = cos(4*x);
% for j = 1:100
%     y = network(f(:,j),WE,bE,act_type); % Encoder
%     f(:,j+1) = network(y,WD,bD,act_type); % Decoder
%     f(:,j+1) = f(:,j+1)/norm(f(:,j+1));
% end
% figure(1)
% surfl(1:101,x,f)    
% 
% 
% figure(2)
% surfl(1:10,x,f(:,1:10)) 

% figure(3)
% plot(x,f(:,1))
% hold on
% for j = 2:4
%     plot(x,f(:,j))
% end
% hold off

% figure(4)
% shift = ones(length(x));
% plot3(x,0*shift,f(:,1),'-b')
% hold on
% for j = 1:12
%     plot3(x,j*shift,f(:,2*j+1),'-b')
% end
% hold off
% xlabel('x')
% ylabel('t')
% zlabel('f')
% view(30,32)


