function [W,b] = read_params(file_prefix,EOD,num_weights)
% Read in W matrices and b vectors from Deep Learning algortihm

for i = 1:num_weights
    W{i} = csvread(strcat(file_prefix,'W',EOD,num2str(i),'.csv'));
    W{i} = W{i}';
    b{i} = csvread(strcat(file_prefix,'b',EOD,num2str(i),'.csv'));
end

end

