function [data] = read_data(data_file,len_time,n_IC,n_x)
% Read in data matrix

temp = csvread(data_file);
data = reshape(temp,[len_time,n_IC,n_x]);

end


