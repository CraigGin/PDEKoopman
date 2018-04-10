function [x] = network(x,W,b,act_type)
% Run data through encoder or decoder

[dummy num_weights] = size(W);

for i = 1:num_weights-1
   x = W{i}*x+b{i};
   if strcmp(act_type,'relu') == 1
     x = max(x,0);
   end
end

x = W{num_weights}*x+b{num_weights};

end

