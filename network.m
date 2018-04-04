function [x] = network(x,W,b)
% Run data through encoder or decoder

[dummy num_weights] = size(W);

for i = 1:num_weights-1
   x = W{i}*x+b{i};
   x = max(x,0);
end

x = W{num_weights}*x+b{num_weights};

end

