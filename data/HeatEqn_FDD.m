function [U] = HeatEqn_FD(D,L,x,t,u_0)
% Solves the 1-D Heat equation using Crank-Nicolson FD method
% Dirichlet BC
% Inputs:
% D = diffusion coeff
% L = length of spatial domain
% x = spatial discretization
% t = time discretization
% u_0 = initial conditions

h = x(2)-x(1);
k = t(2) - t(1);
n = length(x);

% Solve with Crank-Nicolson FD
U(1,:) = u_0;

 % Build the matrix
 A = zeros(n);
 
 % Dirichlet Boundary Conditions
 A(1,1) = 1;
 A(n,n) = 1;
 
 for j = 2:n-1
  A(j,j-1) = -D/(2*h^2);
  A(j,j) = (1/k)+ D/h^2;
  A(j,j+1) = -D/(2*h^2);
 end

 b = zeros(n,1);
 b(1) = u_0(1);
 b(n) = u_0(n);
for i = 1:length(t)-1
   
    for j = 2:n-1
        b(j) = D/(2*h^2)*U(i,j-1)+(1/k - D/h^2)*U(i,j)+D/(2*h^2)*U(i,j+1);
    end
    
    temp = A\b;
    U(i+1,:) = temp.';  
end


end

