function [U] = HeatEqn_FDN(D,L,x,t,u_0)
% Solves the 1-D Heat equation using Crank-Nicolson FD method
% Neumann BC
% Inputs:
% D = diffusion coeff
% L = length of spatial domain
% x = spatial discretization
% t = time discretization
% u_0 = initial conditions

h = x(2)-x(1);
k = t(2) - t(1);
n = length(x);
c1 = (-u_0(3)+4*u_0(2)-3*u_0(1))/(2*h); % left bdry derivative
c2 = (u_0(n-2)-4*u_0(n-1)+3*u_0(n))/(2*h);; % right bdry derivative

% Solve with Crank-Nicolson FD
U(1,:) = u_0;

 % Build the matrix
 A = zeros(n);
 
 % Neumann Boundary Conditions
 A(1,1) = (1/k)+ D/h^2; A(1,2) = -D/h^2;
 A(n,n-1) = -D/h^2; A(n,n) = (1/k)+ D/h^2;
 
 for j = 2:n-1
  A(j,j-1) = -D/(2*h^2);
  A(j,j) = (1/k)+ D/h^2;
  A(j,j+1) = -D/(2*h^2);
 end

 b = zeros(n,1);
for i = 1:length(t)-1
   
    b(1) = D/(h^2)*U(i,2)+(1/k - D/h^2)*U(i,1) -2*D*c1/h;
    b(n) = D/(h^2)*U(i,n-1)+(1/k - D/h^2)*U(i,n) +2*D*c2/h;
    
    for j = 2:n-1
        b(j) = D/(2*h^2)*U(i,j-1)+(1/k - D/h^2)*U(i,j)+D/(2*h^2)*U(i,j+1);
    end
    
    temp = A\b;
    U(i+1,:) = temp.';  
end


end

