from __future__ import division
import numpy as np
from pyDOE import lhs # Must install this package, see https://pythonhosted.org/pyDOE/index.html
from scipy.stats import geom
from PDEsolvers import KS_Periodic

# Create data sets for Experiment 4
# All data comes from solutions to Kuramoto-Sivashinsky equation. 
# Testing data:
#   Initial conditions:
#       White noise, Sines, Square waves, Guassians, Triangle waves
#   Solve from t = 0 to 50 in steps of 1
#   128 spatial points in [-pi,pi)

np.random.seed(0)  

# Inputs (data)
exp_num = 'exp4'
n = 128 # Number of grid points
n_IC = 1000 # Number of initial conditions in each file
M = n_IC # Samples from latin hypercube

# Inputs (KS)
L = 8*np.pi # Length of domain
dt = 1.0 # Size of time step for data
n_time = 51 # Number of time steps
T = dt*(n_time-1)  # End time

# Discretize x
x = np.linspace(-L/2,L/2,n+1)
x = x[:n]

# Create vectors of random values for sines

# Sampling of A and phi
X = lhs(2, samples=M, criterion='maximin')
A_vect = X[:,0]
phi_vect = 2*np.pi*X[:,1]

# Sampling of omega
max_omega = 10
cum_distrib = geom.cdf(np.arange(1,max_omega+1),0.25)
cum_distrib = cum_distrib/cum_distrib[-1]
numbs = np.random.uniform(size=M)

omega_vect = np.zeros(M)

for k in range(max_omega):
    omega_vect = omega_vect + (numbs < cum_distrib[k])

omega_vect = 11 - omega_vect

# Create vectors of random values for square waves

# Sampling of A, c, and w
X = lhs(3, samples=M, criterion='maximin')
A2_vect = X[:,0]
c2_vect = L*X[:,1] - L/2
w2_vect = (L-4*(x[1]-x[0]))*X[:,2]+2*(x[1]-x[0]) 

# Sampling of mean and sigma for Gaussians
X = lhs(2, samples=M, criterion='maximin')
mean_vect = X[:,0]
sigma_vect = (1-(x[1]-x[0]))*X[:,1]+(x[1]-x[0]) 

# Create vectors of random values for triangle waves

# Sampling of A, c, and w
X = lhs(3, samples=M, criterion='maximin')
A3_vect = X[:,0]
c3_vect = L*X[:,1] - L/2
w3_vect = (L-4*(x[1]-x[0]))*X[:,2]+2*(x[1]-x[0])


# White noise
Data = np.zeros((n_time*n_IC,n), dtype=np.float32)
for k in range(n_IC):
    ut = np.zeros(n, dtype=np.complex128)
    ut[0] = np.random.normal()
    ut[1:n//2] = np.random.normal(size=(n//2-1))+1j*np.random.normal(size=(n//2-1))
    ut[n//2] = np.random.normal()
    ut[n//2+1:] = np.flipud(np.conj(ut[1:n//2]))
    u = np.real(np.fft.ifft(ut))
    u_0 = u-np.mean(u)
    Data[k*n_time:(k+1)*n_time,:] = KS_Periodic(x,T,n_time,u_0)
    
data_set = 'test1_x'
np.save(('KS_Eqn_%s_%s' % (exp_num, data_set)), Data, allow_pickle=False)
    
# Sines
Data = np.zeros((n_time*n_IC,n), dtype=np.float32)
for k in range(n_IC):
    u_0 = A_vect[k]*np.sin(2*np.pi*omega_vect[k]/L*x+phi_vect[k])
    Data[k*n_time:(k+1)*n_time,:] = KS_Periodic(x,T,n_time,u_0)
    
data_set = 'test2_x'
np.save(('KS_Eqn_%s_%s' % (exp_num, data_set)), Data, allow_pickle=False)

# Square waves
Data = np.zeros((n_time*n_IC,n), dtype=np.float32)
for k in range(n_IC):
    u = A2_vect[k]*np.logical_or(np.logical_or(np.abs(x-c2_vect[k])<w2_vect[k]/2,np.abs(x+L-c2_vect[k]) < w2_vect[k]/2),abs(x-L-c2_vect[k]) < w2_vect[k]/2)
    u_0 = u - np.mean(u)
    Data[k*n_time:(k+1)*n_time,:] = KS_Periodic(x,T,n_time,u_0)
    
data_set = 'test3_x'
np.save(('KS_Eqn_%s_%s' % (exp_num, data_set)), Data, allow_pickle=False)

# Gaussians
Data = np.zeros((n_time*n_IC,n), dtype=np.float32)
for k in range(n_IC):
    Gmean = mean_vect[k]
    sigma = sigma_vect[k]
    u = 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-Gmean)**2/(2*sigma**2))
    u_0 = u - np.mean(u)
    Data[k*n_time:(k+1)*n_time,:] = KS_Periodic(x,T,n_time,u_0)
    
data_set = 'test4_x'
np.save(('KS_Eqn_%s_%s' % (exp_num, data_set)), Data, allow_pickle=False)

# Triangle Waves
Data = np.zeros((n_time*n_IC,n), dtype=np.float32)
for k in range(n_IC):
    u1 = 2*A3_vect[k]/w3_vect[k]*(x-c3_vect[k]+w3_vect[k]/2)*np.logical_and(-w3_vect[k]/2<=x-c3_vect[k],x-c3_vect[k] <=0)\
        +2*A3_vect[k]/w3_vect[k]*(x-L-c3_vect[k]+w3_vect[k]/2)*np.logical_and(-w3_vect[k]/2<=x-L-c3_vect[k],x-L-c3_vect[k]<=0)     
    u2 = -2*A3_vect[k]/w3_vect[k]*(x-c3_vect[k]-w3_vect[k]/2)*np.logical_and(w3_vect[k]/2>x-c3_vect[k],x-c3_vect[k] >0)\
        -2*A3_vect[k]/w3_vect[k]*(x+L-c3_vect[k]-w3_vect[k]/2)*np.logical_and(w3_vect[k]/2>x+L-c3_vect[k],x+L-c3_vect[k]>0)
    u = u1+u2
    u_0 = u - np.mean(u)
    Data[k*n_time:(k+1)*n_time,:] = KS_Periodic(x,T,n_time,u_0)
    
data_set = 'test5_x'
np.save(('KS_Eqn_%s_%s' % (exp_num, data_set)), Data, allow_pickle=False)



