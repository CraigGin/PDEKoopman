from __future__ import division
import numpy as np
from pyDOE import lhs # Must install this package, see https://pythonhosted.org/pyDOE/index.html
from scipy.stats import geom
from PDEsolvers import KS_Periodic

# Create data sets for Experiment 5
# All data comes from solutions to Kuramoto-Sivashinsky equation. 
# Training data:
#   Initial conditions:
#       White noise, Sines, Square waves 
#   Solve from t = 0 to 6.25 in steps of 0.125
#   128 spatial points in [-pi,pi)
# Validation data:
#    same structure as training data but with 20000 

np.random.seed(0)  

# Inputs (data)
exp_num = 'exp7'
n = 128 # Number of grid points
n_IC = 6000 # Number of initial conditions in each file
n_train = 20 # Number of training files
M = n_IC*n_train//3 # Samples from latin hypercube

# Inputs (KS)
L = 8*np.pi # Length of domain
dt = 0.5 # Size of time step for data
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
c_vect = L*X[:,1] - L/2
w_vect = (L-4*(x[1]-x[0]))*X[:,2]+2*(x[1]-x[0]) 

# Loop over files
sine_ind = 0
square_ind = 0
for train_num in range(n_train):

    data_set = 'train%d_x' % (train_num+1)
    
    # Set Initial Conditions
    u_0 = np.zeros((n_IC,n))
    
    # White noise
    for k in range(0,n_IC-2,3):
        ut = np.zeros(n, dtype=np.complex128)
        ut[0] = np.random.normal()
        ut[1:n//2] = np.random.normal(size=(n//2-1))+1j*np.random.normal(size=(n//2-1))
        ut[n//2] = np.random.normal()
        ut[n//2+1:] = np.flipud(np.conj(ut[1:n//2]))
        u = np.real(np.fft.ifft(ut))
        u_0[k,:] = u-np.mean(u)
    
    # Sines
    for k in range(1,n_IC-1,3):
        u_0[k,:] = A_vect[sine_ind]*np.sin(2*np.pi*omega_vect[sine_ind]/L*x+phi_vect[sine_ind])
        sine_ind += 1

    # Square waves
    for k in range(2,n_IC,3):
        u = A2_vect[square_ind]*np.logical_or(np.logical_or(np.abs(x-c_vect[square_ind])<w_vect[square_ind]/2,np.abs(x+L-c_vect[square_ind]) < w_vect[square_ind]/2),abs(x-L-c_vect[square_ind]) < w_vect[square_ind]/2)
        u_0[k,:] = u - np.mean(u)
        square_ind += 1

    # Solve KS Equation
    Data = np.zeros((n_time*n_IC,n), dtype=np.float32)
    for k in range(n_IC):
        Data[k*n_time:(k+1)*n_time,:] = KS_Periodic(x,T,n_time,u_0[k,:])     
    
    np.save(('KS_Eqn_%s_%s' % (exp_num, data_set)), Data, allow_pickle=False)


# Validation Data
n_IC = 30000 # Number of initial conditions
data_set = 'val_x'
M = n_IC//3 # Samples from latin hypercube

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
c_vect = L*X[:,1] - L/2
w_vect = (L-4*(x[1]-x[0]))*X[:,2]+2*(x[1]-x[0]) 

# Set Initial Conditions
u_0 = np.zeros((n_IC,n))
    
# White noise
for k in range(0,n_IC-2,3):
    ut = np.zeros(n, dtype=np.complex128)
    ut[0] = np.random.normal()
    ut[1:n//2] = np.random.normal(size=(n//2-1))+1j*np.random.normal(size=(n//2-1))
    ut[n//2] = np.random.normal()
    ut[n//2+1:] = np.flipud(np.conj(ut[1:n//2]))
    u = np.real(np.fft.ifft(ut))
    u_0[k,:] = u-np.mean(u)
    
# Sines
sine_ind = 0
for k in range(1,n_IC-1,3):
    u_0[k,:] = A_vect[sine_ind]*np.sin(2*np.pi*omega_vect[sine_ind]/L*x+phi_vect[sine_ind])
    sine_ind += 1

# Square waves
square_ind = 0
for k in range(2,n_IC,3):
    u = A2_vect[square_ind]*np.logical_or(np.logical_or(np.abs(x-c_vect[square_ind])<w_vect[square_ind]/2,np.abs(x+L-c_vect[square_ind]) < w_vect[square_ind]/2),abs(x-L-c_vect[square_ind]) < w_vect[square_ind]/2)
    u_0[k,:] = u - np.mean(u)
    square_ind += 1

# Solve Burgers' Equation
Data = np.zeros((n_time*n_IC,n), dtype=np.float32)
for k in range(n_IC):
    Data[k*n_time:(k+1)*n_time,:] = KS_Periodic(x,T,n_time,u_0[k,:])     
    
np.save(('KS_Eqn_%s_%s' % (exp_num, data_set)), Data, allow_pickle=False)
