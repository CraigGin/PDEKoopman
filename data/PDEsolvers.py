from __future__ import division
import numpy as np

def HeatEqn_FT(D,L,x,t,u_0):
    # Solves the 1-D Heat equation using the Fourier Transform
    # Periodic BC
    # Inputs:
    # D = diffusion coeff
    # L = length of spatial domain
    # x = spatial discretization
    # t = time discretization
    # u_0 = initial conditions
    
    n = x.size
    k = (2*np.pi/L)*np.fft.fftfreq(n,d=1/n)
    u_0t = np.fft.fft(u_0)
    
    U = np.zeros((t.size,x.size))
    for ti in range(t.size):
        U[ti,:] = np.fft.ifft(np.exp(-D*k**2*t[ti])*u_0t)
        
    return U

def Burgers_Periodic(mu,eps,x,t_output,dt_factor,u_k):
    # Solves the Burgers' equation:
    # du/dt + eps*u*(du/dx) = mu*(d^2u/dx^2)
    # Uses a second order FD method
    # Periodic BC
    # Inputs:
    # mu = diffusion coeff/ viscosity
    # eps = strength of advection
    # x = spatial discretization
    # t_output = time discretization for output
    # dt_factor = divide the dt from the above time discretization when solving
    # (i.e. refine the time step for numerical stability, but only output every 
    #  dt_factor steps)
    # u_k = initial conditions
    
    n = x.size
    dx = x[1]-x[0]
    dt = (t_output[1]-t_output[0])/dt_factor
    nt = (t_output.size-1)*dt_factor+1
    
    # Solve with FD scheme
    U = np.zeros((t_output.size,n))
    U[0,:] = u_k
    plus_ind = np.arange(1,n+1) % n
    minus_ind = np.arange(-1,n-1) % n
    for ti in range(1,nt):
        F_plus = (u_k**2 + u_k[plus_ind]**2)/4
        F_minus = (u_k**2 + u_k[minus_ind]**2)/4
        u_k = u_k + dt*(mu*(u_k[plus_ind]-2*u_k+u_k[minus_ind])/dx**2-eps*(F_plus-F_minus)/dx)

        if ti % dt_factor == 0:
            U[ti//dt_factor,:] = u_k
        
    return U

def KS_Periodic(x,tmax,ntime,u):
    # Solves the Kuramoto-Sivashinsky equation:
    # u_t = -u*u_x - u_xx - u_xxxx
    # Periodic BC
    # Inputs:
    # x = spatial discretization
    # tmax = end time for simulation
    # ntime = number of time steps
    # u = initial condition
    
    N = x.size
    #x = np.reshape(x,(N,1))
    #u = np.reshape(u,(N,1))
    v = np.fft.fft(u)
    
    # Precompute various ETDRK4 scalar quantities:
    h = 0.025
    k = (2*np.pi/(x[-1]-2*x[0]+x[1]))*np.fft.fftfreq(N,d=1/N)
    L = k**2 - k**4
    E = np.exp(h*L)
    E2 = np.exp(h*L/2)
    M = 64
    r = np.exp(1j*np.pi*(np.arange(1,M+1)-.5)/M)
    LR = h*np.tile(np.reshape(L,(N,1)),(1,M)) + np.tile(r,(N,1))
    Q = h*np.real(np.mean((np.exp(LR/2)-1)/LR ,axis=1))
    f1 = h*np.real(np.mean((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3 , axis=1))
    f2 = h*np.real(np.mean((2+LR+np.exp(LR)*(-2+LR))/LR**3 ,axis=1))
    f3 = h*np.real(np.mean((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3 ,axis=1))
    
    # Main time-stepping loop:
    uu = u
    nmax = int(np.round(tmax/h)) 
    nplt = int(np.floor((tmax/(ntime-1))/h))
    g = -0.5j*k
    for n in range(nmax):
        t = (n+1)*h
        Nv = g*np.fft.fft(np.real(np.fft.ifft(v))**2)
        a = E2*v + Q*Nv
        Na = g*np.fft.fft(np.real(np.fft.ifft(a))**2)
        b = E2*v + Q*Na
        Nb = g*np.fft.fft(np.real(np.fft.ifft(b))**2)
        c = E2*a + Q*(2*Nb-Nv)
        Nc = g*np.fft.fft(np.real(np.fft.ifft(c))**2)
        v = E*v + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3
        if np.mod(n+1,nplt)==0:
            u = np.real(np.fft.ifft(v))
            uu = np.vstack((uu,u)) 
    
    return uu