import numpy as np
import scipy.fft as fft
import time
from .most import vertical_profiles
from scipy.linalg import expm
#import logging

def ivp_solver( fftp0, fftq0, u, v, K, z, Lx, Ly ):

    fftp, fftq = np.copy(fftp0), np.copy(fftq0)

    nz = len(z)
    dz = np.diff(z,axis=0)

    for i in range(nz-1):

        Ti =  -K[i]*(Lx**2 + Ly**2) - 1j*u[i]*Lx - 1j*v[i]*Ly
        Kinv = 1.0/K[i]
        dzi = dz[i]
        # exponential integrator method
        # eig = np.sqrt(Ti/K[i])
        # dum = np.cos(eig*dz[i])*fftp-1.0/K[i]/eig*np.sin(eig*dz[i])*fftq
        # fftq = Ti/eig*np.sin(eig*dz[i])*fftp+np.cos(eig*dz[i])*fftq
        # fftp = dum

        # Taylor series for exponential method up to 3rd order
        # a = 1.0 - 0.5 * Kinv * Ti * dzi**2 
        # b = -Kinv * dzi - 1.0/6.0 * Kinv**2 * Ti *dzi**3
        # c = Ti * dzi - 1.0/6.0 * Kinv * Ti**2 *dzi**3
        # d = 1.0 - 0.5 * Kinv * Ti * dzi**2

        # dum  = a * fftp + b * fftq
        # fftq = c * fftp + d * fftq
        # fftp = dum

        # Semi-implicit Euler method
        fftp = fftp - dzi*Kinv*fftq
        fftq = fftq + dzi*Ti*fftp

        # Explicit Euler method
        # dum = fftp - dz[i]/K[i]*fftq
        # fftq = fftq + dz[i]*Ti*fftp
        # fftp = dum



    return fftp, fftq

def steady_state_transport_solver( u, v, K, z, nx, ny, dx, dy, p000=0.0, q0=np.array([]), green=False, constant=False ):
    '''
    Solves the steady-state advection diffusion equation for a concentration 
    with flux boundary condition
    given vertical profiles of wind and eddy diffusivity
    using the Fourier and multilayer method

    p000   -  laterally averaged concentration at z=z0
    q0     -  flux boundary condition at z=z0 
    u,v    -  zonal and meridional wind
    K      -  eddy diffusivity
    z      -  grid points
    nx,ny  -  # of gridpoints in lateral direction
    dx,dy  -  grid increments

    Returns concentration p and kinematic flux q at zm.
    If green=True, it returns Green's functions
    '''

    if green:
        fftq0 = np.ones((ny,nx),dtype=complex)
    else:
        fftq0 = fft.fft2(q0) # fft of source
    
    fftp = np.zeros((ny,nx),dtype=complex) # initialization
    fftq = np.zeros((ny,nx),dtype=complex) # initialization

    lx = 2.0*np.pi*fft.fftfreq(nx,dx) # Definition of Fourier space 
    ly = 2.0*np.pi*fft.fftfreq(ny,dy)
    
    Lx, Ly = np.meshgrid(lx, ly)
    # Lx, Ly = lx.reshape((1,-1)), ly.reshape((-1,1))
    
    dz = np.diff(z,axis=0)
    nz = len(z)

    # solve boundary value problem for (n,m) =/= (0,0) 
    # use linear shooting method
    msk = np.ones((ny,nx),dtype=bool) # all n and m not equal 0
    msk[0,0] = False

    one  = np.ones( (ny,nx),dtype=complex)[msk]
    zero = np.zeros((ny,nx),dtype=complex)[msk]

    eigval = np.sqrt( Lx[msk]**2 + Ly[msk]**2 + 1j*u[nz-1]/K[nz-1]*Lx[msk] + 1j*v[nz-1]/K[nz-1]*Ly[msk] ) 

    if constant:

        # constant profiles solution
        dz = z[nz-1]-z[0]
        fftp[0,0] = p000 - fftq0[0,0] / K[nz-1] * dz
        fftq[msk] = fftq0[msk] * np.exp( -eigval * dz )
        fftp[msk] = fftq0[msk] / K[nz-1] / eigval 
    
    else:                                        
    
        # solve auxillary initial value problem  
        fftp1, fftq1 = ivp_solver(one, zero,      u,v,K,z,Lx[msk],Ly[msk])
        fftp2, fftq2 = ivp_solver(zero,fftq0[msk],u,v,K,z,Lx[msk],Ly[msk])
                                                 
        alpha = -( fftq2 - K[nz-1]*eigval*fftp2  ) / ( fftq1 - K[nz-1]*eigval*fftp1 )
        fftp[msk] = alpha * fftp1 + fftp2        
        fftq[msk] = alpha * fftq1 + fftq2        
                                                 
        # solve degenerated problem for (n,m) =  (0,0)
        fftp[0,0] = p000                         
        for i in range(nz-1):                    
            fftp[0,0] = fftp[0,0] - fftq[0,0] / K[i] * dz[i]

    p = fft.ifft2(fftp).real # concentration  
    q = fft.ifft2(fftq).real # kinematic flux 

    return p, q


def convolve(f,g,iy,ix):

    g = np.roll(g,(-iy,-ix),axis=(0,1))

    return np.sum(f*g)


if __name__=='__main__':

    import matplotlib.pyplot as plt

    nx, ny, nz    = 1024, 512, 2
    xmx, ymx, zmx = 2000.0, 1000.0, 20.0
    um, vm, Km    = 1.0, -3.0, 2.0
    ix, iy        = 300, 128 
    ustar, mol    = 0.25, 1e9

    constant = False

    R0  = xmx/12

    dx = xmx/nx
    dy = ymx/ny
    dz = zmx/nz

    x = np.arange(0.0, xmx, dx)
    y = np.arange(0.0, ymx, dy)

    X, Y = np.meshgrid(x,y)

    # z, u, v, K = vertical_profiles(nz, zmx, um, vm, ustar, mol)
    z = np.linspace( 0, zmx, nz )
    u = um * np.ones(nz)
    v = vm * np.ones(nz)
    K = Km * np.ones(nz)

    p000 = 1.0
    q0 = np.zeros([ny,nx])

    R = np.sqrt((X-xmx/2)**2+(Y-ymx/2)**2)
    q0 = np.where(R<R0,1.0,0.0)

    tic = time.time()
    # direct computation by upgraded solver
    p, q = steady_state_transport_solver(u,v,K,z,nx,ny,dx,dy,p000,q0,constant=constant)
    toc = time.time()
    plt.imshow(p,origin="lower")
    plt.title("Concentration at zm")
    plt.xlabel("ix")
    plt.ylabel("iy")
    plt.plot(ix,iy,'ro') 
    plt.colorbar()
    plt.show()
    plt.imshow(q,origin="lower")
    plt.title("Vertical kinematic flux at zm")
    plt.xlabel("ix")
    plt.ylabel("iy")
    plt.plot(ix,iy,'ro') 
    plt.colorbar()
    plt.show()
    print('Direct method')
    print('time ',toc-tic,'s')
    print('p    = ',p[iy,ix])
    print('q    = ',q[iy,ix])

    # compute Green function by upgraded solver
    pg, qg = steady_state_transport_solver(u,v,K,z,nx,ny,dx,dy,green=True,constant=constant)

    # compute solution by convolution with Green function
    tic = time.time()
    p = p000/nx/ny + convolve(q0,pg,iy,ix)
    q = convolve(q0,qg,iy,ix)
    toc = time.time()
    print('Convolution with Green function')
    print('time ',toc-tic,'s')
    print('p    = ',p)
    print('q    = ',q)

    plt.imshow(qg,origin="lower")
    plt.colorbar()
    plt.show()



