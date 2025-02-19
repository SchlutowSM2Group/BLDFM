import numpy as np
import scipy.fft as fft
import time
from .most import vertical_profiles
#import logging

def ivp_solver( fftp0, fftq0, u, v, K, z, Lx, Ly ):

    fftp, fftq = np.copy(fftp0), np.copy(fftq0)

    nz = len(z)
    dz = np.diff(z,axis=0)

    # explicit forward Euler method
    for i in range(nz-1):

        Ti = np.sqrt( -K[i]*(Lx**2 + Ly**2) + 1j*u[i]*Lx + 1j*v[i]*Ly ) 

        fftp = fftp - fftq / K[i] * dz[i]
        fftq = fftq + fftp * Ti * dz[i]

    # exponential integrator method
    # to be continued...

    return fftp, fftq

def steady_state_transport_solver( u, v, K, z, nx, ny, dx, dy, p000=0.0, q0=np.array([]), green=False ):
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
        fftq = np.ones((ny,nx),dtype=complex)/nx/ny
    else:
        fftq = fft.ifft2(q0) # fft of source
    
    fftp = np.zeros((ny,nx),dtype=complex) # initialization

    lx = 2.0*np.pi*fft.fftfreq(nx,dx) # Definition of Fourier space 
    ly = 2.0*np.pi*fft.fftfreq(ny,dy)
    
    Lx, Ly = np.meshgrid(lx, ly)
    
    dz = np.diff(z,axis=0)
    nz = len(z)

    # solve boundary value problem for (n,m) =/= (0,0) 
    # use linear shooting method
    msk = np.ones((ny,nx),dtype=bool) # all n and m not equal 0
    msk[0,0] = False

    one  = np.ones( (ny-1,nx-1),dtype=complex)
    zero = np.zeros((ny-1,nx-1),dtype=complex)

    # solve auxillary initial value problem
    fftp1, fftq1 = ivp_solver(one, zero,     u,v,K,z,Lx[msk],Ly[msk])
    fftp2, fftq2 = ivp_solver(zero,fftq[msk],u,v,K,z,Lx[msk],Ly[msk])

    eigval = np.sqrt( Lx[msk]**2 + Ly[msk]**2 - 1j*u[nz]/K[nz]*Lx[msk] - 1j*v[nz]/K[nz]*Ly[msk] ) 

    fftp[msk] = fftp1 * ( K[nz]*eigval*fftp2 - fftq2 ) / ( fftq1 - K[nz]*eigval*fftp1 )
    fftq[msk] = fftq2

    # solve degenarated problem for (n,m) = (0,0)
    fftp[0,0] = p000

    for i in range(nz-1):
        
        fftp[0,0] = fftp[0,0] - fftq[0,0] / K[i] * dz[i]

    p = fft.fft2(fftp).real # concentration  
    q = fft.fft2(fftq).real # kinematic flux 

    return q, p


def convolve(f,g,iy,ix):

    g = np.roll(g,(-iy,-ix),axis=(0,1))

    return np.sum(f*g)


if __name__=='__main__':

    import matplotlib.pyplot as plt

    nx, ny, nz    = 512, 256, 100
    xmx, ymx, zmx = 2000.0, 1000.0, 5.0
    um, vm, Km    = 1.0, -2.0, 2.0
    ix, iy        = 300, 128 
    ustar, mol    = 0.25, 100.0

    R0  = xmx/12

    dx = xmx/nx
    dy = ymx/ny
    dz = zmx/nz

    x = np.arange(0.0, xmx, dx)
    y = np.arange(0.0, ymx, dy)

    X, Y = np.meshgrid(x,y)

    z, u, v, K = vertical_profiles(nz, zmx, um, vm, ustar, mol)

    p0 = 1.0
    q0 = np.zeros([ny,nx])

    R = np.sqrt((X-xmx/2)**2+(Y-ymx/2)**2)
    q0 = np.where(R<R0,1.0,0.0)

    tic = time.time()
    # direct computation by upgraded solver
    q, p = steady_state_transport_solver(u,v,K,z,nx,ny,dx,dy,p0,q0)
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
    qg, pg = steady_state_transport_solver(u,v,K,z,nx,ny,dx,dy,green=True)

    # compute solution by convolution with Green function
    tic = time.time()
    p = p0 + convolve(q0,pg,iy,ix)
    q = convolve(q0,qg,iy,ix)
    toc = time.time()
    print('Convolution with Green function')
    print('time ',toc-tic,'s')
    print('p    = ',p)
    print('q    = ',q)

    plt.imshow(qg,origin="lower")
    plt.colorbar()
    plt.show()



    


