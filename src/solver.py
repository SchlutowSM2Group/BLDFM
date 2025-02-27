import numpy as np
import scipy.fft as fft
import time
from .most import vertical_profiles
from scipy.signal import convolve2d
#import logging

def ivp_solver( fftp0, fftq0, u, v, K, z, Lx, Ly, method='SIE' ):
    '''
    Solves the initial value problem resulting from 
    the discretization of the steady-state advection-diffusion equation
    with the Fast Fourier Transform
    '''

    fftp, fftq = np.copy(fftp0), np.copy(fftq0)

    nz = len(z)
    dz = np.diff(z,axis=0)

    for i in range(nz-1):

        Ti =  -K[i]*(Lx**2 + Ly**2) + 1j*u[i]*Lx + 1j*v[i]*Ly
        Kinv = 1.0 / K[i]
        dzi = dz[i]

        # exponential integrator (exact) method
        if method == 'EI':
            eig = np.sqrt( Ti*Kinv )
            dum = np.cos(eig*dzi) * fftp - Kinv/eig*np.sin(eig*dzi) * fftq
            fftq = Ti/eig*np.sin(eig*dzi) * fftp + np.cos(eig*dzi) * fftq
            fftp = dum

        # Taylor series for exponential integrator method up to 3rd order
        if method == 'TSEI3':
            a = 1.0 - 0.5 * Kinv * Ti * dzi**2 
            b = -Kinv * dzi - 1.0/6.0 * Kinv**2 * Ti *dzi**3
            c = Ti * dzi - 1.0/6.0 * Kinv * Ti**2 *dzi**3
            d = 1.0 - 0.5 * Kinv * Ti * dzi**2

            dum  = a * fftp + b * fftq
            fftq = c * fftp + d * fftq
            fftp = dum

        # Semi-implicit Euler method
        if method == 'SIE':
            fftp = fftp - dzi * Kinv * fftq
            fftq = fftq + dzi * Ti * fftp

        # Explicit Euler method
        if method == 'EE':
            dum = fftp - dz[i]/K[i]*fftq
            fftq = fftq + dz[i]*Ti*fftp
            fftp = dum

    return fftp, fftq


def steady_state_transport_solver(u, v, K, z, 
                                  nx, ny, dx, dy, 
                                  p000      = 0.0, 
                                  q0        = np.array([]), 
                                  green     = False, 
                                  constant  = False,
                                  pxy = 0 ):
    '''
    Solves the steady-state advection diffusion equation for a concentration 
    with flux boundary condition
    given vertical profiles of wind and eddy diffusivity
    using the Fourier and multilayer method

    p000  -  laterally averaged concentration at z=z0
    q0    -  flux boundary condition at z=z0 
    u,v   -  zonal and meridional wind
    K     -  eddy diffusivity
    z     -  grid points
    nx,ny -  # of gridpoints in lateral direction
    dx,dy -  grid increments
    pxy   -  number of ghost cells around domain for zero padding

    Returns concentration p and kinematic flux q at zm.
    If green=True, it returns the respective 
    Green's functions instead
    '''

    q0 = np.pad( q0, pxy, mode='constant', constant_values=0.0 )

    nx = nx + 2*pxy
    ny = ny + 2*pxy

    if green:
        fftq0 = np.ones((ny,nx),dtype=complex)/nx/ny
    else:
        fftq0 = fft.ifft2(q0) # fft of source
    
    fftp = np.zeros((ny,nx),dtype=complex) # initialization
    fftq = np.zeros((ny,nx),dtype=complex) # initialization

    lx = 2.0*np.pi*fft.fftfreq(nx,dx) # Definition of Fourier space 
    ly = 2.0*np.pi*fft.fftfreq(ny,dy)
    
    Lx, Ly = np.meshgrid(lx, ly)
    
    dz = np.diff(z,axis=0)
    nz = len(z)

    # solve boundary value problem for (n,m) =/= (0,0) 
    # use linear shooting method

    # define mask to seperate degenerated and non-degenerated system
    msk      = np.ones((ny,nx),dtype=bool) # all n and m not equal 0
    msk[0,0] = False

    one  = np.ones( (ny,nx),dtype=complex)[msk]
    zero = np.zeros((ny,nx),dtype=complex)[msk]

    Kinv = 1.0 / K[nz-1]

    eigval = np.sqrt( 
        Lx[msk]**2 + Ly[msk]**2 
       -1j * u[nz-1] * Kinv * Lx[msk] 
       -1j * v[nz-1] * Kinv * Ly[msk]
                     )
    fftq[0,0] = fftq0[0,0] # conservation by design

    if constant:

        # constant profiles solution
        # for validation purposes
        dz = z[nz-1]-z[0]
        fftp[0,0] = p000 - fftq0[0,0] * Kinv  * dz
        fftq[msk] = fftq0[msk] * np.exp( -eigval * dz )
        fftp[msk] = fftq[msk] * Kinv / eigval 
    
    else:                                        
    
        # solve non-degenerated problem for (n,m) =/= (0,0)
        # by two auxillary initial value problems  
        fftp1, fftq1 = ivp_solver(one, zero,      u,v,K,z,Lx[msk],Ly[msk])
        fftp2, fftq2 = ivp_solver(zero,fftq0[msk],u,v,K,z,Lx[msk],Ly[msk])
                                                 
        # linear combination of the two solution of the IVP 
        alpha = -( fftq2 - K[nz-1]*eigval*fftp2 ) \
               / ( fftq1 - K[nz-1]*eigval*fftp1 )

        fftp[msk] = alpha * fftp1 + fftp2        
        fftq[msk] = alpha * fftq1 + fftq2        
                                                 
        # solve degenerated problem for (n,m) =  (0,0)
        # with Euler forward method
        fftp[0,0] = p000                         
        for i in range(nz-1):                    
            fftp[0,0] = fftp[0,0] - fftq0[0,0] / K[i] * dz[i]

    p = fft.fft2(fftp).real # concentration  
    q = fft.fft2(fftq).real # kinematic flux 

    if green:
        p = np.roll(p,(ny//2,nx//2),axis=(0,1))
        q = np.roll(q,(ny//2,nx//2),axis=(0,1))

    return p[pxy:ny-pxy,pxy:nx-pxy], q[pxy:ny-pxy,pxy:nx-pxy]


def convolve( f, g, iy, ix):

    ny, nx = g.shape
                           
    g = np.roll(g,(ny//2-iy,nx//2-ix),axis=(0,1))

    return np.sum(f*g)


if __name__=='__main__':

    import matplotlib.pyplot as plt

    nx, ny, nz    = 512, 256, 20
    xmx, ymx, zmx = 2000.0, 1000.0, 5.0
    xm, ym        = 1500.0, 700.0
    um, vm        = 1.2, 0.5
    ustar, mol    = 0.25, 10.0

    pad_width = 0

    R0  = xmx/12

    dx = xmx/nx
    dy = ymx/ny
    dz = zmx/nz

    ix, iy = int(xm/xmx*nx), int(ym/ymx*ny)

    x = np.arange(0.0, xmx, dx)
    y = np.arange(0.0, ymx, dy)

    X, Y = np.meshgrid(x,y)

    p000 = 1.0
    q0 = np.zeros([ny,nx])

    R = np.sqrt((X-xmx/2)**2+(Y-ymx/2)**2)
    q0 = np.where(R<R0,1.0,0.0)

    
    # direct computation with constant profile
    z, u, v, K = vertical_profiles(nz, zmx, um, vm, ustar, constant=True)
    tic = time.time()
    p, q = steady_state_transport_solver(u,v,K,z,nx,ny,dx,dy,p000,q0,constant=True,pxy=pad_width)
    toc = time.time()
    plt.imshow(p,origin="lower",extent=[0,xmx,0,ymx])
    # plt.contour(X,Y,p)
    plt.title("Concentration at zm for constant profile")
    plt.xlabel("x")
    plt.ylabel("y")
    #plt.plot(xm,ym,'ro') 
    plt.colorbar()
    plt.show()
    print('Constant profile')
    print('time ',toc-tic,'s')
    print('p    = ',p[iy,ix])
    print('q    = ',q[iy,ix])
 
    # direct computation by upgraded solver
    tic = time.time()
    z, u, v, K = vertical_profiles(nz, zmx, um, vm, ustar, mol, constant=True)
    p, q = steady_state_transport_solver(u,v,K,z,nx,ny,dx,dy,p000,q0,pxy=pad_width)
    toc = time.time()
    plt.imshow(p,origin="lower",extent=[0,xmx,0,ymx])
    plt.title("Concentration at zm")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(xm,ym,'ro') 
    plt.colorbar()
    plt.show()
    plt.imshow(q,origin="lower",extent=[0,xmx,0,ymx])
    plt.title("Vertical kinematic flux at zm")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(xm,ym,'ro') 
    plt.colorbar()
    plt.show()
    print('Direct method')
    print('time ',toc-tic,'s')
    print('p    = ',p[iy,ix])
    print('q    = ',q[iy,ix])

    # compute Green function by upgraded solver
    pg, qg = steady_state_transport_solver(u,v,K,z,nx,ny,dx,dy,green=True,pxy=pad_width)

    # compute solution by convolution with Green function
    tic = time.time()
    p = p000 + convolve(q0,pg,iy,ix)
    q = convolve(q0,qg,iy,ix)
    toc = time.time()
    print('Convolution with Green function')
    print('time ',toc-tic,'s')
    print('p    = ',p)
    print('q    = ',q)

    # Scipy convolution with Green function
    # tic = time.time()
    # p = p000 + convolve2d(q0,np.roll(pg,(-iy-1,-ix-1),axis=(0,1)),mode='valid')
    # q = convolve2d(q0,np.roll(qg,(-iy-1,-ix-1),axis=(0,1)),mode='valid')
    # toc = time.time()
    # print('Convolution from scipy')
    # print('time ',toc-tic,'s')
    # print('p    = ',p)
    # print('q    = ',q)


    # plt.imshow(qg,origin="lower")
    # plt.imshow(np.roll(pg,(ny//2,nx//2),axis=(0,1)),origin='lower',extent=[0,xmx,0,ymx])
    plt.imshow(pg,origin='lower',extent=[0,xmx,0,ymx])
    plt.title("Green's function for concentration at zm")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.show()

    # plt.imshow(np.roll(qg,(ny//2,nx//2),axis=(0,1)),origin='lower',extent=[0,xmx,0,ymx])
    plt.imshow(qg,origin='lower',extent=[0,xmx,0,ymx])
    plt.title("Green's function for flux aka footprint at zm")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.show()




