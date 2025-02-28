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

        Ti =  -K[i]*(Lx**2 + Ly**2) - 1j*u[i]*Lx - 1j*v[i]*Ly
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
                                  dx, dy, 
                                  nlx, nly,
                                  q0, 
                                  p000      = 0.0, 
                                  green     = False, 
                                  analytic  = False,
                                  fetch     = 0.0 ):
    """
    Solves the steady-state advection-diffusion equation for a concentration 
    with flux boundary condition
    given vertical profiles of wind and eddy diffusivity
    using the Fourier, linear shooting and semi-implicit Euler methods

    Paramters:
        p000 (float): laterally averaged concentration at z=z0 [1]
        q0 (float): kinematic flux boundary condition at z=z0 [m s-1]
        u,v (float): zonal and meridional wind [m s-1]
        K (float): eddy diffusivity [m2 s-1]
        z (float): grid points [m]
        nlx,nly (int): number of Fourier modes
        dx,dy (float): grid increments [m]
        fetch (float): zero-flux halo around domain [m]
        green (bool): returns Green's function if True

    Returns:
        p (float): concentration at zm
        q (float): kinematic flux at zm.
    """
    
    # number of grid cells
    ny, nx = q0.shape

    # pad width
    px = int(fetch/dx) 
    py = int(fetch/dy)

    # construct zero-flux halo by padding
    q0  = np.pad( q0, ((py,py),(px,px)), mode='constant', constant_values=0.0 )

    # extent domain
    nx = nx + 2*px
    ny = ny + 2*py

    if (nlx>nx) or (nly>ny):
        print("Warning: Number of Fourier modes must not exeed number of grid cells.")
        print("Setting both equal.")
        nlx, nly = nx, ny

    # Deltas for truncated Fourier transform
    dlx, dly = (nx-nlx)//2, (ny-nly)//2

    if green:
        tfftq0 = np.ones((nly,nlx),dtype=complex)/nx/ny
    else:
        fftq0 = fft.fft2(q0,norm='forward') # fft of source

        # truncate fourier series by removing higher-frequency components
        tfftq0 = np.zeros((nly, nlx), dtype=complex)

        # shift zero wave number to center of array
        fftq0 = fft.fftshift(fftq0)

        # truncate
        tfftq0 = fftq0[dly:ny-dly,dlx:nx-dlx]

        # shift back
        tfftq0 = fft.ifftshift(tfftq0)

    # initialization
    tfftp  = np.zeros((nly,nlx),dtype=complex) 
    tfftq  = np.zeros((nly,nlx),dtype=complex) # initialization

    # Fourier summation index
    ilx = fft.fftfreq( nlx, d=1.0/nlx  ) 
    ily = fft.fftfreq( nly, d=1.0/nly  ) 

    # wavenumbers
    lx = 2.0*np.pi/dx/nx * ilx 
    ly = 2.0*np.pi/dy/ny * ily 

    Lx, Ly = np.meshgrid(lx, ly)

    dz = np.diff(z,axis=0)
    nz = len(z)

    # solve boundary value problem for (n,m) =/= (0,0) 
    # use linear shooting method

    # define mask to seperate degenerated and non-degenerated system
    msk      = np.ones((nly,nlx),dtype=bool) # all n and m not equal 0
    msk[0,0] = False

    one  = np.ones( (nly,nlx),dtype=complex)[msk]
    zero = np.zeros((nly,nlx),dtype=complex)[msk]

    Kinv = 1.0 / K[nz-1]

    eigval = np.sqrt( 
        Lx[msk]**2 + Ly[msk]**2 
       +1j * u[nz-1] * Kinv * Lx[msk] 
       +1j * v[nz-1] * Kinv * Ly[msk]
                     )
    tfftq[0,0] = tfftq0[0,0] # conservation by design

    if analytic:

        # constant profiles solution
        # for validation purposes
        dz = z[nz-1]-z[0]
        tfftp[0,0] = p000 - tfftq0[0,0] * Kinv  * dz
        tfftq[msk] = tfftq0[msk] * np.exp( -eigval * dz )
        tfftp[msk] = tfftq[msk] * Kinv / eigval 
    
    else:                                        
    
        # solve non-degenerated problem for (n,m) =/= (0,0)
        # by two auxillary initial value problems  
        tfftp1, tfftq1 = ivp_solver(one, zero,       u,v,K,z,Lx[msk],Ly[msk])
        tfftp2, tfftq2 = ivp_solver(zero,tfftq0[msk],u,v,K,z,Lx[msk],Ly[msk])
                                                 
        # linear combination of the two solution of the IVP 
        alpha = -( tfftq2 - K[nz-1]*eigval*tfftp2 ) \
               / ( tfftq1 - K[nz-1]*eigval*tfftp1 )

        tfftp[msk]  = alpha * tfftp1 + tfftp2        
        tfftq[msk]  = alpha * tfftq1 + tfftq2        
                                                 
        # solve degenerated problem for (n,m) =  (0,0)
        # with Euler forward method
        tfftp[0,0]  = p000                         
        for i in range(nz-1):                    
            tfftp[0,0] = tfftp[0,0] - tfftq0[0,0] / K[i] * dz[i]

    # untruncate
    tfftp  = fft.fftshift(tfftp)
    tfftq  = fft.fftshift(tfftq)
    fftp   = np.pad(tfftp, ((dly,dly),(dlx,dlx)), mode='constant', constant_values=0.0)
    fftq   = np.pad(tfftq, ((dly,dly),(dlx,dlx)), mode='constant', constant_values=0.0)
    fftp   = fft.ifftshift(fftp)
    fftq   = fft.ifftshift(fftq)

    p  = fft.ifft2(fftp,norm='forward').real # concentration  
    q  = fft.ifft2(fftq,norm='forward').real # kinematic flux 

    if green:
        p  = np.roll(p,(ny//2,nx//2),axis=(0,1))
        q  = np.roll(q,(ny//2,nx//2),axis=(0,1))

    return p[py:ny-py,px:nx-px], q[py:ny-py,px:nx-px]


def convolve( f, g, iy, ix):

    ny, nx = g.shape
                           
    g = np.roll(g,(ny//2-iy,nx//2-ix),axis=(0,1))

    return np.sum(f*g)


if __name__=='__main__':

    import matplotlib.pyplot as plt

    nx, ny, nz    = 1024, 512, 20
    nlx, nly      = 512, 256  
    xmx, ymx, zmx = 2000.0, 1000.0, 10.0
    fetch         = 2000.0
    xm, ym        = 1500.0, 700.0
    um, vm        = 1.2, 0.5
    ustar, mol    = 0.25, 100.0

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
    p, q = steady_state_transport_solver(u,v,K,z,dx,dy,nlx,nly,q0,p000,analytic=True,fetch=fetch)
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
 
    # direct computation 
    tic = time.time()
    z, u, v, K = vertical_profiles(nz, zmx, um, vm, ustar, mol, constant=True)
    p, q = steady_state_transport_solver(u,v,K,z,dx,dy,nlx,nly,q0,p000,fetch=fetch)
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
    dum = np.zeros([ny,nx]) # dummy for dimensions
    pg, qg = steady_state_transport_solver(u,v,K,z,dx,dy,nlx,nly,q0=dum,green=True,fetch=fetch)

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




