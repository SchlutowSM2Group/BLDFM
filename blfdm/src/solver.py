import numpy as np
import scipy.fft as fft
import time
from .most import vertical_profiles
# from scipy.signal import convolve2d
# import logging


def steady_state_transport_solver(
        surf_flx,
        z, 
        profiles, 
        grid_incr, 
        modes     = (256, 256),
        meas_pt   = (0.0, 0.0),
        surf_bg   = 0.0, 
        footprint = False, 
        analytic  = False,
        fetch     = -1e9 
        ):
    """
    Solves the steady-state advection-diffusion equation for a concentration 
    with flux boundary condition
    given vertical profiles of wind and eddy diffusivity
    using the Fourier, linear shooting and semi-implicit Euler methods

    Parameters:
        surf_flx: array(float)
            2D field of surface kinematic flux at z=z0 [scalar_unit m s-1]
        z: array(float) 
            1D array of vertical grid points from z0 to zm [m]
        profiles: array(float)
            List of 1D arrays of vertical profiles of zonal wind, meridional wind [m s-1]
            and eddy diffusivity [m2 s-1]
        grid_incr: scalar(float) 
            List of grid increments dx and dy [m]
        surf_bg: scalar(float), optional 
            Surface background concentration at z=z0 [scalar_unit]
        modes: scalar(int) 
            List of number of zonal and meridional Fourier modes nlx and nly
        meas_pt: scalar(float), optional
            Coordinates of measurement point xm and ym [m] relative to surf_flx
            where the origin is at surf_flx[0,0]
        footprint: scalar(bool), optional 
            Activates footprints aka Green's function for output
        analytic: scalar(bool), optional
            Analytic solution for constant wind and eddy diffusivity
        fetch: scalar(float), optional 
            Width of zero-flux halo around domain [m].
            Default is min(xmax, ymax).
            fetch = 0.0, same as periodic boundaries.

    Returns:
        p0: array(float)
            2D field of surface concentrations at z=z0 [scalar_unit]
        pm00: scalar(float) 
            Background concentration at z=zm [scalar_unit]
        pm: array(float) 
            2D field of concentration at z=zm or Green's function [scalar_unit]
        qm: array(float) 
            2D field of kinematic flux at z=zm or Footprint [scalar_unit m s-1]
    """
    
    q0       = surf_flx 
    p000     = surf_bg
    u, v, K  = profiles
    dx, dy   = grid_incr
    nlx, nly = modes 
    xm, ym   = meas_pt

    # number of grid cells
    ny, nx = q0.shape

    # domain size
    xmx, ymx = dx*nx, dy*ny

    if fetch < 0.0:
        fetch = min(xmx,ymx)

    # pad width
    px = int(fetch/dx) 
    py = int(fetch/dy)

    # construct zero-flux halo by padding
    q0 = np.pad(q0, ((py,py),(px,px)), mode='constant', constant_values=0.0)

    # extent domain
    nx = nx + 2*px
    ny = ny + 2*py

    if (nlx>nx) or (nly>ny):
        print("Warning: Number of Fourier modes must not exeed number of grid cells.")
        print("Setting both equal.")
        nlx, nly = nx, ny

    # Deltas for truncated Fourier transform
    dlx, dly = (nx-nlx)//2, (ny-nly)//2

    if footprint:
        # Fourier trafo of delta distribution
        tfftq0 = np.ones((nly,nlx),dtype=complex)/nx/ny
    else:
        fftq0 = fft.fft2(q0,norm='forward') # fft of source

        # shift zero wave number to center of array
        fftq0 = fft.fftshift(fftq0)

        # truncate fourier series by removing higher-frequency components
        tfftq0 = fftq0[dly:ny-dly,dlx:nx-dlx]

        # unshift
        tfftq0 = fft.ifftshift(tfftq0)

    # Fourier summation index
    ilx = fft.fftfreq(nlx, d=1.0/nlx) 
    ily = fft.fftfreq(nly, d=1.0/nly) 

    # define truncated zonal and meridional wavenumbers
    lx = 2.0*np.pi/dx/nx * ilx 
    ly = 2.0*np.pi/dy/ny * ily 

    Lx, Ly = np.meshgrid(lx, ly)

    dz = np.diff(z,axis=0)
    nz = len(z)

    # define mask to seperate degenerated and non-degenerated system
    msk      = np.ones((nly,nlx),dtype=bool) # all n and m not equal 0
    msk[0,0] = False

    one  = np.ones( (nly,nlx),dtype=complex)[msk]
    zero = np.zeros((nly,nlx),dtype=complex)[msk]

    Kinv = 1.0 / K[nz-1]

    # Eigenvalue determining solution for z > zm
    eigval = np.sqrt(Lx[msk]**2 + Ly[msk]**2 
                    +1j * u[nz-1] * Kinv * Lx[msk] 
                    +1j * v[nz-1] * Kinv * Ly[msk])

    # initialization
    tfftp0 = np.zeros((nly,nlx),dtype=complex) 
    tfftpm = np.zeros((nly,nlx),dtype=complex) 
    tfftqm = np.zeros((nly,nlx),dtype=complex)

    tfftp0[0,0] = p000
    tfftqm[0,0] = tfftq0[0,0] # conservation by design

    if analytic:

        # constant profiles solution
        # for validation purposes
        h = z[nz-1]-z[0]
        tfftp0[msk] = tfftq0[msk] * Kinv / eigval 
        tfftpm[0,0] = p000 - tfftq0[0,0] * Kinv  * h
        tfftqm[msk] = tfftq0[msk] * np.exp(-eigval * h)
        tfftpm[msk] = tfftqm[msk] * Kinv / eigval 
    
    else:                                        
    
        # solve non-degenerated problem for (n,m) =/= (0,0)
        # by linear shooting method
        # use two auxillary initial value problems  
        tfftp1, tfftq1 = ivp_solver(one, zero,       u,v,K,z,Lx[msk],Ly[msk])
        tfftp2, tfftq2 = ivp_solver(zero,tfftq0[msk],u,v,K,z,Lx[msk],Ly[msk])
                                                 
        alpha = -(tfftq2 - K[nz-1]*eigval*tfftp2) \
               / (tfftq1 - K[nz-1]*eigval*tfftp1)

        # linear combination of the two solution of the IVP 
        tfftp0[msk] = alpha       
        tfftpm[msk] = alpha * tfftp1 + tfftp2        
        tfftqm[msk] = alpha * tfftq1 + tfftq2        
                                                 
        # solve degenerated problem for (n,m) =  (0,0)
        # with Euler forward method
        tfftpm[0,0] = p000                         
        for i in range(nz-1):                    
            tfftpm[0,0] = tfftpm[0,0] - tfftq0[0,0] / K[i] * dz[i]

    # shift green function in Fourier space to measurement point
    if footprint:
        tfftp0 = tfftp0 * np.exp(1j * (Lx*(xm+fetch) + Ly*(ym+fetch)))
        tfftpm = tfftpm * np.exp(1j * (Lx*(xm+fetch) + Ly*(ym+fetch)))
        tfftqm = tfftqm * np.exp(1j * (Lx*(xm+fetch) + Ly*(ym+fetch)))
    # shift such that xm, ym are in the middle of the domain 
    elif xm**2 + ym**2 > 0.0:
        tfftp0 = tfftp0 * np.exp(1j * (Lx*(xm-xmx/2) + Ly*(ym-ymx/2)))
        tfftpm = tfftpm * np.exp(1j * (Lx*(xm-xmx/2) + Ly*(ym-ymx/2)))
        tfftqm = tfftqm * np.exp(1j * (Lx*(xm-xmx/2) + Ly*(ym-ymx/2)))

    # shift zero to center
    tfftp0  = fft.fftshift(tfftp0)
    tfftpm  = fft.fftshift(tfftpm)
    tfftqm  = fft.fftshift(tfftqm)

    # untruncate
    fftp0 = np.pad(tfftp0, ((dly,dly),(dlx,dlx)), mode='constant', constant_values=0.0)
    fftpm = np.pad(tfftpm, ((dly,dly),(dlx,dlx)), mode='constant', constant_values=0.0)
    fftqm = np.pad(tfftqm, ((dly,dly),(dlx,dlx)), mode='constant', constant_values=0.0)

    # unshift
    fftp0 = fft.ifftshift(fftp0)
    fftpm = fft.ifftshift(fftpm)
    fftqm = fft.ifftshift(fftqm)

    if footprint:
        # use fft to reverse sign, make green's function to footprint
        p0 = fft.fft2(fftp0,norm='backward').real # concentration  
        pm = fft.fft2(fftpm,norm='backward').real # concentration  
        qm = fft.fft2(fftqm,norm='backward').real # kinematic flux 
    else: 
        # use ifft as usual
        p0 = fft.ifft2(fftp0,norm='forward').real # concentration  
        pm = fft.ifft2(fftpm,norm='forward').real # concentration  
        qm = fft.ifft2(fftqm,norm='forward').real # kinematic flux 

    return p0[py:ny-py,px:nx-px], fftpm[0,0].real, \
           pm[py:ny-py,px:nx-px], qm[py:ny-py,px:nx-px]


def ivp_solver( fftp0, fftq0, u, v, K, z, Lx, Ly, method='SIE' ):
    '''
    Solves the initial value problem resulting from 
    the discretization of the steady-state advection-diffusion equation
    with the Fast Fourier Transform
    '''

    fftpm, fftqm = np.copy(fftp0), np.copy(fftq0)

    nz = len(z)
    dz = np.diff(z,axis=0)

    for i in range(nz-1):

        Ti =  -K[i]*(Lx**2 + Ly**2) - 1j*u[i]*Lx - 1j*v[i]*Ly
        Kinv = 1.0 / K[i]
        dzi = dz[i]

        # exponential integrator (exact) method
        if method == 'EI':
            eig = np.sqrt( Ti*Kinv )
            dum = np.cos(eig*dzi) * fftpm - Kinv/eig*np.sin(eig*dzi) * fftqm
            fftqm = Ti/eig*np.sin(eig*dzi) * fftpm + np.cos(eig*dzi) * fftqm
            fftpm = dum

        # Taylor series for exponential integrator method up to 3rd order
        if method == 'TSEI3':
            a = 1.0 - 0.5 * Kinv * Ti * dzi**2 
            b = -Kinv * dzi - 1.0/6.0 * Kinv**2 * Ti *dzi**3
            c = Ti * dzi - 1.0/6.0 * Kinv * Ti**2 *dzi**3
            d = 1.0 - 0.5 * Kinv * Ti * dzi**2

            dum   = a * fftpm + b * fftqm
            fftqm = c * fftpm + d * fftqm
            fftpm = dum

        # Semi-implicit Euler method
        if method == 'SIE':
            fftpm = fftpm - dzi * Kinv * fftqm
            fftqm = fftqm + dzi * Ti * fftpm

        # Explicit Euler method
        if method == 'EE':
            dum = fftpm - dz[i]/K[i]*fftqm
            fftqm = fftqm + dz[i]*Ti*fftpm
            fftpm = dum

    return fftpm, fftqm


def point_measurement(f, g):
    """
    Computes the convolution of two 2D arrays evaluated at xm, ym.
    scipy.convolve2d(...,mode='valid') is slighly faster,
    but gives less precise results.
    """
    return np.sum(f*g)


if __name__=='__main__':

    import matplotlib.pyplot as plt

    nx, ny, nz = 512, 256, 10
    nlx, nly   = 512, 512 
    xmx, ymx   = 2000.0, 1000.0
    fetch      = 2000.0
    xm, ym, zm = 1501.0, 700.5, 6.0
    um, vm     = 2.0, 0.5
    ustar, mol = 0.2, 100.0

    R0  = xmx/12

    dx = xmx/nx
    dy = ymx/ny
    dz = zm/nz

    x = np.arange(0.0, xmx, dx)
    y = np.arange(0.0, ymx, dy)

    X, Y = np.meshgrid(x,y)

    p000 = 1.0
    q0 = np.zeros([ny,nx])

    # R = np.sqrt((X-xmx/2)**2 + (Y-ymx/2)**2)
    R = np.abs(X-xmx/2) + np.abs(Y-ymx/2)
    q0 = np.where(R<R0,1.0,0.0)

    # direct computation minimal example with varying profiles
    tic = time.time()
    z, u, v, K = vertical_profiles(nz, zm, um, vm, ustar, mol, constant=False)
    p0, pm00, pm, qm = steady_state_transport_solver(q0,z,(u,v,K),(dx,dy))
    toc = time.time()
    print('Minimal example for stratified BL and default settings')
    print('time ',toc-tic,'s')
    plt.imshow(p0,origin="lower",extent=[0,xmx,0,ymx])
    plt.title("Concentration at z0")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.show()
    plt.imshow(pm,origin="lower",extent=[0,xmx,0,ymx])
    plt.title("Concentration at zm")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.show()
    plt.imshow(qm,origin="lower",extent=[0,xmx,0,ymx])
    plt.title("Vertical kinematic flux at zm")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.show()
    
    # Exact solution with constant profiles
    z, u, v, K = vertical_profiles(nz, zm, um, vm, ustar, constant=True)
    tic = time.time()
    p0,pm00,pm, qm = steady_state_transport_solver(q0,
                                                   z,
                                                   (u,v,K),
                                                   (dx,dy),
                                                   (nlx,nly),
                                                   (xm,ym),
                                                   p000,
                                                   analytic=True,
                                                   fetch=fetch)
    toc = time.time()
    plt.imshow(p0,origin="lower",extent=[0,xmx,0,ymx])
    plt.title("Concentration at z0 for constant profile")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(xmx/2,ymx/2,'ro') 
    plt.colorbar()
    plt.show()
    plt.imshow(pm,origin="lower",extent=[0,xmx,0,ymx])
    # plt.contour(X,Y,p)
    plt.title("Concentration at zm for constant profile")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(xmx/2,ymx/2,'ro') 
    plt.colorbar()
    plt.show()
    print('Constant profile')
    print('time ',toc-tic,'s')
    print('pm00          = ',pm00)
    print('pm at xm,ym   = ',pm[ny//2,nx//2])
    print('qm at xm,ym   = ',qm[ny//2,nx//2])

    # direct computation 
    tic = time.time()
    z, u, v, K = vertical_profiles(nz, zm, um, vm, ustar, mol, constant=True)
    p0, pm00, pm, qm = steady_state_transport_solver(q0,
                                                     z,
                                                     (u,v,K),
                                                     (dx,dy),
                                                     (nlx,nly),
                                                     (xm,ym),
                                                     p000,
                                                     fetch=fetch)
    toc = time.time()
    plt.imshow(p0,origin="lower",extent=[0,xmx,0,ymx])
    plt.title("Concentration at z0")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(xmx/2,ymx/2,'ro') 
    plt.colorbar()
    plt.show()
    plt.imshow(pm,origin="lower",extent=[0,xmx,0,ymx])
    plt.title("Concentration at zm")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(xmx/2,ymx/2,'ro') 
    plt.colorbar()
    plt.show()
    plt.imshow(qm,origin="lower",extent=[0,xmx,0,ymx])
    plt.title("Vertical kinematic flux at zm")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(xmx/2,ymx/2,'ro') 
    plt.colorbar()
    plt.show()
    print('Direct method')
    print('time ',toc-tic,'s')
    print('pm00         = ',pm00)
    print('pm at xm,ym  = ',pm[ny//2,nx//2])
    print('qm at xm,ym  = ',qm[ny//2,nx//2])

    # compute Green function by upgraded solver
    _,_,pg, qg = steady_state_transport_solver(q0,
                                               z,
                                               (u,v,K),
                                               (dx,dy),
                                               (nlx,nly),
                                               (xm,ym),
                                               footprint=True,
                                               fetch=fetch)

    # compute solution by convolution with Green function
    tic = time.time()
    pm = p000 + point_measurement(q0,pg)
    qm = point_measurement(q0,qg)
    toc = time.time()
    print('Convolution with Green function')
    print('time ',toc-tic,'s')
    print('pm at xm,ym   = ',pm)
    print('qm at xm,ym   = ',qm)

    # plt.imshow(qg,origin="lower")
    # plt.imshow(np.roll(pg,(ny//2,nx//2),axis=(0,1)),origin='lower',extent=[0,xmx,0,ymx])
    plt.imshow(pg,origin='lower',extent=[0,xmx,0,ymx])
    plt.title("Flipped Green's function for concentration at zm")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.show()

    plt.imshow(qg,origin='lower',extent=[0,xmx,0,ymx])
    # qg = np.where(qg<0.002,0.0,qg)
    # plt.contour(X,Y,qg)
    plt.title("Footprint")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.show()

