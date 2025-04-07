import numpy as np
import scipy.fft as fft
import time
# from scipy.signal import convolve2d
# import logging


def steady_state_transport_solver(
        srf_flx,
        z, 
        profiles, 
        domain, 
        modes       = (512, 512),
        meas_pt     = (0.0, 0.0),
        srf_bg_conc = 0.0, 
        footprint   = False, 
        analytic    = False,
        fetch       = -1e9 
        ):
    """
    Solves the steady-state advection-diffusion equation for a concentration 
    with flux boundary condition
    given vertical profiles of wind and eddy diffusivity
    using the Fourier, linear shooting and semi-implicit Euler methods

    Parameters:
        srf_flx: array(float)
            2D field of surface kinematic flux at z=z0 [scalar_unit m s-1]
        z: array(float) 
            1D array of vertical grid points from z0 to zm [m]
        profiles: array(float)
            List of 1D arrays of vertical profiles of zonal wind, meridional wind [m s-1]
            and eddy diffusivity [m2 s-1]
        domain: scalar(float) 
            List of domain sizes xmax and ymax [m]
        srf_bg_conc: scalar(float), optional 
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
        srf_conc: array(float)
            2D field of surface concentrations at z=z0 [scalar_unit]
        bg_conc: scalar(float) 
            Background concentration at z=zm [scalar_unit]
        conc: array(float) 
            2D field of concentration at z=zm or Green's function [scalar_unit]
        flx: array(float) 
            2D field of kinematic flux at z=zm or Footprint [scalar_unit m s-1]
    """
    
    q0       = srf_flx 
    p000     = srf_bg_conc
    u, v, K  = profiles
    xmx, ymx = domain
    nlx, nly = modes 
    xm, ym   = meas_pt

    # number of grid cells
    ny, nx = q0.shape

    # grid increments
    dx, dy = xmx/nx, ymx/ny

    if fetch < 0.0:
        fetch = max(xmx,ymx)

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
    tfftp = np.zeros((nly,nlx),dtype=complex) 
    tfftq = np.zeros((nly,nlx),dtype=complex)

    tfftp0[0,0] = p000
    tfftq[0,0] = tfftq0[0,0] # conservation by design

    if analytic:

        # constant profiles solution
        # for validation purposes
        h = z[nz-1]-z[0]
        tfftp0[msk] = tfftq0[msk] * Kinv / eigval 
        tfftp[0,0] = p000 - tfftq0[0,0] * Kinv  * h
        tfftq[msk] = tfftq0[msk] * np.exp(-eigval * h)
        tfftp[msk] = tfftq[msk] * Kinv / eigval 
    
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
        tfftp[msk] = alpha * tfftp1 + tfftp2        
        tfftq[msk] = alpha * tfftq1 + tfftq2        
                                                 
        # solve degenerated problem for (n,m) =  (0,0)
        # with Euler forward method
        tfftp[0,0] = p000                         
        for i in range(nz-1):                    
            tfftp[0,0] = tfftp[0,0] - tfftq0[0,0] / K[i] * dz[i]

    # shift green function in Fourier space to measurement point
    if footprint:
        tfftp0 = tfftp0 * np.exp(1j * (Lx*(xm+fetch) + Ly*(ym+fetch)))
        tfftp = tfftp * np.exp(1j * (Lx*(xm+fetch) + Ly*(ym+fetch)))
        tfftq = tfftq * np.exp(1j * (Lx*(xm+fetch) + Ly*(ym+fetch)))
    # shift such that xm, ym are in the middle of the domain 
    elif xm**2 + ym**2 > 0.0:
        tfftp0 = tfftp0 * np.exp(1j * (Lx*(xm-xmx/2) + Ly*(ym-ymx/2)))
        tfftp = tfftp * np.exp(1j * (Lx*(xm-xmx/2) + Ly*(ym-ymx/2)))
        tfftq = tfftq * np.exp(1j * (Lx*(xm-xmx/2) + Ly*(ym-ymx/2)))

    # shift zero to center
    tfftp0  = fft.fftshift(tfftp0)
    tfftp  = fft.fftshift(tfftp)
    tfftq  = fft.fftshift(tfftq)

    # untruncate
    fftp0 = np.pad(tfftp0, ((dly,dly),(dlx,dlx)), mode='constant', constant_values=0.0)
    fftp = np.pad(tfftp, ((dly,dly),(dlx,dlx)), mode='constant', constant_values=0.0)
    fftq = np.pad(tfftq, ((dly,dly),(dlx,dlx)), mode='constant', constant_values=0.0)

    # unshift
    fftp0 = fft.ifftshift(fftp0)
    fftp = fft.ifftshift(fftp)
    fftq = fft.ifftshift(fftq)

    if footprint:
        # use fft to reverse sign, make green's function to footprint
        p0 = fft.fft2(fftp0,norm='backward').real # concentration  
        p = fft.fft2(fftp,norm='backward').real # concentration  
        q = fft.fft2(fftq,norm='backward').real # kinematic flux 
    else: 
        # use ifft as usual
        p0 = fft.ifft2(fftp0,norm='forward').real # concentration  
        p = fft.ifft2(fftp,norm='forward').real # concentration  
        q = fft.ifft2(fftq,norm='forward').real # kinematic flux 

    srf_conc = p0[py:ny-py,px:nx-px]
    bg_conc  = fftp[0,0].real
    conc     = p[py:ny-py,px:nx-px] 
    flx      = q[py:ny-py,px:nx-px]

    return srf_conc, bg_conc, conc, flx


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

            dum   = a * fftp + b * fftq
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

