import numpy as np
import scipy.fft as fft


def compute_wind_fields(u_rot, wind_dir):

    wind_dir = np.deg2rad(wind_dir)
    u = u_rot * np.sin(wind_dir)
    v = u_rot * np.cos(wind_dir)

    return u, v

def point_source(nxy, domain, src_pt):

    xs, ys   = src_pt
    nx, ny   = nxy
    xmx, ymx = domain

    dx, dy = xmx/nx, ymx/ny

    # Fourier summation index
    ilx = fft.fftfreq(nx, d=1.0/nx) 
    ily = fft.fftfreq(ny, d=1.0/ny) 

    # define zonal and meridional wavenumbers
    lx = 2.0*np.pi/dx/nx * ilx 
    ly = 2.0*np.pi/dy/ny * ily 

    Lx, Ly = np.meshgrid(lx, ly)

    fftq0 = np.ones((ny,nx),dtype=complex)
    
    # shift to source point in Fourier space
    fftq0 = fftq0 * np.exp(-1j * (Lx*xs + Ly*ys))/nx/ny

    # normalize
    fftq0 = fftq0 / dx / dy

    return fft.ifft2(fftq0,norm='forward').real


def ideal_source(nxy, domain, shape='diamond'):
    """
    Returns a circle or diamond shaped surface flux source field.
    Useful for testing.
    """

    nx, ny   = nxy
    xmx, ymx = domain

    x = np.linspace(0.0, xmx, nx)
    y = np.linspace(0.0, ymx, ny)

    X, Y = np.meshgrid(x,y)

    q0 = np.zeros([ny,nx])

    # Circular source
    # R = np.sqrt((X-xmx/2)**2 + (Y-ymx/2)**2)

    # Diamond source  
    R = np.abs(X-xmx/4) + np.abs(Y-ymx/4)
    
    R0  = xmx/12

    q0 = np.where(R<R0,1.0,0.0)

    return q0


def point_measurement(f, g):
    """
    Computes the convolution of two 2D arrays evaluated at xm, ym.
    scipy.convolve2d(...,mode='valid') is slighly faster,
    but gives less precise results.
    """
    return np.sum(f*g)




