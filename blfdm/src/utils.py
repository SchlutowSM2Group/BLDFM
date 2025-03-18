import numpy as np

def compute_wind_fields(u_rot, wind_dir):

    wind_dir = np.deg2rad(wind_dir)
    u = u_rot * np.sin(wind_dir)
    v = u_rot * np.cos(wind_dir)

    return u, v


def ideal_source(nxy, domain, shape='diamond'):
    """
    Returns a circle or diamond shaped surface flux source field.
    Useful for testing.
    """

    nx, ny = nxy
    xmx, ymx = domain

    x = np.linspace(0.0, xmx, nx)
    y = np.linspace(0.0, ymx, ny)

    X, Y = np.meshgrid(x,y)

    q0 = np.zeros([ny,nx])

    # Circular source
    # R = np.sqrt((X-xmx/2)**2 + (Y-ymx/2)**2)

    # Diamond source  
    R = np.abs(X-xmx/2) + np.abs(Y-ymx/2)
    
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




