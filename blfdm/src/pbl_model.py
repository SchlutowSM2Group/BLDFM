import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import newton

def vertical_profiles(
        n,
        meas_height,
        wind,
        ustar,
        mol = 1e9,
        prsc = 1.0,
        model = "MOST",
        z0 = -1e9,
        z0_min = 0.001,
        z0_max = 2.0
        ):
    """ 
    Computes profiles of horizontal wind and eddy diffusivity 
    depending on the boundary layer's stratifictation
    from point measurements with, e.g., Eddy Covariance systems
    according to Monin-Obukhov Similarity Theory (MOST) 
    (Kormann and Meixner, 2001, https://doi.org/10.1023/A:1018991015119).
    
    Parameters:
        n: scalar(int)
            Number of vertical grid points
        meas_height: scalar(float)
            Measurement height 
        wind: array(float)
            List of zonal and meridional wind components u and v and 
            the measurement height
        ustar: scalar(float)
            Friction velocity
        mol: scalar(float)
            Monin-Obukhov length
        prsc: scalar(float)
            Prandtl to Schmidt number ratio depending on the scalar 
            that is subject to atmospheric dispersion
        constant: scalar(bool)
            Switch for returning constant profiles
            
    Returns:
        z: array(float)
            1D array of vertical grid points
        profiles: array(float)
            List of 1D arrays of horizontal wind components u and v
            as well as the eddy diffusivity K
    """

    zm, (um, vm) = meas_height, wind

    kap = 0.4 # Karman constant

    if model == "CONSTANT":

        Km = kap * ustar * zm / prsc
        z = np.linspace( 0.0, zm, n )
        u = um * np.ones(n)
        v = vm * np.ones(n)
        K = Km * np.ones(n)

    if model == "MOST":

        # absolute wind at zm
        absum = np.sqrt( um**2 + vm**2 ) 

        if z0 < 0.0:

            # roughness length
            z0 = zm * np.exp( -kap * absum / ustar + psi( zm/mol ) )

            # sanity checks
            z0 = min(z0, z0_max)
            z0 = max(z0, z0_min)

        else:

            ustar = absum * kap / (np.log(zm/z0) + psi(zm/mol)) 

        # equidistant vertical grid
        z = np.linspace( z0, zm, n )

        absu = ustar / kap * ( np.log( z/z0 ) + psi( z/mol ) ) 

        u = um / absum * absu
        v = vm / absum * absu

        K = kap * ustar * z / phi( z/mol ) / prsc

    # One-and-a-half order closure
    # if model == "OAAHOC":


    print('Stats from vertical_profiles')
    print('z0    = %.3f m' % z[0])
    print('ustar = %.3f m s-1' % ustar)
    print('umax  = %.3f m s-1, vmax = %.3f m s-1, Kmax = %.3f m2 s-1' 
          % (max(u), max(v), max(K)))
    print('umin  = %.3f m s-1, vmin = %.3f m s-1, Kmin = %.3f m2 s-1' 
          % (min(u), min(v), min(K)))
    print()

    # we do the transpose here to make the time series the first axis
    return z.T, (u.T, v.T, K.T)


def psi(x):
    '''
    Businger–Dyer relationship
    x =  z / mol
    '''
    xi = np.where(
            x > 0.0, 
            np.nan, 
            np.power(1.0 - 16.0 * x, 0.25, dtype = complex).real)
    return np.where(
            x > 0.0, 
            5.0 * x, 
            -2.0 * np.log(0.5 * (1.0 + xi)) - np.log(0.5 * (1.0 + xi**2))
            +2.0 * np.arctan(xi) - 0.5 * np.pi)

def phi(x):
    '''
    Businger–Dyer relationship
    x =  z / mol 
    '''
    return np.where(
            x > 0.0, 
            1.0 + 5.0 * x, 
            np.power(1.0 - 16.0 * x, -0.5, dtype = complex).real)


def absu_oaahoc(z0, zm, absum, ustar, tke, mol, n):

    cl = 0.845
    cm = 0.0856
    z = np.linspace(z0, zm, n)
    dz = np.diff(z,axis=0)

    x = np.ones(n)
    y = np.ones(n)

    tke0 = 0.2
    x[0] = 0.0
    y[0] = ustar / np.sqrt(tke0)

    for i in range(n-1):

        print(i)
        y[i+1] = newton(
                tke_balance, 
                y[i], 
                tke_balance_prime, 
                args=(mol, z[i+1]),
                maxiter = 100)
        x[i+1] = x[i] + dz[i] / cm / cl / z[i+1] * y[i+1]

    absu = x * ustar
    tke  = (ustar / y)**2

    return z, absu, tke

def tke_balance(y, mol, z):

    ce = 0.845
    cl = 0.845
    cm = 0.0856

    a = cm * cl * z / mol
    b = ce / cl

    return y**4 + a * y**3 - b

def tke_balance_prime(y, mol, z):

    ce = 0.845
    cl = 0.845
    cm = 0.0856

    a = cm * cl * z / mol
    b = ce / cl

    return 4.0 * y**3 + 3.0 * a * y**2



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    z0, zm, absum, ustar, tke, mol, n = 0.10, 6.0, 3.0, 0.2, 10.0, -200, 100
    z, u, tke = absu_oaahoc(z0, zm, absum, ustar, tke, mol, n)

    #print(tke_balance(ustar/np.sqrt(tke), z0, zm, mol, 1.0))

    plt.plot(u,z)
    plt.show()

#    fig, axs = plt.subplots(2)
#    for mol in [-500,-400,-100,-10,10,100,400,500,1000]:
#        z, (u, v, K) = vertical_profiles(
#                n=100,
#                meas_height=5.0,
#                wind=(3.0, 1.0),
#                ustar=0.25,
#                mol=mol)
#        axs[0].plot(u,z)
#        axs[0].set_xlabel('u [m/s]')
#        axs[0].set_ylabel('z [m]')
#        axs[1].plot(K,z)
#        axs[1].set_xlabel('K [m2 s-1]')
#        axs[1].set_ylabel('z [m]')
#    plt.show()

