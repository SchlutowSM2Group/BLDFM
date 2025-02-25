import numpy as np

def psi(x):
    '''
    Businger–Dyer relationship
    x =  z / L 
    '''
    xi = np.where( x>0.0, np.nan, (1.0-16.0*x)**0.25 )
    xi = xi.real
    return np.where( x>0.0, 
                     5.0*x, 
                    -2.0*np.log(0.5*(1.0+xi))-np.log(0.5*(1.0+xi**2))
                    +2.0*np.arctan(xi)-0.5*np.pi)

def phi(x):
    '''
    Businger–Dyer relationship
    x =  z / L 
    '''
    return np.where( x>0.0, 1.0+5.0*x, (1.0-16.0*x)**-0.5 )


def vertical_profiles(nz,zmx,um,vm,ustar,mol=1e9,prsc=0.8,constant=False):

    '''
    z     - array of vertical grid points
    zm    - height of measurement starting 
    um    - zonal wind at measurement height 
    vm    - meridional wind at measurement height
    ustar - friction velocity
    mol   - Monin Obukhov length
    prsc  - Prandtl/Schmidt number

    MOST based on https://doi.org/10.1023/A:1018991015119
    '''

    kap = 0.4 # Karman constant

    if constant:

        Km = kap * ustar * zmx / prsc
        z = np.linspace( 0.0, zmx, nz )
        u = um * np.ones(nz)
        v = vm * np.ones(nz)
        K = Km * np.ones(nz)

    else:
        absum = np.sqrt( um**2 + vm**2 ) # absolute wind at zm
        z0 = zmx * np.exp( -kap * absum / ustar + psi( zmx/mol ) ) # roughness length

        z = np.linspace( z0, zmx, nz )

        absu = ustar / kap * ( np.log( z/z0 ) + psi( z/mol ) ) 

        u = um / absum * absu
        v = vm / absum * absu

        K = kap * ustar * z / phi( z/mol ) / prsc

    print('umax, vmax, Kmax', max(u), max(v), max(K))
    print('umin, vmin, Kmin', min(u), min(v), min(K))

    # we do the transpose here to make the time series the first axis
    return z.T, u.T, v.T, K.T

def compute_wind_fields(u_rot, wind_dir):
    wind_dir = np.deg2rad(wind_dir)
    u = u_rot * np.sin(wind_dir)
    v = u_rot * np.cos(wind_dir)

    return u, v


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2)
    for mol in [-500,-400,-100,-10,-5,5,10,100,400,500,1000]:
        print('mol = ',mol)
        z,u,v,K = vertical_profiles(nz=100,zmx=5.0,um=3.0,vm=1.0,ustar=0.25,mol=mol)
        print("z0  = ",z[0])
        print("psi = ", psi( 5.0/mol ))
        axs[0].plot(u,z)
        axs[0].set_xlabel('u [m/s]')
        axs[0].set_ylabel('z [m]')
        axs[1].plot(K,z)
        axs[1].set_xlabel('K [m2 s-1]')
        axs[1].set_ylabel('z [m]')
    plt.show()

