import numpy as np
import matplotlib.pyplot as plt

from src.pbl_model import vertical_profiles
from src.utils import ideal_source
from src.solver import steady_state_transport_solver
from src.ffm_kormann_meixner import estimateFootprint as FKM

nxy = 512, 512
nz = 128
modes = 1024, 1024
# modes       = 128, 128
domain = 200.0, 600.0
fetch = 1000.0
meas_pt = 100.0, 0.0
meas_height = 10.0
wind = 0.0, -6.0
ustar = 0.4
z0 = 0.1
mol = 20.0

wd = np.arctan(wind[0] / wind[1]) * 180.0 / np.pi
umean = np.sqrt(wind[0] ** 2 + wind[1] ** 2)

############################################################
###  BLDFM
############################################################

surf_flx = ideal_source(nxy, domain)

z, profs = vertical_profiles(nz, meas_height, wind, ustar, z0=z0, mol=mol)

srf_flx, bg_conc, conc, flx = steady_state_transport_solver(
    surf_flx,
    z,
    profs,
    domain,
    modes=modes,
    meas_pt=meas_pt,
    footprint=True,
    fetch=fetch,
)

############################################################
###  Kljun's FFP footprint model
############################################################

#
# FFP_res = FFP(
#        zm = meas_height,
#        z0 = z0,
#        umean = umean,
#        h = 2000.,
#        ol = mol,
#        sigmav = 0.2,
#        ustar = 0.437,
#        wind_dir = 180.0,
#        nx = 1000,
#        rs= [20., 40., 60., 80.],
#        fig = False,
#        crop = 0)
#
# msk  = (FFP_res["y_2d"] <= domain[0]/2) & \
#       (FFP_res["y_2d"] >= -domain[0]/2) & \
#       (FFP_res["x_2d"] >=  0.0) & \
#       (FFP_res["x_2d"] <=  domain[1])
#
# f_2d = np.ma.masked_array(FFP_res["f_2d"], msk)
##f_2d = FFP_res["f_2d"][msk]
##x_2d = FFP_res["x_2d"][msk]
##y_2d = FFP_res["y_2d"][msk]
#
# print(f_2d.shape)
##print(x_2d.shape)
##print(y_2d.shape)

#
############################################################
### Korman and Meixner's footprint model
############################################################

grid_x, grid_y, grid_ffm = FKM(
    zm=meas_height,
    z0=z0,
    ws=umean,
    ustar=ustar,
    mo_len=mol,
    sigma_v=0.5 * ustar,
    grid_domain=[0, domain[0], 0, domain[1]],
    grid_res=domain[0] / nxy[0],
    mxy=meas_pt,
    wd=wd,
)


############################################################
### plotting
############################################################
if __name__ == "__main__":
    shrink = 0.5
    cmap = "turbo"

    fig, axs = plt.subplots(1, 2, figsize=[6, 8], sharey=True, layout="constrained")

    # BLDFM
    plot = axs[0].imshow(
        flx, origin="lower", cmap=cmap, extent=[0, domain[0], 0, domain[1]]
    )

    axs[0].set_title("BLDFM")
    axs[0].set_xlabel("x [m]")
    axs[0].set_ylabel("y [m]")

    # FFP
    # plot = axs[1].pcolormesh(x_2d, y_2d, f_2d)
    # plot = axs[1].imshow(
    #        f_2d,
    #        origin="lower",
    #        cmap=cmap,
    #        extent=[0,domain[0],0,domain[1]])
    #
    # axs[1].set_title("FFP")
    # axs[1].set_xlabel("x [m]")

    # FKM
    plot = axs[1].imshow(
        grid_ffm, origin="upper", cmap=cmap, extent=[0, domain[0], 0, domain[1]]
    )

    axs[1].set_title("FKM")
    axs[1].set_xlabel("x [m]")
    cbar = fig.colorbar(plot, ax=axs, shrink=shrink, location="bottom")
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)
    cbar.set_label("$m^{-2}$")

    # plt.show()
    plt.savefig("plots/comparison_footprint.png", dpi=300)
