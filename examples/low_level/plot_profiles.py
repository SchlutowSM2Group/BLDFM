"""Plot vertical profiles of wind and eddy diffusivity under MOST closure."""

import matplotlib.pyplot as plt
import numpy as np

from bldfm.utils import get_logger
from bldfm.pbl_model import vertical_profiles

logger = get_logger("plot_profiles")


if __name__ == "__main__":
    logger.info("Plotting vertical profiles for MOST closure")

    meas_height = 10.0
    mols = [-1000, -100, -10, 10, 100, 1000]

    colormap = plt.cm.turbo
    colors = [colormap(i) for i in np.linspace(0, 1, len(mols))]

    fig, axs = plt.subplots(1, 2)

    for mol, color in zip(mols, colors):

        z, (u, v, Kx, Ky, Kz) = vertical_profiles(
            n=8,
            meas_height=meas_height,
            wind=(5.0, 0.0),
            z0=0.1,
            closure="MOST",
            mol=mol,
        )
        K = Kz

        axs[0].plot(u, z, "+", label="L = " + str(mol) + " m", color=color)
        axs[1].plot(K, z, "+", color=color)

    axs[0].axhline(meas_height, linestyle="dashed", color="gray", label="meas_height")
    axs[1].axhline(meas_height, linestyle="dashed", color="gray")
    axs[0].set_ylabel("z [m]")
    axs[0].set_xlabel("u [m s-1]")
    axs[0].legend()
    axs[1].set_xlabel("K [m2 s-1]")
    fig.suptitle("Profiles for MOST")

    plt.savefig("plots/examples_low_level_most_profiles.png")
