"""
Example module demonstrating the calculation of flux footprints.
"""

import matplotlib.pyplot as plt
import numpy as np

from bldfm.pbl_model import vertical_profiles
from bldfm.utils import ideal_source, get_logger, get_source_area
from bldfm.solver import steady_state_transport_solver

logger = get_logger("Minimal footprint_example")

nx = 512
ny = 256
nz = 32
domain = 100.0, 700.0
meas_pt = 50.0, 0.0
meas_height = 10.0
wind = 0.0, -6.0
ustar = 0.5

area = np.zeros([ny, nx])

z, profs = vertical_profiles(nz, meas_height, wind, ustar)

grid, conc, flx = steady_state_transport_solver(
    area, z, profs, domain, nz, meas_pt=meas_pt, footprint=True
)

if __name__ == "__main__":

    X, Y, Z = grid

    plt.figure()
    plt.pcolormesh(X, Y, flx)
    plt.title("Flux footprint")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    cbar = plt.colorbar()
    cbar.set_label("$m^{-2}$")
    plt.savefig("plots/flux_footprint.png")

    flx_rescaled = get_source_area(flx, flx)

    # Contribution contour levels and colors
    levels = [0.25, 0.5, 0.75]
    contour_colors = ['white', 'magenta', 'cyan']

    plt.figure()
    pcm = plt.pcolormesh(X, Y, flx, shading='auto')
    cs = plt.contour(X, Y, flx_rescaled, levels=levels, colors=contour_colors)
    plt.clabel(cs, fmt='%.2f')
    plt.title("Flux footprint with contribution contours")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    cbar = plt.colorbar(pcm)
    cbar.set_label("$m^{-2}$")
    plt.savefig("plots/flux_footprint_contours.png")

    # Circular contribution contours centered at tower
    xm, ym = meas_pt
    g = -((X - xm)**2 + (Y - ym)**2)
    flx_rescaled_circular = get_source_area(flx, g)

    plt.figure()
    pcm = plt.pcolormesh(X, Y, flx, shading='auto')
    cs = plt.contour(X, Y, flx_rescaled_circular, levels=levels, colors=contour_colors)
    plt.clabel(cs, fmt='%.2f')
    plt.title("Flux footprint with circular contribution contours")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    cbar = plt.colorbar(pcm)
    cbar.set_label("$m^{-2}$")
    plt.tight_layout()
    plt.savefig("plots/flux_footprint_circular_contours.png")

    # Upwind contribution contours (slope only)
    # Wind direction unit vectors
    u, v = wind
    wind_speed = np.sqrt(u**2 + v**2)
    u_hat, v_hat = u / wind_speed, v / wind_speed
    # Distance along wind direction (positive = downwind from tower)
    g = u_hat * (X - xm) + v_hat * (Y - ym)
    flx_rescaled_slope = get_source_area(flx, g)

    plt.figure()
    pcm = plt.pcolormesh(X, Y, flx, shading='auto')
    cs = plt.contour(X, Y, flx_rescaled_slope, levels=levels, colors=contour_colors)
    plt.clabel(cs, fmt='%.2f')
    plt.title("Flux footprint with upwind contribution contours")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    cbar = plt.colorbar(pcm)
    cbar.set_label("$m^{-2}$")
    plt.tight_layout()
    plt.savefig("plots/flux_footprint_upwind_contours.png")

    # Crosswind contribution contours (ridge only)
    # Perpendicular distance from wind axis through tower
    g = -(-v_hat * (X - xm) + u_hat * (Y - ym))**2
    flx_rescaled_ridge = get_source_area(flx, g)

    plt.figure()
    pcm = plt.pcolormesh(X, Y, flx, shading='auto')
    cs = plt.contour(X, Y, flx_rescaled_ridge, levels=levels, colors=contour_colors)
    plt.clabel(cs, fmt='%.2f')
    plt.title("Flux footprint with crosswind contribution contours")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    cbar = plt.colorbar(pcm)
    cbar.set_label("$m^{-2}$")
    plt.tight_layout()
    plt.savefig("plots/flux_footprint_crosswind_contours.png")

    # Spiral contribution contours centered at tower
    # Symmetric left/right spiral forming a ridge facing upwind
    theta = np.arctan2(Y - ym, X - xm)
    # Upwind direction angle (opposite of wind direction)
    theta_upwind = np.arctan2(-v, -u)
    # Angular deviation from upwind direction
    theta_rel = theta - theta_upwind
    # Wrap to [-pi, pi]
    theta_rel = np.arctan2(np.sin(theta_rel), np.cos(theta_rel))
    # Ridge: highest along upwind axis, symmetric left/right
    g = -np.abs(theta_rel)
    flx_rescaled_spiral = get_source_area(flx, g)

    plt.figure()
    pcm = plt.pcolormesh(X, Y, flx, shading='auto')
    cs = plt.contour(X, Y, flx_rescaled_spiral, levels=levels, colors=contour_colors)
    plt.clabel(cs, fmt='%.2f')
    plt.title("Flux footprint with sector contribution contours")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    cbar = plt.colorbar(pcm)
    cbar.set_label("$m^{-2}$")
    plt.tight_layout()
    plt.savefig("plots/flux_footprint_spiral_contours.png")

# %%
