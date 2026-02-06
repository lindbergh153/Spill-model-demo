"""
Oil Transport Functions
=======================

Advection-diffusion transport for oil particles in the water column
and on the sea surface using random walk diffusion.

References
----------
Visser, A.W. (1997). Using random walk models to simulate the vertical
    distribution of particles in a turbulent water column.
    Marine Ecology Progress Series, 158, 275-281.

"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from SPM import ModelParams

# Coordinate conversion constant
degree_to_m = 111120  # meters per degree latitude


def transport_underwater(u_water, v_water, us, p, y_sub, dt_sub, Dxyz):
    """
    A function to compute the advective velocity in
    x- (degree/hour), y- (degree/hour), and z- (m/s) direction

    :param u_water: float
        Current velocity in the x-direction of the particle (m/s)
    :param v_water: float
        Current velocity in the y-direction of the particle (m/s)
    :param us: float
        Rise velocity of the particle (m/s)
    :param p: SPM.ModelParams
        Container for the fixed model parameters
    :param y_sub: ndarray
        Current value for the state space vector of a particle in Cloud.

    :return u_x: float
        Derivative of current velocity corrected by simulation step size
        and relation between degree and meters in x-direction (degree/hour)
    :return u_y: float
        Derivative of current velocity corrected by simulation step size
        and relation between degree and meters in y-direction (degree/hour)
    :return u_z: float
        Derivative of current velocity corrected by simulation step size in z-direction (meter/hour)

    """
    # Set velocities as zero when particle mass is zero
    if y_sub[4:-1].sum == 0:
        u_x, u_y, u_z = 0, 0, 0
    elif u_water < -10 or v_water < -10:
        u_x, u_y, u_z = 0, 0, 0
    else:
        # Horizontal velocities (degree/hour) = horizontal velocities (m/s) * 3600 (s/h) / 111100 (degree/m)
        ux_diff = np.random.uniform(-1, 1) * DDxy(Dxyz[0], dt_sub, p)
        uy_diff = np.random.uniform(-1, 1) * DDxy(Dxyz[1], dt_sub, p)
        # uz_diff = np.random.uniform(-1, 1) * DDz(Dxyz[2], dt_sub, p)

        u_x = (u_water + ux_diff) * p.time_interval / (degree_to_m * np.cos(np.deg2rad(y_sub[1])))
        u_y = (v_water + uy_diff) * p.time_interval / degree_to_m
        u_z = -us * p.time_interval

    return u_x, u_y, u_z


def transport_surface(para_current, para_wind, u_water, v_water, u_wind, v_wind, p, y_sur, dt_sur, Dxyz):
    """
    A function to compute derivatives for surface advection in x- and y- (degree/hour) directions

    :param para_current: float, default=1
        Current drift coefficient
    :param para_wind: float, default=0.03
        Wind drift coefficient
    :param u_water: float
        Current velocity in the x-direction of the particle (m/s)
    :param v_water: float
        Current velocity in the y-direction of the particle (m/s)
    :param u_wind: float
        Wind velocity in the x-direction of the particle (m/s)
    :param v_wind: float
        Wind velocity in the y-direction of the particle (m/s)
    :param p: SPM.ModelParams
        Container for the fixed model parameters
    :param y_sur: ndarray
        Current value for the state space vector of a slick.

    :return u_x: float
        Derivative of current velocity corrected by simulation step size
        and relation between degree and meters in x-direction (degree/hour)
    :return u_y: float
        Derivative of current velocity corrected by simulation step size
        and relation between degree and meters in x-direction (degree/hour)
    :return u_z: float
        Derivative of current velocity corrected by simulation step size in z-direction (meter/hour)

    """

    # Set velocities as zero when slick mass is zero
    if y_sur[4:-1].sum == 0:
        u_x, u_y, u_z = 0, 0, 0
    elif u_water < -10 or v_water < -10:
        u_x, u_y, u_z = 0, 0, 0
    else:
        # Horizontal velocities (degree/hour) =
        # [Horizontal current velocities (m/s) * Current drift coefficient +
        # Wind velocities (m/s) * Wind drift coefficient ] * 3600 (s/h) / 111100 (degree/m)

        ux_diff = np.random.uniform(-1, 1) * DDxy(Dxyz[0], dt_sur, p)
        uy_diff = np.random.uniform(-1, 1) * DDxy(Dxyz[1], dt_sur, p)

        wind_speed = (u_wind ** 2 + v_wind ** 2) ** 0.5
        para = 8 * wind_speed ** 0.5 if wind_speed < 25 else 0
        theta = np.degrees(np.deg2rad(40 - para))

        u_wind = u_wind * np.cos(np.deg2rad(theta)) + v_wind * np.sin(np.deg2rad(theta))
        v_wind = -u_wind * np.sin(np.deg2rad(theta)) + v_wind * np.cos(np.deg2rad(theta))

        u_x = (para_current * u_water + para_wind * u_wind + ux_diff) * p.time_interval \
              / (degree_to_m * np.cos(np.deg2rad(y_sur[1])))
        u_y = (para_current * v_water + para_wind * v_wind + uy_diff) * p.time_interval / degree_to_m
        u_z = 0

    return u_x, u_y, u_z


def DDxy(D, dt, p):
    return (6 * D / 10000 / dt / p.time_interval) ** 0.5


def DDz(D, dt, p):
    return (6 * D / 10000 / dt / p.time_interval) ** 0.5


def current_velocity(u_water, v_water, y, t):
    """
    Get the current velocity (m/s) from RegularGridInterpolator

    :param u_water: RegularGridInterpolator
        A method to obtain velocity at specific time and location
    :param v_water: RegularGridInterpolator
        A method to obtain velocity at specific time and location
    :param y: ndarray
        Current value for the state space vector of particle or Slick.
    :param t: float
        The right time corresponding to the current data (hour)


    :return u: float
        Current velocity in the x-direction (m/s)
    :return v: float
        Current velocity in the y-direction (m/s)

    """
    # Extract location from the state space vector
    lat, lon, depth = y[1], y[0], y[2]

    try:
        if depth < 0: depth = 0
        u, v = u_water([t, depth, lat, lon])[0], v_water([t, depth, lat, lon])[0]
    except ValueError:
        print(t, depth, lat, lon)
        raise Exception('current_velocity in transport_functions is wrong')

    return u, v


def wind_velocity(u_wind, v_wind, y, t):
    """
    Get the wind velocity (m/s) from RegularGridInterpolator

    :param u_wind: RegularGridInterpolator
        A method to obtain velocity at specific time and location
    :param v_wind: RegularGridInterpolator
        A method to obtain velocity at specific time and location
    :param y: ndarray
        Current value for the state space vector of particle or Slick.
    :param t: float
        The right time corresponding to the wind data (hour)

    :return u: float
        Wind velocity in the x-direction (m/s)
    :return v: float
        Wind velocity in the y-direction (m/s)

    """
    # Extract location from the state space vector
    lat, lon, depth = y[1], y[0], y[2]

    try:
        u, v = u_wind([t, lat, lon])[0], v_wind([t, lat, lon])[0]
    except ValueError:
        print(t, lat, lon)
        raise Exception('wind_velocity in transport_functions is wrong')

    return u, v
