"""
SPM Integration Functions
=========================

ODE integration and derivative functions for underwater particle
transport and surface slick evolution.

"""

from __future__ import annotations
from scipy import integrate
from shapely.geometry import Point

from conversion_functions import convert_loc
from transport_functions import *
from weathering_functions import *
from SPM_utilities import Slick


def ic_underwater(cloud, particle_release_time, release_location):
    """
    Generate the initial condition for subsurface oil simulation

    :param cloud: SPM_utilities.ParticleCloud
        Object describing the properties of the cloud (a collection of particles with the same size).
    :param particle_release_time: float
        The time that oil released from the near-field (hour)
    :param release_location: list
        The initial location of a particle cloud: [Lat, lon, depth]

    :return t0_sub: float
        The time that oil released from the near-field (hour)
    :return y0_sub: ndarray
        Initial condition for the state space along the cloud trajectory
        y0[0:3]: x, y, z
        y0[3:3+i]: each PC's mass in a particle (i in total)
        y0[3+i]: particle heat

    """

    # Extract the particle properties at the exit point
    ze, me, cp, Te, te = cloud.particle.z, cloud.particle.m, cloud.particle.cp, \
        cloud.particle.T, cloud.particle.t

    # Convert the particle location at the exit point from xy-coordinate to geo-coordinate
    ye, xe = convert_loc(release_location, cloud.particle)

    # Add the particle location at the exit point to a ndarray
    X0 = np.array([xe, ye, ze])

    # Assemble the state space of cloud into a ndarray
    y0_sub = np.hstack((X0, me, Te * np.sum(me) * cp))

    # Get the exit time
    t0_sub = particle_release_time

    return t0_sub, y0_sub


def calculation_underwater(profile, cloud, p, t0_sub, y0_sub, dt23, dt_sub, D_xyz_sub, cond='unsteady'):
    """
    Calculate the trajectory and fate of a single particle rising through the water column

    :param profile: list
        A data ensemble required by far-field modeling, including:
        1) interpolation_water_u, 2) interpolation_water_v, 3) interpolation_wind_u,
        4) interpolation_wind_v, 5) min_time_current, 6) max_time_current, 7) min_depth_current,
        8) max_depth_current, 9) min_lat_current,  10) max_lat_current, 11) min_lon_current,
        12) max_lon_current, 13) min_time_wind, 14) max_time_wind, 15) min_lat_wind,
        16) max_lat_wind, 17) min_lon_wind, 18) max_lon_wind, 19) time_interval,
        20) profile1d.get_values: P, Ta, Sa, Ua, Va, Wa.
    :param cloud: SPM_utilities.ParticleCloud
        Object describing the properties of the cloud (a collection of particles with the same size).
    :param p: SPM.ModelParams
        Container for the fixed model parameters
    :param t0_sub: float
        The initial time for subsurface oil simulation (hour)
    :param y0_sub: ndarray
        The initial condition for subsurface oil simulation
    :param dt23: float
        End time of simulation (hours). Also relate to the initial time of the current data.
    :param dt_sub: float
        Time step of far-field model for underwater oil (hours)
    :param D_xyz_sub: list
        A container includes diffusion coefficients for subsurface cloud
    :param cond: string, default 'unsteady'
        If cond (condition) is 'unsteady', underwater current field is set as unsteady flows;
        otherwise, underwater current field is steady field.

    :return t_sub: ndarray
        Times (h) associated with the state space of the particle cloud
    :return y_sub: ndarray
        The state space of the particle cloud
        y[0:3]: x, y, z
        y[3:3+i]: each PC's mass in a particle (i in total)
        y[3+i]: particle heat
    :return cloud: SPM_utilities.ParticleCloud
        Object describing the properties of the cloud (a collection of particles with the same size).

    """

    # Create an integrator object: use "vode" with "Backward differentiation formula" for stiff ODEs
    # Select derivatives based on the stability of inputted flow   bdf Adams
    if cond == 'unsteady':
        r = integrate.ode(derivs_underwater).set_integrator('vode', method='bdf', atol=1e-3,
                                                            rtol=1e-3, order=4, max_step=dt_sub)
    else:
        r = integrate.ode(derivs_underwater_steady).set_integrator('vode', method='bdf', atol=1e-6,
                                                                   rtol=1e-3, order=4, max_step=dt_sub)

    # Initialize the state space
    r.set_initial_value(y0_sub, t0_sub)

    # Set passing variables for derivs method
    r.set_f_params(profile, cloud, p, dt_sub, D_xyz_sub)

    # Create lists to store the solution and its associate time
    t = [t0_sub]
    y = [y0_sub]

    # Integrate to the sea surface
    k = 0
    f = 1
    psteps = 30.
    stop = False
    while r.successful() and not stop:

        # Print progress to the screen
        m0 = np.sum(y[0][3:-1])
        mt = np.sum(y[-1][3:-1])
        f = mt / m0

        if np.remainder(float(k), psteps) == 0.:
            print('    Depth: %g (m), Lat: %g, Lon: %g, t:  %g (hrs), k: %d, f: %g (--), phase: %d'
                  % (r.y[2], r.y[1], r.y[0], t[-1], k, f, cloud.q_type))

        # Perform one step of the integration
        r.integrate(t[-1] + dt_sub, step=True)

        # Loop all the PC's masses, avoid them become negative
        r.y[3:-1][r.y[3:-1] < 0] = 0

        t.append(r.t)
        y.append(r.y)
        k += 1
        # Update the properties of the particle cloud, i.e., particle mass, x, y, z, dt
        cloud.update(y[-1][3:-1], y[-1][0], y[-1][1], y[-1][2], dt_sub)

        # Evaluate stop criteria
        if r.successful():
            # Check if particle reached the surface
            if r.y[2] <= p.z_min:
                stop = True
                # Mark the cloud as not underwater
                cloud.underwater = False
            # Check if particle dissolved (us = 0 or based on fdis)
            elif f < cloud.particle.fdis:
                stop = True
            # Check if simulation exceed the user-defined end time
            elif t[-1] >= dt23:
                stop = True
            # Check if cloud pass through the data boundary
            # elif r.y[0] <= p.left_bound or r.y[0] >= p.right_bound \
            #         or r.y[1] <= p.lower_bound or r.y[1] >= p.upper_bound:
            #     stop = True
            #     print('lat:{1}, lon:{0}'.format(r.y[1], r.y[0]))
            #     print('reach the boundary: left:{0}, right:{1}, upper:{2}, lower:{3}'.
            #           format(p.left_bound, p.right_bound, p.upper_bound, p.lower_bound))

    # Remove any negative depths due to overshooting the surface
    t = np.array(t)
    y = np.array(y)
    rows = y[:, 2] >= 0
    t_sub = t[rows]
    y_sub = y[rows, :]

    print('    Depth: %g (m), Lat: %g, Lon: %g, t:  %g (hrs), k: %d, f: %g (--), phase: %d'
          % (y_sub[-1, 2], y_sub[-1, 1], y_sub[-1, 0], t[-1], k, f, cloud.q_type))

    return t_sub, y_sub, cloud


def derivs_underwater(t_sub, y_sub, profile, cloud, p, dt_sub, Dxyz_sub):
    """
    Assemble the right hand side of the ODE for the trajectory and fate of
    a single particle rising through the water column.
    This function is used for an unsteady current field.

    :param t_sub: float
        Current value for the independent variable (time in hours).
    :param y_sub: ndarray
        Current value for the state space vector of a particle in Cloud.
    :param profile: list
        A data ensemble required by far-field modeling, including:
        1) interpolation_water_u, 2) interpolation_water_v, 3) interpolation_wind_u,
        4) interpolation_wind_v, 5) min_time_current, 6) max_time_current, 7) min_depth_current,
        8) max_depth_current, 9) min_lat_current,  10) max_lat_current, 11) min_lon_current,
        12) max_lon_current, 13) min_time_wind, 14) max_time_wind, 15) min_lat_wind,
        16) max_lat_wind, 17) min_lon_wind, 18) max_lon_wind, 19) time_interval,
        20) profile1d.get_values: P, Ta, Sa, Ua, Va, Wa.
    :param cloud: SPM_utilities.ParticleCloud
        Object describing the properties of the cloud (a collection of particles with the same size).
    :param p: SPM.ModelParams
        Container for the fixed model parameters

    :return yp_sub: ndarray
        A set of derivatives on the right hand side of the ODE for the trajectory
        and fate of a single particle rising through the water column.
        The state space of the particle cloud:
        yp[0:3]: derivatives for advection
        yp[3:3+i]: derivatives for each PC's dissolution (i in total)
        yp[3+i]: derivatives for particle heat
    """

    # Set up a container for derivatives
    yp_sub = np.zeros(y_sub.shape)

    # Get the right time corresponding to the current data
    t_current = t_sub + p.dt02_current

    # Get RegularGridInterpolator for current velocities in x- and y- directions
    u_water, v_water = current_velocity(profile[0], profile[1], y_sub, t_current)

    # Extract the state space variables for depth, masses of PC, and temperature
    z = y_sub[2]
    m = y_sub[3:-1]
    T = y_sub[-1] / (np.sum(m) * cloud.particle.cp)

    # Get the current ambient data
    P, Ta, Sa, Ua, Va, Wa = profile[-1](z)

    if not np.all(m == 0):
        # Get the physical particle properties
        us, rho_p, A, Cs, beta, beta_T, T = cloud.particle.properties(m, T, P, Sa, Ta, t_sub)

        # Set derivatives for subsurface advection
        yp_sub[0], yp_sub[1], yp_sub[2] = transport_underwater(u_water, v_water, us, p, y_sub, dt_sub, Dxyz_sub)

        k_bio = cloud.particle.FluidParticle.k_bio
        md_biodeg = -k_bio * m

    else:
        us, A, Cs, beta = 0, 0, np.zeros_like(m), np.zeros_like(m)

        yp_sub[0], yp_sub[1], yp_sub[2] = 0, 0, 0

        md_biodeg = 0

    # Set derivatives for particle dissolution
    if len(Cs) > 0:
        md_diss = - A * beta * Cs * p.time_interval
    else:
        md_diss = 0

    yp_sub[3:-1] = md_diss + md_biodeg

    # Account for heat lost due to decrease in mass
    yp_sub[-1] += cloud.particle.cp * np.sum(md_diss) * T

    # Return the derivatives
    return yp_sub


def derivs_underwater_steady(t_sub, y_sub, profile, cloud, p, dt_sub, Dxyz_sub):
    """
    Assemble the right hand side of the ODE for the trajectory and fate of
    a single particle rising through the water column.
    This function is used for a steady current field.

    :param t_sub: float
        Current value for the independent variable (time in hours).
    :param y_sub: ndarray
        Current value for the state space vector of a particle in Cloud.
    :param profile: list
        A data ensemble required by far-field modeling, including:
        1) interpolation_water_u, 2) interpolation_water_v, 3) interpolation_wind_u,
        4) interpolation_wind_v, 5) min_time_current, 6) max_time_current, 7) min_depth_current,
        8) max_depth_current, 9) min_lat_current,  10) max_lat_current, 11) min_lon_current,
        12) max_lon_current, 13) min_time_wind, 14) max_time_wind, 15) min_lat_wind,
        16) max_lat_wind, 17) min_lon_wind, 18) max_lon_wind, 19) time_interval,
        20) profile1d.get_values: P, Ta, Sa, Ua, Va, Wa.
    :param cloud: SPM_utilities.ParticleCloud
        Object describing the properties of the cloud (a collection of particles with the same size).
    :param p: SPM.ModelParams
        Container for the fixed model parameters

    :return yp_sub: ndarray
        A set of derivatives on the right hand side of the ODE for the trajectory
        and fate of a single particle rising through the water column.
        The state space of the particle cloud:
        yp[0:3]: derivatives for advection
        yp[3:3+22]: derivatives for each PC's dissolution
        yp[3+22]: derivatives for particle heat

    """

    # Set up a container for derivatives
    yp_sub = np.zeros(y_sub.shape)

    # Extract the state space variables for depth, masses of PC, and temperature
    z = y_sub[2]
    m = y_sub[3:-1]
    T = y_sub[-1] / (np.sum(m) * cloud.particle.cp)

    # Get the current ambient data
    P, Ta, Sa, u_water, v_water, Wa = profile[-1](z)

    # Get the physical particle properties
    us, rho_p, A, Cs, beta, beta_T, T = cloud.particle.properties(m, T, P, Sa, Ta, t_sub)

    # Set derivatives for subsurface advection
    yp_sub[0], yp_sub[1], yp_sub[2] = transport_underwater(u_water, v_water, us, p, y_sub, dt_sub, Dxyz_sub)

    # Set derivatives for particle dissolution
    if len(Cs) > 0:
        md_diss = - A * beta * Cs * p.time_interval
    else:
        md_diss = 0
    yp_sub[3:-1] = md_diss

    # Account for heat lost due to decrease in mass
    yp_sub[-1] += cloud.particle.cp * np.sum(md_diss) * T

    # Return the derivatives
    return yp_sub


def ic_surface(cloud, surfacing_time, p, profile, y_sub):
    """
    Generate the initial condition for surface oil simulation

    :param cloud: SPM_utilities.ParticleCloud
        Object describing the properties of the cloud (a collection of particles with the same size).
    :param surfacing_time: float
        Time that particle reaches the sea surface (h)
    :param p: SPM.ModelParams
        Container for the fixed model parameters
    :param profile: list
        A data ensemble required by far-field modeling, including:
        1) interpolation_water_u, 2) interpolation_water_v, 3) interpolation_wind_u,
        4) interpolation_wind_v, 5) min_time_current, 6) max_time_current, 7) min_depth_current,
        8) max_depth_current, 9) min_lat_current,  10) max_lat_current, 11) min_lon_current,
        12) max_lon_current, 13) min_time_wind, 14) max_time_wind, 15) min_lat_wind,
        16) max_lat_wind, 17) min_lon_wind, 18) max_lon_wind, 19) time_interval,
        20) profile1d.get_values: P, Ta, Sa, Ua, Va, Wa.
    :param y_sub: ndarray
        The state space of the particle cloud
        y[0:3]: x, y, z
        y[3:3+i]: each PC's mass in a particle (i in total)
        y[3+i]: particle heat

    :return t0_sur: float
        The time (hour) that oil reach sea surface is set as
        the initial time for surface oil simulation
    :return y0_sur:
        The state space of the slick
        y[0:3]: x, y, z
        y[3]: A
        y[4:4+i]: each PC's mass in a slick (i in total)
        y[4+i]: slick water content
    :return slick: SPM_utilities.Slick
        Object describing the properties of the oil slick.

    """
    # Extract the surface water properties from SPM.ModelParams
    T_sur, S_sur, P_sur = p.T_sur, p.S_sur, p.P_sur

    # Compute the initial area of a slick
    A0 = initial_area(cloud, T_sur, S_sur, P_sur)

    # Assume oil particles do not occur emulsification
    Y0 = 0

    # Compute the maximum water content of a slick
    Ymax = max_water_fraction(cloud)

    # Compute the initial viscosity and density of a slick
    vis0 = cloud.particle.FluidParticle.viscosity(cloud.m_particle, T_sur, P_sur)
    rho0 = cloud.particle.FluidParticle.density(cloud.m_particle, T_sur, P_sur)
    sigma0 = cloud.particle.FluidParticle.interface_tension(cloud.m_particle, T_sur, S_sur, P_sur)

    # Compute the wave height
    wh0 = get_wave_height(profile, y_sub, surfacing_time, p)

    # Compute the time required to form emulsion
    time_emulsion, emul_type = get_time_emulsion(cloud.mf, vis0, rho0, wh0)

    # Create a Slick object
    slick = Slick(cloud.particle, cloud.m_Cloud, cloud.x, cloud.y, cloud.z,
                  A0, Y0, Ymax, vis0, rho0, sigma0, time_emulsion, emul_type)

    # Horizontally merge those variables to create the initial condition for surface oil simulation
    y0_sur = np.hstack([cloud.x, cloud.y, p.z_min, slick.A, cloud.m_Cloud, slick.Y])

    # Get the time that oil reach the sea surface
    t0_sur = surfacing_time

    return t0_sur, y0_sur, slick


def calculation_surface(profile, slick, p, t0_sur, y0_sur, dt23, dt_sur, Dxyz_sur,
                        para_current, para_wind, cond='unsteady', shore_polygon=None):
    """
    Calculate the trajectory and fate of a slick on the water surface

    :param profile: list
        A data ensemble required by far-field modeling, including:
        1) interpolation_water_u, 2) interpolation_water_v, 3) interpolation_wind_u,
        4) interpolation_wind_v, 5) min_time_current, 6) max_time_current, 7) min_depth_current,
        8) max_depth_current, 9) min_lat_current,  10) max_lat_current, 11) min_lon_current,
        12) max_lon_current, 13) min_time_wind, 14) max_time_wind, 15) min_lat_wind,
        16) max_lat_wind, 17) min_lon_wind, 18) max_lon_wind, 19) time_interval,
        20) profile1d.get_values: P, Ta, Sa, Ua, Va, Wa.
    :param slick: SPM_utilities.Slick
        Object describing the properties of the oil slick.
    :param p: SPM.ModelParams
        Container for the fixed model parameters
    :param t0_sur: float
        The initial time for surface oil simulation (hour)
    :param y0_sur: ndarray
        The initial condition for surface oil simulation
    :param dt23: float
        End time of simulation (hours). Also relate to the initial time of the current data.
    :param dt_sur: float
        Time step of far-field model for surface oil (hours)
    :param Dxyz_sur: list
        A container includes diffusion coefficients for surface slick
    :param para_current: float, default=1
        Current drift coefficient
    :param para_wind: float, default=0.03
        Wind drift coefficient
    :param cond: string, default 'unsteady'
        If cond (condition) is 'unsteady', surface current field is set as unsteady flows;
        otherwise, surface current field is steady field.

    :return t_sur: ndarray
        Times (h) associated with the state space of the slick
    :return y_sur: ndarray
        The state space of the slick
        y[0:3]: x, y, z
        y[3]: A
        y[4:4+i]: each PC's mass in a slick (i in total)
        y[4+i]: slick water content
    :return slick: SPM_utilities.Slick
        Object describing the properties of the oil slick.

    """

    # Create an integrator object: use "vode" with "Backward differentiation formula" for stiff ODEs
    # Select derivatives based on the stability of inputted flow
    if cond == 'unsteady':
        r = integrate.ode(derivs_surface).set_integrator('vode', method='Adams', atol=1.e-3,
                                                         rtol=1e-3, order=4, max_step=dt_sur)
    else:
        r = integrate.ode(derivs_surface_steady).set_integrator('vode', method='Adams', atol=1.e-3,
                                                                rtol=1e-3, order=4, max_step=dt_sur)

    # Initialize the state space
    r.set_initial_value(y0_sur, t0_sur)

    # Set passing variables for derivs method
    r.set_f_params(profile, slick, p, para_current, para_wind, dt_sur, Dxyz_sur)

    # Create lists to store the solution and its associate time
    t = [t0_sur]
    y = [y0_sur]

    k = 0
    f = 1
    psteps = 100.
    stop = False

    if shore_polygon is None:
        while r.successful() and not stop:
            f = sum(slick.m) / sum(slick.m0)
            # Print progress to the screen
            if np.remainder(k, psteps) == 0.:
                print('    Lat: %g, Lon: %g, t:  %g (hrs), k: %d, f: %g (--)' % (r.y[1], r.y[0], t[-1], k, f))

            # Perform one step of the integration
            r.integrate(t[-1] + dt_sur, step=True)

            # Correct the slick area and mass
            correct_area_mass(r, slick)

            t.append(r.t)
            y.append(r.y)
            k += 1

            # Update the properties of the slick, i.e., slick mass, x, y, slick area, water content, p, time step
            slick.update(r.y[4:-1], r.y[0], r.y[1], r.y[3], r.y[-1], p, dt_sur)

            # Record the current iteration times of solver
            slick.get_k(iter_solver=k, iter_weather=None)

            # Evaluate stop criteria
            if r.successful():
                # Check if a slick pass through the data boundary
                if (r.y[0] <= p.left_bound or r.y[0] >= p.right_bound or
                      r.y[1] <= p.lower_bound or r.y[1] >= p.upper_bound):
                    stop = True
                # Check if slick weathered out (based on fdis)
                elif f < slick.particle.fdis:
                    stop = True
                # Check if simulation exceed the user-defined end time
                elif t[-1] > dt23:
                    stop = True
    else:
        while r.successful() and not stop:
            f = sum(slick.m) / sum(slick.m0)
            # Print progress to the screen
            if np.remainder(k, psteps) == 0.:
                print('    Lat: %g, Lon: %g, t:  %g (hrs), k: %d, f: %g (--)' % (r.y[1], r.y[0], t[-1], k, f))

            # Perform one step of the integration
            r.integrate(t[-1] + dt_sur, step=True)

            # Correct the slick area and mass
            correct_area_mass(r, slick)

            t.append(r.t)
            y.append(r.y)
            k += 1

            # Update the properties of the slick, i.e., slick mass, x, y, slick area, water content, p, time step
            slick.update(r.y[4:-1], r.y[0], r.y[1], r.y[3], r.y[-1], p, dt_sur)

            # Record the current iteration times of solver
            slick.get_k(iter_solver=k, iter_weather=None)

            # Evaluate stop criteria
            if r.successful():
                # Check if a slick pass through the data boundary
                if Point(r.y[0], r.y[1]).within(shore_polygon):
                    slick.strand = True
                    stop = True
                elif (r.y[0] <= p.left_bound or r.y[0] >= p.right_bound or
                      r.y[1] <= p.lower_bound or r.y[1] >= p.upper_bound):
                    stop = True
                # Check if slick weathered out (based on fdis)
                elif f < slick.particle.fdis:
                    stop = True
                # Check if simulation exceed the user-defined end time
                elif t[-1] > dt23:
                    stop = True

    t_sur = np.array(t)
    y_sur = np.array(y)
    y_sur[:, 2] = 0

    print('    Lat: %g, Lon: %g, t:  %g (hrs), k: %d, f: %g (--)'
          % (y_sur[-1, 1], y_sur[-1, 0], t_sur[-1], k, f))

    return t_sur, y_sur, slick


def derivs_surface(t_sur, y_sur, profile, slick, p, para_current, para_wind, dt_sur, Dxyz_sur):
    """
    Assemble the right hand side of the ODE for the trajectory and fate of
    a slick at the water surface.
    This function is used for an unsteady current field.

    :param t_sur: float
        Current value for the independent variable (time in hours).
    :param y_sur: ndarray
        Current value for the state space vector of a slick.
    :param profile: list
        A data ensemble required by far-field modeling, including:
        1) interpolation_water_u, 2) interpolation_water_v, 3) interpolation_wind_u,
        4) interpolation_wind_v, 5) min_time_current, 6) max_time_current, 7) min_depth_current,
        8) max_depth_current, 9) min_lat_current,  10) max_lat_current, 11) min_lon_current,
        12) max_lon_current, 13) min_time_wind, 14) max_time_wind, 15) min_lat_wind,
        16) max_lat_wind, 17) min_lon_wind, 18) max_lon_wind, 19) time_interval,
        20) profile1d.get_values: P, Ta, Sa, Ua, Va, Wa.
    :param slick: SPM_utilities.Slick
        Object describing the properties of the oil slick.
    :param p: SPM.ModelParams
        Container for the fixed model parameters
    :param para_current: float, default=1
        Current drift coefficient
    :param para_wind: float, default=0.03
        Wind drift coefficient
    :param dt_sur: float
        Time step of far-field model for surface oil (hours)

    :return yp_sur: ndarray
        A set of derivatives on the right hand side of the ODE for the
        trajectory and fate of a slick at the sea surface.

        The state space of the slick:
        yp[0:3]: derivatives for advection
        yp[3]: derivatives for spreading
        yp[4:4+22]: derivatives for each PC's weathering
        yp[4+22]: derivatives for emulsification

    """

    # Set up a container for derivatives
    yp_sur = np.zeros(y_sur.shape)

    # Get the right time corresponding to the current and wind data
    t_current = t_sur + p.dt02_current
    t_wind = t_sur + p.dt02_wind

    # Get RegularGridInterpolator for current velocities in x- and y- directions
    u_water, v_water = current_velocity(profile[0], profile[1], y_sur, t_current)

    # Get RegularGridInterpolator for wind velocities in x- and y- directions
    u_wind, v_wind = wind_velocity(profile[2], profile[3], y_sur, t_wind)

    # Compute wind velocity
    wind_speed = np.sqrt(u_wind ** 2 + v_wind ** 2)

    # Set derivatives for surface advection
    yp_sur[0], yp_sur[1], yp_sur[2] = transport_surface(para_current, para_wind,
                                                        u_water, v_water,
                                                        u_wind, v_wind, p, y_sur, dt_sur, Dxyz_sur)

    # Set derivatives for spreading
    yp_sur[3] = spreading_rate(slick, p)

    # Set derivatives for evaporation
    evap = evaporation_rate(slick, p, wind_speed)

    # Set derivatives for dispersion
    # disp = dispersion_rate_GNOME(slick, p, wind_speed)
    disp = dispersion_rate_Johansen(slick, p, wind_speed)
    # disp = dispersion_rate_Fingas(slick, p, wind_speed)

    idx = 4
    num_components = len(slick.m)

    yp_sur[idx:idx + num_components] = evap + disp

    # Set derivatives for emulsification
    yp_sur[idx + num_components] = emulsification_rate(slick, p, wind_speed)

    # Record the weathered oil mass
    slick.update_fate(evap, disp, dt_sur, t_sur)

    # Record the iteration times for oil weathering
    slick.get_k(iter_solver=False, iter_weather=True)

    # Return the derivatives
    return yp_sur


def derivs_surface_steady(t_sur, y_sur, profile, slick, p, para_current, para_wind,
                          dt_sur, Dxyz_sur, u_wind=0, v_wind=0):
    """
    Assemble the right hand side of the ODE for the trajectory and fate of
    a slick at the water surface.
    This function is used for a steady current field.

    :param t_sur: float
        Current value for the independent variable (time in hours).
    :param y_sur: ndarray
        Current value for the state space vector of a slick.
    :param profile: list
        A data ensemble required by far-field modeling, including:
        1) interpolation_water_u, 2) interpolation_water_v, 3) interpolation_wind_u,
        4) interpolation_wind_v, 5) min_time_current, 6) max_time_current, 7) min_depth_current,
        8) max_depth_current, 9) min_lat_current,  10) max_lat_current, 11) min_lon_current,
        12) max_lon_current, 13) min_time_wind, 14) max_time_wind, 15) min_lat_wind,
        16) max_lat_wind, 17) min_lon_wind, 18) max_lon_wind, 19) time_interval,
        20) profile1d.get_values: P, Ta, Sa, Ua, Va, Wa.
    :param slick: SPM_utilities.Slick
        Object describing the properties of the oil slick.
    :param p: SPM.ModelParams
        Container for the fixed model parameters
    :param para_current: float, default=1
        Current drift coefficient
    :param para_wind: float, default=0.03
        Wind drift coefficient
    :param dt_sur: float
        Time step of far-field model for surface oil (hours)
    :param u_wind: float
        Wind speed in x-direction (m/s)
    :param v_wind: float
        Wind speed in y-direction (m/s)

    :return yp_sur: ndarray
        A set of derivatives on the right hand side of the ODE for the
        trajectory and fate of a slick at the sea surface.

        The state space of the slick:
        yp[0:3]: derivatives for advection
        yp[3]: derivatives for spreading
        yp[4:4+22]: derivatives for each PC's weathering
        yp[4+22]: derivatives for emulsification

    """

    # Set up a container for derivatives
    yp_sur = np.zeros(y_sur.shape)

    # Compute wind velocity
    wind_speed = np.sqrt(u_wind ** 2 + v_wind ** 2)

    # Use the current velocity of the shallowest layer in the near-field data
    u_water, v_water = p.Ua_sur, p.Va_sur

    # Set derivatives for surface advection
    yp_sur[0], yp_sur[1], yp_sur[2] = transport_surface(para_current, para_wind, u_water, v_water,
                                            u_wind, v_wind, p, y_sur, dt_sur, Dxyz_sur)

    # Set the derivative for spreading
    yp_sur[3] = spreading_rate(slick, p)

    # Set derivatives for evaporation
    evap = evaporation_rate(slick, p, wind_speed)

    # Set derivatives for dispersion
    disp = dispersion_rate_GNOME(slick, p, wind_speed)

    idx = 4
    num_components = len(slick.m)
    yp_sur[idx:idx + num_components] = evap + disp

    # Set derivatives for emulsification
    yp_sur[idx + num_components] = emulsification_rate(slick, p, wind_speed)

    # Record the weathered oil mass
    slick.update_fate(evap, disp, dt_sur, t_sur)

    # Record the iteration times for oil weathering
    slick.get_k(iter_solver=False, iter_weather=True)

    # Return the derivatives
    return yp_sur
