"""
Plume Model Integration Functions
=================================

ODE integration, entrainment, and derivative functions for the
Lagrangian plume model.

"""

from __future__ import annotations
from copy import deepcopy

import numpy as np
from scipy import integrate
from scipy.optimize import fsolve

import seawater


def calculate(t0, q0, q0_local, profile, p, particles, derivs, dt_max, sd_max):
    """
    Integrate the Lagrangian plume solution

    Parameters
    ----------
    t0 : float
        Initial time (s)
    q0 : ndarray
        Initial values of the state space vector
    q0_local : `bent_plume_model.LagElement`
        Object containing the numerical solution at the initial condition
    profile : `ambient.Profile` object
        The ambient CTD object used by the simulation.
    p : `ModelParams` object
        Object containing the fixed model parameters for the bent
        plume model.
    particles : list of `Particle` objects
        List of `bent_plume_model.Particle` objects containing the dispersed
        phase local conditions and behavior.
    derivs : function handle
        Pointer to the function where the derivatives of the ODE system are
        stored.  Should be `lmp.derivs`.
    dt_max : float
        Maximum step size to use in the simulation (s).  The ODE solver
        in `calculate` is set up with adaptive step size integration, so
        this value determines the largest step size in the output data, but
        not the numerical stability of the calculation.
    sd_max : float
        Maximum number of nozzle diameters to compute along the plume
        centerline (s/D)_max.  This is the only stop criteria that is user-
        selectable.

    Returns
    -------
    t : ndarray
        Vector of times when the plume solution is obtained (s).
    y : ndarray
        Matrix of the plume state space solutions.  Each row corresponds to
        a time in `t`.

    """
    # Create an integrator object
    r = integrate.ode(derivs).set_integrator('vode', method='bdf', atol=1e-6,
                                             rtol=1e-3, order=5, max_step=dt_max)

    # Push the initial state space to the integrator object
    r.set_initial_value(q0, t0)

    # Make a copy of the q1_local object needed to evaluate the entrainment
    q1_local = deepcopy(q0_local)
    q0_hold = deepcopy(q1_local)

    # Create vectors (using the list data type) to store the solution
    t = [t0]
    q = [q0]

    # Integrate a finite number of time steps
    k = 0
    psteps = 30.
    stop = False
    neutral_counter = 0
    top_counter = 0
    while r.successful() and not stop:

        # Print progress to the screen
        if np.remainder(float(k), psteps) == 0.:
            print('    Distance:  %g (m), time:  %g (s), k:  %d' % \
                  (q[-1][10], t[-1], k))

        # Perform one step of the integration
        r.set_f_params(q0_local, q1_local, profile, p, particles)
        r.integrate(t[-1] + dt_max, step=True)
        q1_local.update(r.t, r.y, profile, particles)

        # Correct the temperature
        r = correct_temperature(r, particles)

        # Remove particle solution for particles outside the plume
        r = correct_particle_tracking(r, particles)

        # Store the results
        t.append(r.t)
        q.append(r.y)

        # Update the Lagrangian elements for the next time step
        q0_local = q0_hold
        q0_hold = deepcopy(q1_local)

        # Check if the plume has reached a maximum rise height yet
        if np.sign(q0_local.Jz) != np.sign(q1_local.Jz):
            top_counter += 1

        # Check if the plume is at neutral buoyancy in an intrusion layer
        # (e.g., after the top of the plume)
        if top_counter > 0:
            if np.sign(q0_local.rho_a - q0_local.rho) != np.sign(q1_local.rho_a - q1_local.rho):
                # Update neutral buoyancy level counter
                neutral_counter += 1

        # Evaluate the stop criteria
        if neutral_counter >= 1:
            # Passed through the second neutral buoyany level
            print('Passed through the second neutral buoyany level')
            stop = True
        if q[-1][10] / q1_local.diameter > sd_max:
            # Progressed desired distance along the plume centerline
            print('Progressed desired distance along the plume centerline')
            stop = True
        if k >= 50000:
            # Stop after specified number of iterations; used to protect against problems with the solution become stuck
            print('Stop after specified number of iterations')
            stop = True
        if q[-1][9] <= 0.:
            # Reached a location at or above the free surface
            print('Reached a location at or above the free surface')
            stop = True
        if q[-1][10] == q[-2][10]:
            # Progress of motion of the plume has stopped
            print('Progress of motion of the plume has stopped')
            stop = True

        # Update the index counter
        k += 1

    # Convert solution to numpy arrays
    t = np.array(t)
    q = np.array(q)

    # Show user the final calculated point and return the solution
    print('    Distance:  %g (m), time:  %g (s), k:  %d' % (q[-1, 10], t[-1], k))
    return t, q


def correct_temperature(r, particles):
    """
    Parameters
    ----------
    r : `scipy.integrate.ode` object
        ODE solution containing the current values of the state space in
        the solver's extrinsic data.  These values are editable, but an
        intrinsic version of these data are used when the solver makes
        calculations; hence, editing this file does not change the state
        space stored in the actual solver.
    particles : list of `Particle` objects
        List of `bent_plume_model.Particle` objects containing the dispersed
        phase local conditions and behavior.

    Returns
    -------
    r : `sciply.integrate.ode` object
        The updated extrinsic state space with the correct values for heat
        as were used in the calcualtion.

    """
    # Find the heat conservation equation in the model state space for the
    # particles and replace r.y with the correct values.
    idx = 11
    for i in range(len(particles)):
        idx += particles[i].FluidParticle.nc
        r.y[idx] = np.sum(particles[i].m) * particles[i].nbe * particles[i].cp * particles[i].T
        # Advance for heat, time, and position
        idx += 1 + 1 + 3

    # Return the corrected solution
    return r


def correct_particle_tracking(r, particles):
    """
    Remove the particle tracking solution after particles exit plume

    Parameters
    ----------
    r : `scipy.integrate.ode` object
        ODE solution containing the current values of the state space in
        the solver's extrinsic data.  These values are editable, but an
        intrinsic version of these data are used when the solver makes
        calculations; hence, editing this file does not change the state
        space stored in the actual solver.
    particles : list of `Particle` objects
        List of `bent_plume_model.Particle` objects containing the dispersed
        phase local conditions and behavior.

    Returns
    -------
    r : `sciply.integrate.ode` object
        The updated extrinsic state space with the correct values for heat
        as were used in the calcualtion.

    """
    # Skip through the single-phase state space
    idx = 11

    # Check each particle to determine whether they are inside or outside
    # the plume
    for i in range(len(particles)):
        if not particles[i].integrate:
            # Skip the masses, temperature, and time
            idx += particles[i].FluidParticle.nc + 2

            # Particle is outside the plume; replace the coordinates with
            # np.nan
            r.y[idx:idx + 3] = np.nan
            idx += 3

        else:
            # Skip the masses, temperature, time, and coordinates
            idx += particles[i].FluidParticle.nc + 5

        # Check if the integration should stop
        if particles[i].p_fac == 0.:
            # Stop tracking the particle inside the plume
            particles[i].integrate = False

            # Store the properties at the exit point
            particles[i].te = particles[i].t
            particles[i].xe = particles[i].x
            particles[i].ye = particles[i].y
            particles[i].ze = particles[i].z
            particles[i].me = particles[i].m
            particles[i].Te = particles[i].T

    # Return the corrected solution
    return r


def entrainment(q0_local, q1_local, p):
    """
    Compute the total shear and forced entrainment at one time step

    Computes the local entrainment (kg/s) as a combination of shear
    entrainment and forced entrainment for a local Lagrangian element. This
    function follows the approach in Lee and Cheung (1990) to compute both
    types of entrainment, but uses the formulation in Jirka (2004) for the
    shear entrainment term. Like Lee and Cheung (1990), it uses the maximum
    entrainment hypothesis: entrainment = max (shear, forced), with the
    exception that a pure coflowing momentum jet has entrainment = shear +
    forced. This function also makes one correction that in pure coflow
    the forced entrainment should be computed by integrating around the entire
    jet, and not just the half of the jet exposed to the current.

    Parameters
    ----------
    q0_local : `bent_plume_model.LagElement`
        Object containing the numerical solution at the previous time step
    q1_local : `bent_plume_model.LagElement`
        Object containing the numerical solution at the current time step
    p : `ModelParams` object
        Object containing the fixed model parameters for the bent
        plume model.

    Returns
    -------
    md : float
        Total entrainment (kg/s)

    Notes
    -----
    The entrainment computed here is already integrated over the current
    Lagrangian element surface area.  Hence, the result is (kg/s) into the
    element.

    """
    # Find the magnitude and direction of the velocity vector in q1_local
    Ua = np.sqrt(q1_local.ua ** 2 + q1_local.va ** 2 + q1_local.wa ** 2)
    Phi_a = np.arctan2(q1_local.wa, np.sqrt(q1_local.ua ** 2 + q1_local.va ** 2))
    Theta_a = np.arctan2(q1_local.va, q1_local.ua)

    # Get the component of the ambient current along the plume centerline
    Us = Ua * np.cos(q1_local.phi - Phi_a) * np.cos(q1_local.theta - Theta_a)

    # Get the sines and cosines of the new angles
    sin_t = np.sin(q1_local.theta - Theta_a)
    sin_p = np.sin(q1_local.phi - Phi_a)
    cos_t = np.cos(q1_local.theta - Theta_a)
    cos_p = np.cos(q1_local.phi - Phi_a)
    cos_t0 = np.cos(q0_local.theta - Theta_a)
    cos_p0 = np.cos(q0_local.phi - Phi_a)

    # Get the shear entrainment coefficient for the top-hat model.  In this
    # equation, phi has to be with reference to the gravity vector; hence,
    # we pass phi for the fixed coordinate system, but theta has to be the
    # angle from the crossflow direction, so be pass theta - theta_a.
    alpha_s = shear_entrainment(q1_local.V, Us,
                                q1_local.rho, q1_local.rho_a, q1_local.b, q1_local.sin_p, p)

    # Total shear entrainment (kg/s)
    md_s = q1_local.rho_a * np.abs(q1_local.V - Us) * alpha_s * (2.
                                                                 * np.pi * q1_local.b * q1_local.h)

    # Compute the projected area entrainment terms...first, the crossflow
    # projected onto the plume centerline
    a1 = 2. * q1_local.b * np.sqrt(sin_p ** 2 + sin_t ** 2 - sin_p ** 2 *
                                   sin_t ** 2) * q1_local.h
    if (q1_local.s - q0_local.s) / q1_local.b <= 1.e-3:
        # The plume is not progressing along the centerline; assume the
        # expansion and curvature corrections are small since delta s / b is
        # very small.
        a2 = 0.
        a3 = 0.
    else:
        # Second, correction for plume expansion
        a2 = np.pi * q1_local.b * (q1_local.b - q0_local.b) / (q1_local.s -
                                                               q0_local.s) * q1_local.h * cos_p * cos_t
        # Third, correction for plume curvature
        a3 = np.pi * q1_local.b ** 2 / 2. * (cos_p * cos_t - cos_p0 *
                                             cos_t0) / (q1_local.s - q0_local.s) * q1_local.h

    # Get the total projected area for the forced entraiment
    if np.abs(sin_t) <= 1.e-9 and np.abs(sin_p) <= 1.e-9:
        # Jet is in co-flow, shear entrainment model takes care of this case
        # by itself
        A = 0.
    else:
        A = a1 + a2 + a3

    # Total forced entrainment (kg/s)
    md_f = q1_local.rho_a * Ua * A

    # Obtain the total entrainment using the maximum hypothesis from Lee and
    # Cheung (1990)
    if md_s > md_f:
        md = md_s
    else:
        md = md_f

    # Return the entrainment rate
    return md


def track_particles(q0_local, q1_local, md, particles):
    """
    Compute the forcing variables needed to track particles

    The independent variable in the Lagrangian plume model is t for advection
    of the continuous phase.  Because the particles slip through the fluid,
    their advection speed is different; hence, all particle equations have
    a different independent variable for their time, tp.  Also, particle
    motion is computed in the local plume coordinates space (l,n,m); thus,
    the vertical slip velocity needs to be transformed to the local plume
    coordinate system.  Finally, the entrainment velocity pointing toward
    the plume centerline needs to be evaluated, based on the distance the
    particle is from the plume centerline.

    Parameters
    ----------
    q0_local : `bent_plume_model.LagElement`
        Object containing the numerical solution at the previous time step
    q1_local : `bent_plume_model.LagElement`
        Object containing the numerical solution at the current time step
    md : float
        Total entrainment into the Lagrangian element (kg)
    particles : list of `Particle` objects
        List of `bent_plume_model.Particle` objects containing the dispersed
        phase local conditions and behavior.

    Returns
    -------
    fe : float
        Entrainment frequency (1/s)
    up : ndarray
        Slip velocity for each particle projected on the local plume
        coordinate system (l,n,m) (m/s).  Each row is for a different
        particle and the columns are for the velocity in the (l,n,m)
        direction.
    dtp_dt : ndarray
        Differential particle transport time to continuous phase transport
        time evaluated from the previous time step to the current time
        step (--).

    """
    # Compute the entrainment frequency
    fe = md / (2. * np.pi * q1_local.rho_a * q1_local.b ** 2 *
               q1_local.h)

    # Get the rotation matrix to the local coordinate system (l,n,m)
    ds = q1_local.s - q0_local.s
    A = local_coords(q1_local)

    # Get the velocity of the current plume slice
    V = q1_local.V

    # Compute particle properties
    up = np.zeros((len(particles), 3))
    dtp_dt = np.zeros(len(particles))

    for i in range(len(particles)):
        # Transform the slip velocity from Cartesian coordinate to the
        # local plume coordinate system (l,n,m)
        up[i, :] = np.dot(A, np.array([0., 0., -particles[i].us]))

        # Get the distance along the particle path
        dsp = np.sqrt((q1_local.x_p[i, 0] - q0_local.x_p[i, 0]) ** 2 +
                      (q1_local.x_p[i, 1] - q0_local.x_p[i, 1]) ** 2 +
                      (q1_local.x_p[i, 2] - q0_local.x_p[i, 2]) ** 2)

        # Get the total velocity of the particle
        Vp = np.sqrt((up[i, 0] + V) ** 2 +
                     (up[i, 1] - fe * q1_local.X_p[i, 1]) ** 2 +
                     (up[i, 2] - fe * q1_local.X_p[i, 2]) ** 2)

        # Compute the particle time correction dtp/dt
        if Vp == 0:
            dtp_dt[i] = 0.
        elif ds == 0:
            dtp_dt[i] = 1.
        else:
            dtp_dt[i] = V / Vp * dsp / ds

    # Return the particle tracking variables
    return fe, up, dtp_dt


def local_coords(q1_local):
    """
    Compute the rotation matrix from (x, y, z) to (l, n, m)

    Computes the rotation matrix from the local Cartesian coordinate system
    (x - xi, y - yi, z - zi), where (xi, yi, zi) is the current location of
    the Lagrangian plume element, to the system tangent to the current plume
    trajectory (l, n, m); l is oriented tangent to the plume centerline,
    n is orthogonal to l and along the line from the local radius of
    curvature, and m is orthogonal to n and l.  The transformation is
    provided in Lee and Chueng (1990).  This method makes a small adaptation
    allowing for infinite radius of curvature (plume propagating along a straight line).

    Parameters
    ----------
    q1_local : `bent_plume_model.LagElement`
        Object containing the numerical solution at the current time step

    Returns
    -------
    A : ndarray
        Rotation matrix from (x, y, z)' to (l, n, m).  The inverse of this
        matrix will convert back from (l, n, m) to (x, y, z)'.

    See Also
    --------
    bent_plume_model.Particle.track

    """
    # l is along the s-axis, n is normal to l and m is normal to n and l
    A = np.array([
        [q1_local.cos_p * q1_local.cos_t, q1_local.cos_p * q1_local.sin_t, q1_local.sin_p],
        [q1_local.cos_t * q1_local.sin_p, q1_local.sin_t * q1_local.sin_p, -q1_local.cos_p],
        [q1_local.sin_t, -q1_local.cos_t, 0.]])

    # Return the rotation matrix
    return A


# ----------------------------------------------------------------------------
# Functions to compute the initial conditions for the first model element
# ----------------------------------------------------------------------------

def ic_plume_model(profile, particles, X, D, phi_0, theta_0, Sj, Tj, p):
    """
    Compute the initial conditions for the Lagrangian plume state space

    Compute the initial conditions at the release location for a Lagrangian
    plume element.  This can either be a pure single-phase plume, a pure
    multiphase plume, or a mixed release of multiphase and continuous phase
    fluid.

    Parameters
    ----------
    profile : `ambient.Profile` object
        The ambient CTD object used by the single bubble model simulation.
    particles : list of `Particle` objects
        List of `bent_plume_model.Particle` objects containing the dispersed
        phase local conditions and behavior.
    X : ndarray
        Release location (x, y, z) in (m)
    D : float
        Diameter for the equivalent circular cross-section of the release
        (m)
    phi_0 : float
        Vertical angle from the horizontal for the discharge orientation
        (rad in range +/- pi/2)
    theta_0 : float
        Horizontal angle from the x-axis for the discharge orientation.
        The x-axis is taken in the direction of the ambient current.
        (rad in range 0 to 2 pi)
    Sj : float
        Salinity of the continuous phase fluid in the discharge (psu)
    Tj : float
        Temperature of the continuous phase fluid in the discharge (T)

    Returns
    -------
    t : float
        Initial time for the simulation (s)
    q : ndarray
        Initial value of the plume state space

    """
    # Get the initial volume flux
    # This is a pure multiphase plume. Estimate the initial conditions using Wuest et al. 1992.
    Q, A, X, Tj, Sj, Pj, rho_j = zfe_volume_flux(profile, particles, p, X, D / 2)

    # Find the effective diameter of the source area
    D = np.sqrt(4. * A / np.pi)

    # Build the initial state space with these initial values
    # Set the dimensions of the initial Lagrangian plume element.
    b = D / 2.
    h = D / 5.

    # Measure the arc length along the plume
    s0 = 0.

    # Determine the time to fill the initial Lagrangian element
    dt = np.pi * b ** 2 * h / Q

    # Compute the mass of jet discharge in the initial Lagrangian element
    Mj = Q * dt * rho_j

    nbe = np.zeros(len(particles))
    for i in range(len(particles)):
        nbe[i] = particles[i].nb0 * dt
        particles[i].nbe = nbe[i]

    # Get the velocity in the component directions
    Uj = flux_to_velocity(Q, A, phi_0, theta_0)

    # Compute the magnitude of the exit velocity
    V = np.sqrt(Uj[0] ** 2 + Uj[1] ** 2 + Uj[2] ** 2)

    # Build the continuous-phase portion of the model state space vector
    t = 0.
    q = [Mj, Mj * Sj, Mj * seawater.cp() * Tj, Mj * Uj[0], Mj * Uj[1],
         Mj * Uj[2], h / V, X[0], X[1], X[2], s0]

    # Add in the state space for the dispersed phase particles
    q.extend(particles_state_space(particles, nbe))

    q = np.array(q)

    chem_names = get_chem_names(particles)

    # Return the initial state space
    return t, q, chem_names


def zfe_volume_flux(profile, particles, p, X0, R):
    """
    Initial volume for a multiphase plume

    Uses the Wueest et al. (1992) plume Froude number method to estimate
    the amount of entrainment at the source of a dispersed phase plume with
    zero continuous phase flux (e.g., a pure bubble, droplet, or particle
    plume)

    Parameters
    ----------
    profile : `ambient.Profile` object
        The ambient CTD object used by the single bubble model simulation.
    particles : list of `Particle` objects
        List of `SingleParticle`, `PlumeParticle`, or
        `bent_plume_model.Particle` objects describing each dispersed phase
        in the simulation
    p : `stratified_plume_model.ModelParams` or `bent_plume_model.ModelParams`
        Object containing the fixed model parameters for one of the integral
        plume models
    X0 : float
        (x, y, depth) coordinates of the release point (m)
    R : float
        Radius of the equivalent circular area of the release (m)

    """
    # The initial condition is valid at the diffuser (e.g., no virtual point
    # source for the Wuest et al. 1992 initial conditions).  Send back
    # exactly what the user supplied
    X = X0

    # Get X0 as a three-dimensional vector for generality
    if not isinstance(X0, np.ndarray):
        if not isinstance(X0, list):
            X0 = np.array([0., 0., X0])
        else:
            X0 = np.array(X0)

    # Get the ambient conditions at the discharge
    P, Ta, Sa, Ua, Va, Wa = profile.get_values(X0[2])
    rho = seawater.density(Ta, Sa, P)

    # Update the particle objects and pull out the multiphase properties.
    # Since this is the release, the particle age is zero.
    lambda_1 = np.zeros(len(particles))
    us = np.zeros(len(particles))
    rho_p = np.zeros(len(particles))
    Q = np.zeros(len(particles))
    for i in range(len(particles)):
        particles[i].update(particles[i].m, particles[i].T, P, Sa, Ta, 0.)
        lambda_1[i] = particles[i].lambda_1
        us[i] = particles[i].us
        rho_p[i] = particles[i].rho_p
        Q[i] = np.sum(particles[i].m) * particles[i].nb0 / rho_p[i]

    # Compute the buoyancy flux weighted average of lambda_1
    lambda_ave = bf_average(particles, rho, p.g, p.rho_r, lambda_1)

    # Calculate the initial velocity of entrained ambient fluid
    u_0 = np.sum(Q) / (np.pi * (lambda_ave * R) ** 2)
    u = wuest_ic(u_0, particles, lambda_1, lambda_ave, us, rho_p, rho, Q, R, p.g, p.Fr_0)

    # Check the void fraction
    xi = void_fraction(u, particles, lambda_1, us, Q, R)
    print('\nUser-defined d0 = %g (m) gives a void fraction of %g' % (2. * R, np.sum(xi)))
    if np.sum(xi) >= 1.:
        # This release is too vigorous to allow a dilute multiphase plume at the
        # orifice.  Change the initial width and re-calculate the initial
        # conditions.
        dA = 3.
        while np.sum(xi) >= 1.:
            R = np.sqrt(dA * np.sum(xi)) * R
            u_0 = np.sum(Q) / (np.pi * (lambda_ave * R) ** 2)
            u = wuest_ic(u_0, particles, lambda_1, lambda_ave, us, rho_p, rho,
                         Q, R, p.g, p.Fr_0)
            xi = void_fraction(u, particles, lambda_1, us, Q, R)

    # The initial plume width is the discharge port width
    A = np.pi * R ** 2

    # Calculate the volume flux
    Q = A * u

    return Q, A, X, Ta, Sa, P, rho


def wuest_ic(u_0, particles, lambda_1, lambda_ave, us, rho_p, rho, Q, R, g, Fr_0):
    """
    Compute the initial velocity of entrained ambient fluid

    Computes the initial velocity of the entrained ambient fluid following
    the method in Wueest et al. (1992).  This method is implicit; thus, an
    initial guess for the velocity and a root-finding approach is required.

    Parameters
    ----------
    u_0 : float
        Initial guess for the entrained fluid velocity (m/s)
    particles : list of `Particle` objects
        List of `SingleParticle`, `PlumeParticle`, or
        `bent_plume_model.Particle` objects describing each dispersed phase
        in the simulation
    lambda_1 : ndarray
        Spreading rate of the each dispersed phase particle in a plume (--)
    lambda_ave : float
        Buoyancy flux averaged value of lambda_1 (--)
    us : ndarray
        Slip velocity of each of the dispersed phase particles (m/s)
    rho_p : ndarray
        Density of each of the dispersed phase particles (kg/m^3)
    rho : float
        Density of the local ambient continuous phase fluid (kg/m^3)
    Q : ndarray
        Total volume flux of particles for each dispersed phase (m^3/s)
    R : float
        Radius of the release port (m)
    g : float
        Acceleration of gravity (m/s^2)
    Fr_0 : float
        Desired initial plume Froude number (--)

    Returns
    -------
    u : float
        The converged value of the entrained fluid velocity in m/s at the
        release location in order to achieve the specified value of Fr_0.

    """

    # The Wuest et al. (1992) initial condition is implicit; define the
    # residual for use in a root-finding algorithm
    def residual(u):
        """
        Compute the residual of the Wueest et al. (1992) initial condition
        using the current guess for the initial velocity u.

        Parameters
        ----------
        u : float
            the current guess for the initial velocity (m/s)

        Notes
        -----
        All parameters of `wuest_ic` are global to this function since it is
        a subfunction of `wuest_ic`.

        """
        # Get the void fraction for the current estimate of the mixture of
        # dispersed phases and entrained ambient water
        xi = void_fraction(u, particles, lambda_1, us, Q, R)

        # Get the mixed-fluid plume density
        rho_m = np.sum(xi * rho_p) + (1. - np.sum(xi)) * rho

        # Calculate the deviation from the desired Froude number
        return Fr_0 - u / np.sqrt(2. * lambda_ave * R * g *
                                  (rho - rho_m) / rho_m)

    return fsolve(residual, u_0)[0]


def bf_average(particles, rho, g, rho_r, parm):
    """
    Compute a buoyancy-flux-weighted average of `parm`

    Computes a weighted average of the values in `parm` using the kinematic
    buoyancy flux of each particle containing parm as the weight in the
    average calculation.

    Parameters
    ----------
    particles : list of `Particle` objects
        List of `SingleParticle`, `PlumeParticle`, or
        `bent_plume_model.Particle` objects describing each dispersed phase
        in the simulation
    rho : float
        Local density of ambient fluid outside plume (kg/m^3).
    g : float
        Acceleration of gravity (m/s^2).
    rho_r : float
        Model reference density (kg/m^3).
    parm : ndarray
        Numpy array of parameters to average, one value for each
        dispersed phase entry (same as elements in parm).

    Returns
    -------
    parm_ave : float
        The weighted average of `parm`.

    """
    # Compute the total buoyancy flux of each dispersed phase particle in the
    # simulation
    F = np.zeros(len(particles))
    for i in range(len(particles)):
        # Get the total particle volume flux
        Q = np.sum(particles[i].m) * particles[i].nb0 / particles[i].rho_p
        # Compute the particle kinematic buoyancy flux
        F[i] = g * (rho - particles[i].rho_p) / rho_r * Q

    # Return the buoyancy-flux-weighted value of parm
    if np.sum(F) == 0.:
        parm = 0.
    else:
        parm = np.sum(F * parm) / np.sum(F)

    return parm


def void_fraction(u, particles, lambda_1, us, Q, R):
    """
    Compute the void fraction for the Wuest et al. (1992) initial condition

    Computes the total void fraction of a set of fluid particles for a given
    geometry and estimated volume flux of entrained fluid.

    Parameters
    ----------
    particles : list of `Particle` objects
        List of `SingleParticle`, `PlumeParticle`, or
        `bent_plume_model.Particle` objects describing each dispersed phase
        in the simulation
    lambda_1 : ndarray
        Spreading rate of the each dispersed phase particle in a plume (--)
    us : ndarray
        Slip velocity of each of the dispersed phase particles (m/s)
    Q : ndarray
        Total volume flux of particles for each dispersed phase (m^3/s)
    R : float
        Radius of the release port (m)

    Returns
    -------
    xi : nd.array
        Individual void fractions for each particle in the particle list (--)

    """
    # Initialize an empty array
    xi = np.zeros(len(particles))

    # Get the void fraction of each particle
    for i in range(len(particles)):
        xi[i] = Q[i] / (np.pi * lambda_1[i] ** 2 * R ** 2 * (us[i] +
                                                             2. * u / (1. + lambda_1[i] ** 2)))

    return xi


def flux_to_velocity(Q, A, phi, theta):
    """
    Convert fluid flow rate to three-component velocity

    Computes the three-component velocity (u, v, w) along the Cartesian
    directions (x, y, depth) from the flow rate, cross-sectional area, and the
    orientation (phi and theta).

    Parameters
    ----------
    Q : Volume flux of continuous phase fluid at the discharge (m^3/s)
    A : Cross-sectional area of the discharge (M^2)
    phi : float
        Vertical angle from the horizontal for the discharge orientation
        (rad in range +/- pi/2)
    theta : float
        Horizontal angle from the x-axis for the discharge orientation.
        The x-axis is taken in the direction of the ambient current.
        (rad in range 0 to 2 pi)

    Returns
    -------
    Uj : ndarray
        Vector of the three-component velocity of continous phase fluid in
        the jet (u, v, w) in the Cartesian direction (x, y, depth)

    """
    # Get the velocity along the jet centerline
    Vj = Q / A

    # Project jet velocity on the three component directions (i, j, k)
    Uj = np.zeros(3)
    Uj[0] = np.cos(phi) * np.cos(theta) * Vj
    Uj[1] = np.cos(phi) * np.sin(theta) * Vj
    Uj[2] = np.sin(phi) * Vj

    # Return the velocity vector
    return Uj


def shear_entrainment(U, Us, rho, rho_a, b, sin_p, p):
    """
    Compute the entrainment coefficient for shear entrainment

    Computes the entrainment coefficient for the shear entrainment for a top
    hat model.  This code can be used by both the bent plume model and the
    stratified plume model.  It is based on the concepts for shear entrainment
    in Lee and Cheung (1990) and adapted by the model in Jirka (2004).  The
    model works for pure jets, pure plumes, and buoyant jets.

    Parameters
    ----------
    U : float
        Top hat velocity of entrained plume water (m/s)
    Us : float
        Component of the ambient current projected along the plume
        centerline (m/s)
    rho : float
        Density of the entrained plume fluid (kg/m^3)
    rho_a : float
        Density of the ambient water at the current height (kg/m^3)
    sin_p : float
        Sine of the angle phi from the horizontal with down being positive (up
        is - pi/2)
        Cosine of the angle theta from the crossflow direction
    p : `bent_plume_model.ModelParams` or `stratified_plume_model.ModelParams`
        Object containing the present model parameters

    Returns
    -------
    alpha_s : float
        The shear entrainment coefficient (--)

    """
    # Gaussian model jet entrainment coefficient
    alpha_j = p.alpha_j

    # Gaussian model plume entrainment coefficient
    if rho_a == rho:
        # This is a pure jet
        alpha_p = 0.
    else:
        # This is a plume; compute the densimetric Gaussian Froude number
        F1 = 2. * np.abs(U - Us) / np.sqrt(p.g * np.abs(rho_a - rho) * (1. +
                                                                        1.2 ** 2) / 1.2 ** 2 / rho_a * b / np.sqrt(2.))

        # Follow Figure 13 in Jirka (2004)
        if np.abs(F1 ** 2 / sin_p) > p.alpha_Fr / 0.028:
            alpha_p = - np.sign(rho_a - rho) * p.alpha_Fr * sin_p / F1 ** 2
        else:
            alpha_p = - (0.083 - p.alpha_j) / (p.alpha_Fr / 0.028) * F1 ** 2 / \
                      sin_p * np.sign(rho_a - rho)

    # Compute the total shear entrainment coefficient for the top-hat model
    if (np.abs(U - Us) + U) == 0:
        alpha_s = np.sqrt(2.) * alpha_j
    else:
        alpha_s = np.sqrt(2.) * (alpha_j + alpha_p) * 2. * U / \
                  (np.abs(U - Us) + U)

    # Return the total shear entrainment coefficient
    return alpha_s


def particles_state_space(particles, nb):
    """
    Create the state space describing the dispersed phase properties

    Constructs a complete state space of masses and heat content for all of
    the particles in the `particles` list.

    Parameters
    ----------
    particles : list of `Particle` objects
        List of `SingleParticle`, `PlumeParticle`, or
        `bent_plume_model.Particle` objects describing each dispersed phase
        in the simulation
    nb : ndarray
        Array of particle numbers for forming the state space.  nb can be in
        number/T, which will give state space variables in mass flux (M/T) or
        in number, which will give state space variables in mass.

    Returns
    -------
    y : ndarray
        Array of state space variables for the `particles` objects.

    """
    # Get the state variables of each particle, one particle as a time
    y = []
    for i in range(len(particles)):
        # Masses of each element in the particle
        y.extend(particles[i].m * nb[i])

        # Add in the heat flux of the particle
        y.append(np.sum(particles[i].m) * nb[i] * particles[i].cp * particles[i].T)

        # Initialize the particle age to zero
        y.append(0)

        # Initialize the particle positions to the center of the plume
        y.extend([0, 0, 0])

    # Return the state space as a list
    return y


def get_chem_names(particles):
    """
    Create a list of chemical names for the dispersed phase particles

    Reads the composition attribute of each particle in a `particles` list
    and compiles a unique list of particle names.

    Parameters
    ----------
    particles : list of `Particle` objects
        List of `SingleParticle`, `PlumeParticle`, or
        `bent_plume_model.Particle` objects describing each dispersed phase
        in the simulation

    Returns
    -------
    chem_names : str list
        List of the chemical composition of particles undergoing dissolution
        in the `particles` list

    """
    # Initialize a list to store the names
    chem_names = []

    # Add the chemicals that are part of the particle composition
    for i in range(len(particles)):
        chem_names += [chem for chem in particles[i].composition if
                       chem not in chem_names]

    # Return the list of chemical names
    return chem_names


def width_projection(Sx, Sy, b):
    """
    Find the location of the plume width in x, y, z space

    Converts the width b and plume orientation phi and theta into an
    (x, y, z) location of the plume edge.  This function provides a two-
    dimensional result given the unit vector along the plume centerline
    (Sx, Sy) along two dimensions in the (x, y, z) space

    Parameters
    ----------
    Sx : float
        Unit vector projection of the plume trajectory on one of the
        coordinate axes in (x, y, z) space.
    Sy : float
        Unit vector projection of the plume trajectory on another of the
        coordinate axes in (x, y, z) space.
    b : float
        Local plume width

    Returns
    -------
    x1 : float
        Plume edge for Sx projection to left of plume translation direction
    y1 : float
        Plume edge for Sy projection to left of plume translation direction
    x2 : float
        Plume edge for Sx projection to right of plume translation direction
    y1 : float
        Plume edge for Sy projection to right of plume translation direction

    Notes
    -----
    The values of S in the (x, y, z) sytem would be::

        Sz = sin ( phi )
        Sx = cos ( phi ) * cos ( theta )
        Sy = cos ( phi ) * sin ( theta )

    Any two of these coordinates of the unit vector can be provided to this
    function as input.

    """
    # Get the angle to the s-direction in the x-y plane
    alpha = np.arctan2(Sy, Sx)

    # Get the coordinates of the plume edge to the right of the s-vector
    # moving with the plume
    x1 = b * np.cos(alpha - np.pi / 2.)
    y1 = b * np.sin(alpha - np.pi / 2.)

    # Get the coordinates of the plume edge to the left of the s-vector
    # moving with the plume
    x2 = b * np.cos(alpha + np.pi / 2.)
    y2 = b * np.sin(alpha + np.pi / 2.)

    return x1, y1, x2, y2