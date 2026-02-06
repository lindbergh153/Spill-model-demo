"""
Lagrangian Plume Model
======================

Lagrangian integral model for simulating buoyant plumes from
deep water oil/gas releases.

Classes
-------
Model_plume : Main simulation controller
ModelParams : Fixed model parameters

References
----------
Socolofsky, S.A., et al. (2011). Intercomparison of oil spill prediction
    models for accidental blowout scenarios. Marine Pollution Bulletin.
Jirka, G.H. (2004). Integral model for turbulent buoyant jets.

"""

from __future__ import annotations

import matplotlib.pyplot as plt
from scipy.linalg import norm

from plume_model_functions import *
from plume_model_utilities import *


class Model_plume(object):
    """
    Object for controlling and post-processing the near-field model (NFM)

    """
    def __init__(self, profile):
        """
        :param profile: `ambient.Profile` object
            An object containing the ambient CTD data and associated methods.

        Attributes
        ----------
        profile : `vertical_profile.Profile1d`
            This object to read ambient CTD data
        p : `ModelParams`
            Container for the fixed model parameter
        sim_stored : bool
            Flag indicating whether a simulation result is stored in the object memory

        """
        self.profile = profile

        # Set the model parameters that the user cannot adjust
        self.p = ModelParams(self.profile)

        # Indicate that the simulation has not yet been conducted
        self.sim_stored = False

    def simulate(self, X, D, phi_0, theta_0, Sj, Tj, particles, dt_max, sd_max):
        """
        Simulate the plume dynamics from given initial conditions

        Simulate the buoyant plume using a Lagrangian plume integral model approach until
        1) the plume reaches the surface,
        or 2) the integration exceeds the given s/D (`sd_max`),
        or 3) the intrusion reaches a point of neutral buoyancy.

        :param X: ndarray
            Release location (x, y, z) in (m)
        :param D: float
            Diameter for the equivalent circular cross-section of the release (m)
        :param phi_0: float
            Vertical angle from the horizontal for the discharge orientation
        :param theta_0: float
            Horizontal angle from the x-axis for the discharge orientation.
        :param Sj: float
            Salinity of the continuous phase fluid in the discharge (psu)
        :param Tj: float
            Temperature of the continuous phase fluid in the discharge (K)
        :param particles: list of `Particle` objects
            List of `PlumeParticle` objects describing each dispersed phase in the simulation
        :param dt_max: float
            Maximum step size to take in the storage of the simulation solution (s)
        :param sd_max: float
            Maximum number of orifice diameters to compute the solution along
            the plume centerline (m/m)

        """

        # Store the input parameters as attributes
        self.X = X
        self.D = D
        self.phi_0 = phi_0
        self.theta_0 = theta_0
        self.Sj = Sj
        self.Tj = Tj
        self.particles = particles
        self.dt_max = dt_max
        self.sd_max = sd_max

        if len(self.particles) > 0:
            self.composition = self.particles[0].composition
        else:
            self.composition = []

        # Create the initial state space from the given input variables
        # simulation time, solutions, chemical composition
        t0, q0, self.chem_names = ic_plume_model(self.profile, self.particles,
                                                 self.X, self.D, self.phi_0,
                                                 self.theta_0, self.Sj, self.Tj, self.p)
        # Store the initial conditions in a Lagrangian element object
        self.q_local = LagElement(t0, q0, D, self.profile, self.particles, self.chem_names)

        # Compute the solution of the buoyant plume
        print('\n--- DWOSM Near-field model ---')
        self.t, self.q, = calculate(t0, q0, self.q_local, self.profile, self.p,
                                    self.particles, derivs, self.dt_max, self.sd_max)

        # Update the status of the solution
        self.sim_stored = True

        # Update the status of the particles
        for particle in self.particles:
            particle.sim_stored = True

        # Get the particles exiting from the plume
        self.particles_outside = self.get_exit_particles()

        print("\n--- {0} in {1} particles exit from the plume ---".
              format(len(self.particles_outside), len(particles)))

    def plot_2d(self):
        """
        Plot the plume and particles from three views,
        as well as plume element mass along with plume centerline

        """
        if self.sim_stored is False:
            print('No simulation results available to plot...')
            print('Plotting nothing.\n')

        print('Plotting the state space...')
        plot_2d(self.t, self.q, self.q_local, self.profile, self.p, self.particles)
        print('Done.\n')

    def plot_3d(self):
        """
        Plot the plume and particles in 3D views

        """
        if self.sim_stored is False:
            print('No simulation results available to plot...')
            print('Plotting nothing.\n')

        print('Plotting the state space...')
        plot_3d(self.t, self.q, self.q_local, self.profile, self.p, self.particles)
        print('Done.\n')

    def get_exit_particles(self):
        particles_outside = []
        if self.sim_stored:
            for particle in self.particles:
                if not particle.integrate:
                    particles_outside.append(particle)
        else:
            print('There is no simulation result')

        return particles_outside


def plot_2d(t, q, q_local, profile, p, particles):
    q0_local = LagElement(t[0], q[0, :], q_local.diameter, profile, particles, q_local.chem_names)
    n_part = q0_local.np
    pchems = 1
    for i in range(n_part):
        if len(particles[i].composition) > pchems:
            pchems = len(particles[i].composition)

    # Store the derived variables
    M0 = np.zeros(t.shape)
    S = np.zeros(t.shape)
    T = np.zeros(t.shape)
    Mpf = np.zeros((len(t), n_part, pchems))
    Hp = np.zeros((len(t), n_part))
    Mp = np.zeros((len(t), n_part))
    Tp = np.zeros((len(t), n_part))
    xp = np.zeros((len(t), 3 * n_part))
    u = np.zeros(t.shape)
    v = np.zeros(t.shape)
    w = np.zeros(t.shape)
    V = np.zeros(t.shape)
    h = np.zeros(t.shape)
    x0 = np.zeros(t.shape)
    y0 = np.zeros(t.shape)
    z0 = np.zeros(t.shape)
    s0 = np.zeros(t.shape)
    rho = np.zeros(t.shape)
    b = np.zeros(t.shape)
    cos_p = np.zeros(t.shape)
    sin_p = np.zeros(t.shape)
    cos_t = np.zeros(t.shape)
    sin_t = np.zeros(t.shape)
    rho_a = np.zeros(t.shape)
    Sa = np.zeros(t.shape)
    Ta = np.zeros(t.shape)
    ua = np.zeros(t.shape)
    E = np.zeros(t.shape)

    for i in range(len(t)):
        if i > 0:
            q0_local.update(t[i - 1], q[i - 1, :], profile, particles)
        q_local.update(t[i], q[i, :], profile, particles)
        M0[i] = q_local.M
        S[i] = q_local.S
        T[i] = q_local.T
        for j in range(n_part):
            Mpf[i, j, 0:len(q_local.M_p[j])] = q_local.M_p[j][:]
            Mp[i, j] = np.sum(particles[j].m[:])
            Tp[i, j] = particles[j].T
            xp[i, j * 3:j * 3 + 3] = q_local.x_p[j, :]
        Hp[i, :] = q_local.H_p
        u[i] = q_local.u
        v[i] = q_local.v
        w[i] = q_local.w
        V[i] = q_local.V
        h[i] = q_local.h
        x0[i] = q_local.x
        y0[i] = q_local.y
        z0[i] = q_local.z
        s0[i] = q_local.s
        rho[i] = q_local.rho
        b[i] = q_local.b
        cos_p[i] = q_local.cos_p
        sin_p[i] = q_local.sin_p
        cos_t[i] = q_local.cos_t
        sin_t[i] = q_local.sin_t
        rho_a[i] = q_local.rho_a
        Sa[i] = q_local.Sa
        Ta[i] = q_local.Ta
        ua[i] = q_local.ua
        E[i] = entrainment(q0_local, q_local, p)

    # Compute the unit vector along the plume axis
    Sz = sin_p
    Sx = cos_p * cos_t
    Sy = cos_p * sin_t

    # Extract the trajectory variables
    x = q[:, 7]
    y = q[:, 8]
    z = q[:, 9]
    s = q[:, 10]
    M = q[:, 0]
    x0, y0, z0 = x[0], y[0], z[0]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 6))

    axes[0, 0].plot(x, z)
    for i in range(len(particles)):
        axes[0, 0].plot(xp[:, i * 3], xp[:, i * 3 + 2], '.--')
    axes[0, 0].set_xlabel('x (m)')
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_ylabel('z (m)')
    axes[0, 0].set_ylim(z0, )
    x1, z1, x2, z2 = width_projection(Sx, Sz, b)
    axes[0, 0].plot(x + x1, z + z1, 'b--')
    axes[0, 0].plot(x + x2, z + z2, 'b--')

    axes[0, 1].plot(y, z)
    for i in range(len(particles)):
        axes[0, 1].plot(xp[:, i * 3 + 1], xp[:, i * 3 + 2], '.--')
    axes[0, 1].set_xlabel('x (m)')
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_ylabel('z (m)')
    axes[0, 1].set_ylim(z0, )
    y1, z1, y2, z2 = width_projection(Sy, Sz, b)
    axes[0, 1].plot(y + y1, z + z1, 'b--')
    axes[0, 1].plot(y + y2, z + z2, 'b--')

    axes[1, 0].plot(x, y)
    for i in range(len(particles)):
        axes[1, 0].plot(xp[:, i * 3], xp[:, i * 3 + 1], '.--')
    axes[1, 0].set_xlabel('x (m)')
    axes[1, 0].set_ylabel('y (m)')
    x1, y1, x2, y2 = width_projection(Sx, Sy, b)
    axes[1, 0].plot(x + x1, y + y1, 'b--')
    axes[1, 0].plot(x + x2, y + y2, 'b--')

    axes[1, 1].plot(s, M)
    axes[1, 1].set_xlabel('s (m)')
    axes[1, 1].set_ylabel('M (kg)')
    axes[1, 1].set_xlim(-2, )

    plt.show()


def plot_3d(t, q, q_local, profile, p, particles):
    # Extract the particle positions from the q state space
    xp = np.zeros((len(t), 3 * len(particles)))
    for i in range(len(t)):
        q_local.update(t[i], q[i, :], profile, particles)
        for j in range(len(particles)):
            xp[i, j * 3:j * 3 + 3] = q_local.x_p[j, :]

    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection='3d')
    plt.gca().invert_zaxis()

    q0_local = LagElement(t[0], q[0, :], q_local.diameter, profile, particles, q_local.chem_names)
    # for i in range(1, len(t)):  # -78 -79  -72
    for i in range(1, len(t)):
        if i > 0:
            q0_local.update(t[i - 1], q[i - 1, :], profile, particles)
        q_local.update(t[i], q[i, :], profile, particles)
        p0 = np.array([q0_local.x, q0_local.y, q0_local.z])
        p1 = np.array([q_local.x, q_local.y, q_local.z])
        R0 = q0_local.b
        R1 = q_local.b
        plot_cone(p0, p1, R0, R1, ax, n=30)

    ax.plot3D(q[:, 7], q[:, 8], q[:, 9], 'r')
    x0, y0, z0 = q[:, 7][0], q[:, 8][0], q[:, 9][0]
    print('Max depth: {0}'.format(max(q[:, 9])))
    print('Intrusion depth: {0}'.format(q[:, 9][-1]))

    # for i in range(len(particles)):
    #     ax.scatter3D(xp[:, i * 3], xp[:, i * 3 + 1], xp[:, i * 3 + 2])

    ax.set_zlim(z0, )
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    # ax.xaxis.set_tick_params(color='white')
    # ax.yaxis.set_tick_params(color='white')
    # ax.zaxis.set_tick_params(color='white')
    # ax.xaxis.set_ticklabels([])
    # ax.yaxis.set_ticklabels([])
    # ax.zaxis.set_ticklabels([])
    elev = 5
    azim = -125
    ax.view_init(elev, azim)
    ax.tick_params(axis='x', pad=5, labelsize=11)  # Adjust the pad for the X-axis
    ax.tick_params(axis='y', pad=5, labelsize=11)  # Adjust the pad for the Y-axis
    ax.tick_params(axis='z', pad=5, labelsize=11)  # Adjust the pad for the Z-axis
    ax.set_xlabel('x (m)', labelpad=15, fontdict={'size': 16})
    ax.set_ylabel('y (m)', labelpad=15, fontdict={'size': 16})
    ax.set_zlabel('z (m)', labelpad=15, fontdict={'size': 16})
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    ax.set_zlim(1500, 900)

    plt.show()
    # plt.savefig('near_field3d.jpg', dpi=900)


def plot_cone(p0, p1, r0, r1, ax, n=10):

    # vector in direction of axis
    v = p1 - p0
    # find magnitude of vector
    mag = norm(v)
    # unit vector in direction of axis
    v = v / mag
    # make some vector not in the same direction as v
    not_v = np.array([1, 1, 0])
    if (v == not_v).all():
        not_v = np.array([0, 1, 0])
    # make vector perpendicular to v
    n1 = np.cross(v, not_v)
    # normalize n1
    n1 /= norm(n1)
    # make unit vector perpendicular to v and n1
    n2 = np.cross(v, n1)
    # surface ranges over t from 0 to length of axis and 0 to 2*pi
    t = np.linspace(0, mag, n)
    theta = np.linspace(0, 2 * np.pi, n)
    # use meshgrid to make 2d arrays
    t, theta = np.meshgrid(t, theta)
    R = np.linspace(r0, r1, n)
    # generate coordinates for surface
    X, Y, Z = [p0[i] + v[i] * t + R * np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]
    ax.plot_surface(X, Y, Z, color='b', linewidth=0, alpha=0.2)


class ModelParams(object):
    """
    Parameters
    ----------
    profile : `vertical_profile.Profile1d` object

    Attributes
    ----------
    alpha_j : float
        Jet shear entrainment coefficient.
    alpha_Fr : float
        Plume entrainment coefficient in Froude-number expression.
    gamma : float
        Momentum amplification factor
    Fr_0 : float
        Initial plume Froude number for the Wuest et al. (1992) multiphase
        plume initial conditions
    rho_r : float
        Reference density (kg/m^3) evaluated at mid-depth of the water body.
    g : float
        Acceleration of gravity (m/s^2)
    R : float
        Ideal gas constant (J/mol/K)

    """
    def __init__(self, profile):
        # Store a reference density for the water column
        z_ave = profile.z_max - (profile.z_max - profile.z_min) / 2.
        P, T, S, U, V, W = profile.get_values(z_ave)
        self.rho_r = seawater.density(T, S, P)

        # Store some physical constants
        self.g = 9.81
        self.R = 8.314510

        # Set the model parameters to the values in Jirka (2004)
        self.alpha_j = 0.057
        self.alpha_Fr = 0.544
        self.gamma = 1.10

        # Set some multiphase plume model parameters
        self.Fr_0 = 1.6