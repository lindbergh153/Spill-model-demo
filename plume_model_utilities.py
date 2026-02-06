"""
Plume Model Utility Classes
===========================

Data classes for Lagrangian element state and plume model parameters.

Classes
-------
LagElement : State container for a Lagrangian plume element
width_projection : Plume width for visualization

"""

from __future__ import annotations

import numpy as np

import seawater
from plume_model_functions import entrainment, track_particles


class LagElement(object):
    """
    Parameters
    ----------
    t0 : float
        Initial time of the simulation (s)
    q0 : ndarray
        Initial values of the simulation state space, q
    diameter : float
        Diameter for the equivalent circular cross-section of the release (m)
    profile : `ambient.Profile`
        Ambient CTD data
    particles : list of `Particle` objects
        List of `Particle` objects describing each dispersed phase in the
        simulation

    Attributes
    ----------
    t0 : float
        Initial time of the simulation (s)
    q0 : ndarray
        Initial values of the simulation state space, q
    diameter : float
        Diameter for the equivalent circular cross-section of the release (m)
    len : int
        Number of variables in the state space q (--)
    np : int
        Number of dispersed phase particles (--)
    t : float
        Independent variable for the current time (s)
    q : ndarray
        Dependent variable for the current state space
    M : float
        Mass of the Lagrangian element (kg)
    Se : float
        Salt in the Lagrangian element (psu kg)
    He : float
        Heat of the Lagrangian element (J)
    Jx : float
        Dynamic momentum of the Lagrangian element in the x-direction
        (kg m/s)
    Jy : float
        Dynamic momentum of the Lagrangian element in the y-direction
        (kg m/s)
    Jz : float
        Dynamic momentum of the Lagrangian element in the z-direction
        (kg m/s)
    H : float
        Relative thickness of the Lagrangian element h/V (s)
    x : float
        Current x-position of the Lagrangian element (m)
    y : float
        Current y-position of the Lagrangian element (m)
    z : float
        Current z-position of the Lagrangian element (m)
    s : float
        Current s-position along the centerline of the plume for the
        Lagrangian element (m)
    M_p : dict of ndarrays
        For integer key: the total mass fluxes (kg/s) of each component in a
        particle.
    H_p : ndarray
        Total heat flux for each particle (J/s)
    t_p : ndarray
        Time since release for each particle (s)
    X_p : ndarray
        Position of each particle in local plume coordinates (l,n,m) (m).
    Pa : float
        Ambient pressure at the current element location (Pa)
    Ta : float
        Ambient temperature at the current element location (K)
    Sa : float
        Ambient salinity at the current element location (psu)
    ua : float
        Crossflow velocity in the x-direction at the current element location
        (m/s)
    rho_a : float
        Ambient density at the current element location (kg/m^3)
    S : float
        Salinity of the Lagrangian element (psu)
    T : float
        Temperature of the Lagrangian element (T)
    u : float
        Velocity in the x-direction of the Lagrangian element (m/s)
    v : float
        Velocity in the y-direction of the Lagrangian element (m/s)
    w : float
        Velocity in the z-direction of the Lagrangian element (m/s)
    hvel : float
        Velocity in the horizontal plane for the Lagrangian element (m/s)
    V : float
        Velocity in the s-direction of the Lagrangian element (m/s)
    h : float
        Current thickness of the Lagrangian element (m)
    rho : float
        Density of the entrained seawater in the Lagrangian element (kg/m^3)
    b : float
        Half-width of the Lagrangian element (m)
    sin_p : float
        The sine of the angle phi (--)
    cos_p : float
        The cosine of the angle phi (--)
    sin_t : float
        The sine of the angle theta (--)
    cos_t : float
        The cosine of the angle theta (--)
    phi : float
        The vertical angle from horizontal of the current plume trajectory
        (rad in range +/- pi/2).  Since z is positive down (depth), phi =
        pi/2 point down and -pi/2 points up.
    theta : float
        The lateral angle in the horizontal plane from the x-axis to the
        current plume trajectory (rad in range 0 to 2 pi)
    mp : ndarray
        Masses of each of the dispersed phase particles in the `particles`
        variable
    fb : ndarray
        Buoyant force for each of the dispersed phase particles in the
        `particles` variable as density difference (kg/m^3)
    x_p : ndarray
    M_p : float
        Total mass of dispersed phases in the Lagrangian element (kg)
    Fb : float
        Total buoyant force as density difference of the dispersed phases in
        the Lagrangian element (kg/m^3)

    """

    def __init__(self, t0, q0, diameter, profile, particles, chem_names):
        # Store the inputs to stay with the Lagrangian element
        self.t0 = t0
        self.q0 = q0
        self.diameter = diameter
        self.len = q0.shape[0]
        self.np = len(particles)
        self.chem_names = chem_names
        self.nchems = len(self.chem_names)

        # Extract the state variables and compute the derived quantities
        self.update(t0, q0, profile, particles)

    def update(self, t, q, profile, particles=[]):
        """
        Update the `LagElement` object with the current local conditions

        Extract the state variables and compute the derived quantities given
        the current local conditions.

        Parameters
        ----------
        t : float
            Current time of the simulation (s)
        q : ndarray
            Current values of the simulation state space, q
        profile : `ambient.Profile`
            Ambient CTD data
        particles : list of `Particle` objects
            List of `Particle` objects describing each dispersed phase in the
            simulation

        """
        # Save the current state space
        self.t = t
        self.q = q

        # Extract the state-space variables from q
        self.M, self.Se, self.He, self.Jx, self.Jy, self.Jz = q[0], q[1], q[2], q[3], q[4], q[5]
        self.H, self.x, self.y, self.z, self.s = q[6], q[7], q[8], q[9], q[10]
        idx = 11
        M_p = {}
        H_p = []
        t_p = []
        X_p = []
        for i in range(self.np):
            M_p[i] = q[idx:idx + particles[i].FluidParticle.nc]
            idx += particles[i].FluidParticle.nc
            H_p.extend(q[idx:idx + 1])
            idx += 1
            t_p.extend(q[idx:idx + 1])
            idx += 1
            X_p.append(q[idx:idx + 3])
            idx += 3
        self.M_p = M_p
        self.H_p = np.array(H_p)
        self.t_p = np.array(t_p)
        self.X_p = np.array(X_p)

        # Get the local ambient conditions
        self.Pa, self.Ta, self.Sa, self.ua, self.va, self.wa = profile.get_values(self.z)
        self.rho_a = seawater.density(self.Ta, self.Sa, self.Pa)
        # Compute the derived quantities
        self.S = self.Se / self.M
        self.T = self.He / (self.M * seawater.cp())
        self.rho = seawater.density(self.T, self.S, self.Pa)
        self.u = self.Jx / self.M
        self.v = self.Jy / self.M
        self.w = self.Jz / self.M
        self.hvel = np.sqrt(self.u ** 2 + self.v ** 2)
        self.V = np.sqrt(self.hvel ** 2 + self.w ** 2)
        self.h = self.H * self.V
        self.b = np.sqrt(self.M / (self.rho * np.pi * self.h))
        self.sin_p = self.w / self.V
        self.cos_p = self.hvel / self.V
        if self.hvel == 0.:
            # if hvel = 0, flow is purely along z; let theta = 0
            self.sin_t = 0.
            self.cos_t = 1.
        else:
            self.sin_t = self.v / self.hvel
            self.cos_t = self.u / self.hvel
        self.phi = np.arctan2(self.w, self.hvel)
        self.theta = np.arctan2(self.v, self.u)

        # Get the particle characteristics
        self.mp = np.zeros(self.np)
        self.fb = np.zeros(self.np)
        self.x_p = np.zeros((self.np, 3))

        for i in range(self.np):
            # If this is a post-processing call, update the status of the integration flag
            if particles[i].sim_stored:
                if np.isnan(self.X_p[i][0]):
                    particles[i].integrate = False
                else:
                    particles[i].integrate = True

            # Update the particles with their current properties
            m_p = self.M_p[i] / particles[i].nbe
            T_p = self.H_p[i] / (np.sum(self.M_p[i]) * particles[i].cp)
            particles[i].update(m_p, T_p, self.Pa, self.S, self.T, self.t_p[i])

            # Track the particle in the plume
            self.t_p[i], self.x_p[i, :] = particles[i].track(self.t_p[i],
                                                             np.array([self.x, self.y, self.z]),
                                                             self.X_p[i], self)

            # Get the mass of particles following this Lagrangian element
            self.mp[i] = np.sum(m_p) * particles[i].nbe

            # Compute the buoyant force coming from this set of particles
            self.fb[i] = self.rho / particles[i].rho_p * self.mp[i] * \
                         (self.rho_a - particles[i].rho_p) * particles[i].p_fac

            # Force the particle mass and bubble force to zero if the bubble has dissolved
            if self.rho == particles[i].rho_p:
                self.mp[i] = 0.
                self.fb[i] = 0.

        # Compute the net particle mass and buoyant force
        self.Fb = np.sum(self.fb)


def derivs(t, q, q0_local, q1_local, profile, p, particles):
    """
    Calculate the derivatives for the system of ODEs for a Lagrangian plume

    Parameters
    ----------
    t : float
        Current value for the independent variable (time in s).
    q : ndarray
        Current value for the plume state space vector.
    q0_local : `bent_plume_model.LagElement`
        Object containing the numerical solution at the previous time step
    q1_local : `bent_plume_model.LagElement`
        Object containing the numerical solution at the current time step
    profile : `ambient.Profile` object
        The ambient CTD object used by the simulation.
    p : `ModelParams` object
        Object containing the fixed model parameters for the bent
        plume model.
    particles : list of `Particle` objects
        List of `bent_plume_model.Particle` objects containing the dispersed
        phase local conditions and behavior.

    Returns
    -------
    yp : ndarray
        A vector of the derivatives of the plume state space.
    """

    # Set up the output from the function to have the right size and shape
    qp = np.zeros(q.shape)

    # Update the local Lagrangian element properties
    q1_local.update(t, q, profile, particles)

    # Get the entrainment flux
    md = entrainment(q0_local, q1_local, p)

    # Get the dispersed phase tracking variables
    fe, up, dtp_dt = track_particles(q0_local, q1_local, md, particles)

    # Conservation of Mass
    qp[0] = md

    # Conservation of salt and heat
    qp[1] = md * q1_local.Sa
    qp[2] = md * seawater.cp() * q1_local.Ta

    # Conservation of continuous phase momentum.  Note that z is positive down (depth).
    qp[3] = md * q1_local.ua
    qp[4] = md * q1_local.va
    qp[5] = - p.g / (p.gamma * p.rho_r) * \
            (q1_local.Fb + q1_local.M * (q1_local.rho_a - q1_local.rho)) + md * q1_local.wa

    # Constant h/V thickness to velocity ratio
    qp[6] = 0.

    # Lagrangian plume element advection (x, y, z) and s along the centerline trajectory
    qp[7] = q1_local.u
    qp[8] = q1_local.v
    qp[9] = q1_local.w
    qp[10] = q1_local.V

    # Conservation equations for each dispersed phase
    idx = 11

    # Compute mass and heat transfer for each particle
    for i in range(len(particles)):

        # Only simulate particles inside the plume
        if particles[i].integrate:

            # Dissolution mass transfer for each particle component
            dm_pc = - particles[i].A * particles[i].nbe * particles[i].beta * particles[i].Cs * dtp_dt[i]

            # Update continuous phase temperature with heat of solution
            qp[2] += np.sum(dm_pc * particles[i].FluidParticle.neg_dH_solR * p.R / particles[i].FluidParticle.M)

            # Biodegradation for each particle component
            k_bio_s = -particles[i].FluidParticle.k_bio / 3600
            dm_pb = k_bio_s * particles[i].m * particles[i].nbe * dtp_dt[i]

            # Conservation of mass for dissolution and biodegradation
            qp[idx:idx + q1_local.nchems] = dm_pc + dm_pb

            # Update position in state space
            idx += q1_local.nchems

            # Heat transfer between the particle and the ambient
            # qp[idx] = - particles[i].A * particles[i].nbe * particles[i].rho_p * particles[i].cp * \
            #           particles[i].beta_T * (particles[i].T - q1_local.T) * dtp_dt[i]
            qp[idx] = 0

            # Heat loss due to mass loss
            qp[idx] += np.sum(dm_pc + dm_pb) * particles[i].cp * particles[i].T

            # Take the heat leaving the particle and put it in the continuous phase fluid
            qp[2] -= qp[idx]
            idx += 1

            # Particle age
            qp[idx] = dtp_dt[i]
            idx += 1

            # Follow the particles in the local coordinate system (l,n,m) relative to the plume centerline
            qp[idx] = 0.
            idx += 1
            qp[idx] = (up[i, 1] - fe * q[idx]) * dtp_dt[i]
            idx += 1
            qp[idx] = (up[i, 2] - fe * q[idx]) * dtp_dt[i]
            idx += 1
        else:
            idx += particles[i].FluidParticle.nc + 5

    # Return the slopes
    return qp