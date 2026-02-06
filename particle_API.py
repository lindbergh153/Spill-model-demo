"""
Particle API Classes
====================

High-level interfaces for plume particles with state tracking
and property calculations.

Classes
-------
SingleParticle : Base interface for FluidParticle objects
PlumeParticle : Particle tracking within the plume model

"""

from __future__ import annotations

import numpy as np
from numpy.linalg import inv

import seawater
from plume_model_functions import local_coords


class SingleParticle(object):
    """
    Interface to the `FluidParticle` module and container for model parameters.
    This class provides a uniform interface to the `FluidParticle` module objects and
    methods and stores the particle-specific model parameters.

    Parameters
    ----------
    FluidParticle : particle.FluidParticle
        Object describing the particle properties and behavior
    m0 : ndarray
        Initial masses of the components of the `FluidParticle` object (kg)
    T0 : float
        Initial temperature of the of `FluidParticle` object (K)
    fdis : float, default = 1e-6
        Fraction of the initial total mass (--) remaining when the particle
        should be considered dissolved.
    t_hyd : float, default = 0.
        Hydrate film formation time (s). Mass transfer is computed by clean
        bubble methods for t less than t_hyd and by dirty bubble methods
        thereafter.  The default behavior is to assume the particle is dirty
        or hydrate covered from the release.

    Attributes
    ----------
    composition : str list
        Copy of the `composition` attribute of the `dbm_particle` object.
    cp : float
        Heat capacity at constant pressure (J/(kg K)) of the particle.
    diss_indices : ndarray bool
        Indices of m0 that are non-zero.

    Dissolution is turned off component by component as each component mass
    becomes fdis times smaller than the initial mass.  Once all the initial
    components have been turned off, the particle is assumed to have a
    density equation to the ambient water and a slip velocity of zero.

    Heat transfer is turned off once the particle comes within 0.1 K of the
    ambient temperature.  Thereafter, the temperature is forced to track
    the ambient temperature.
    """

    def __init__(self, FluidParticle, m0, T0, fdis=1e-6, t_hyd=0):
        # Store the input parameters
        self.FluidParticle = FluidParticle
        self.composition = FluidParticle.composition
        self.m0 = m0
        self.T0 = T0
        self.t_hyd = t_hyd
        self.cp = seawater.cp() * 0.5
        self.fdis = fdis

        # Store parameters to track the dissolution of the initial masses
        self.diss_indices = self.m0 > 0

    def properties(self, m, T, P, Sa, Ta, t):
        """
        Return the particle properties from FluidMixture

        Provides a single interface to the `return_all` methods of the fluid
        particle objects defined in the `particle` module.
        This method also applies the particle-specific model parameters to
        adjust the mass and heat transfer and determine the dissolution state.

        Parameters
        ----------
        m : ndarray
             mass of the particle (kg)
        T : float
             particle temperature (K)
        P : float
            particle pressure (Pa)
        Sa : float
            salinity of ambient seawater (psu)
        Ta : float
            temperature of ambient seawater (K)
        t : float
            age of the particle--time since it was released into the water column (s)

        Returns
        -------
        A tuple containing:

            us : float
                slip velocity (m/s)
            rho_p : float
                particle density (kg/m^3)
            A : float
                surface area (m^2)
            Cs : ndarray, size (nc)
                solubility (kg/m^3)
            K * beta : ndarray, size (nc)
                effective mass transfer coefficient(s) (m/s)
            K_T * beta_T : float
                effective heat transfer coefficient (m/s)
            T : float
                temperature of the particle (K)
        """

        # Decide which slip velocity and mass and heat transfer to use
        if t < self.t_hyd:
            # Treat the particle as clean for slip velocity and mass transfer
            status = 1
        else:
            # Use the dirty bubble slip velocity and mass transfer
            status = -1

        # Stop oscillations at small mass
        m[m < 0] = 0.

        # Get all the particle properties required by simulation
        shape, de, rho_p, us, A, Cs, beta, beta_T = self.FluidParticle.return_all(m, T, P, Sa, Ta, status)

        # Turn off dissolution for "dissolved" components
        # present fraction of oil components
        frac_diss = np.ones(np.size(m))

        # present fraction of oil components
        frac_diss[self.diss_indices] = m[self.diss_indices] / self.m0[self.diss_indices]

        # set the mass transfer coefficients of dissolved components as zero
        beta[frac_diss < self.fdis] = 0.

        # Shut down bubble forces when particles fully dissolve
        if np.sum(beta[self.diss_indices]) == 0.:
            # Injected chemicals have dissolved
            if np.sum(m[self.diss_indices]) > np.sum(m[~self.diss_indices]):
                # The whole particle has dissolved
                us = 0
                rho_p = seawater.density(Ta, Sa, P)

        # Return the particle properties
        return us, rho_p, A, Cs, beta, beta_T, T

    def diameter(self, m, T, P):
        """
        Compute the diameter of a particle from mass and density

        :param m: ndarray, size (nc)
            masses of each component in a particle (kg)
        :param T: float
            particle temperature (K)
        :param P: float
            particle pressure (Pa)
        :return: de: float
            equivalent spherical diameter of a fluid particle (m)
        """
        de = self.FluidParticle.diameter(m, T, P)

        return de


class PlumeParticle(SingleParticle):
    def __init__(self, x, y, z, FluidParticle, m0, T0, nb0, lambda_1, P, Sa, Ta, fdis=1.e-6, t_hyd=0.):
        """
        Special model properties for tracking inside a Lagrangian plume object

        This new `Particle` class allows dispersed phase particles to be tracked
        within the Lagrangian plume element during the solution and to exit the plume at the right time.

        This object inherits the `SingleParticle` object and
        adds functionality for three-dimensional positioning and particle tracking.

        Combine TAMOC's PlumeParticle and bent_plume_model.Particle

        :param x: float
            Initial position of the particle in the x-direction (m)
        :param y: float
            Initial position of the particle in the y-direction (m)
        :param z: float
            Initial position of the particle in the z-direction (m)
        :param FluidParticle: Particle.FluidParticle
            Object describing the particle properties and behavior
        :param m0: ndarray
            Initial masses of one particle for the components of the `FluidParticle` object (kg)
        :param T0: float
            Initial temperature of the of `FluidParticle` particle object (K)
        :param nb0: float
            Initial number flux of particles at the release (/s)
        :param lambda_1: float
            spreading rate of the dispersed phase in a plume (--)
        :param P: float
            Local pressure (Pa)
        :param Sa: float
            Local salinity surrounding the particle (psu)
        :param Ta: float
            Local temperature surrounding the particle (K)
        :param fdis: float, default = 1.e-6
            Fraction (--) of the initial mass of each component of the mixture
            when that component should be considered totally dissolved.
        :param t_hyd: float, default = 0.
            Hydrate film formation time (s).  Mass transfer is computed by clean
            bubble methods for t less than t_hyd and by dirty bubble methods
            thereafter.  The default behavior is to assume the particle is dirty
            or hydrate covered from the release.

        Attributes
        ----------
        nb0 : float
            Initial number flux of particles at the release (#/s)
        t : float
            Current time since the particle was released (s)
        x : float
            Current position of the particle in the x-direction (m)
        y : float
            Current position of the particle in the y-direction (m)
        z : float
            Current position of the particle in the z-direction (m)
        integrate : bool
            Flag indicating whether the particle is still inside the plume,
            where its trajectory should continue to be integrated.
        sim_stored : bool
            Flag indicating whether a simulation result is stored in the object memory

        """
        super(PlumeParticle, self).__init__(FluidParticle, m0, T0, fdis, t_hyd)

        # Store the input variables related to the particle description
        self.nb0 = nb0
        # Store the model parameters
        self.lambda_1 = lambda_1

        # Store the initial particle locations
        self.t = 0.
        self.x = x
        self.y = y
        self.z = z

        # Particles start inside the plume and should be integrated
        self.integrate = True

        # Indicate that the simulation has not yet been conducted
        self.sim_stored = False

        # Set the local masses and temperature to their initial values.
        # The particle age is zero at instantiation
        self.update(m0, T0, P, Sa, Ta, self.t)

    def update(self, m, T, P, Sa, Ta, t):
        """
        Store the instantaneous values of the particle properties.
        During the simulation, keep the state space variables for each particle stored within the particle.

        Parameters
        ----------
        m : ndarray
            Current masses (kg) of the particle components
        T : float
            Current temperature (K) of the particle
        P : float
            Local pressure (Pa)
        Sa : float
            Local salinity surrounding the particle (psu)
        Ta : float
            Local temperature surrounding the particle (K)
        t : float
            age of the particle--time since it was released into the water column (s)

        """

        # Update the variables with their current values
        self.m = m
        if np.sum(self.m) > 0:
            self.us, self.rho_p, self.A, self.Cs, self.beta, self.beta_T, self.T = self.properties(m, T, P, Sa, Ta, t)
        else:
            self.us = 0.
            self.rho_p = seawater.density(Ta, Sa, P)
            self.A = 0.
            self.Cs = np.zeros(len(self.composition))
            self.beta = np.zeros(len(self.composition))
            self.beta_T = 0.
            self.T = Ta

    def track(self, t_p, X_cl, X_p, q_local):
        """
        This method converts local plume coordinates to Cartesian coordinates.

        Track the location of the particle within a Lagrangian plume model
        element and stop the integration when the particle exits the plume.

        Parameters
        ----------
        t_p : float
            Time since the particle was released (s)
        X_cl : ndarray
            Array of Cartesian coordinates (x,y,z) for the plume centerline (m).
        X_p : ndarray
            Array of local plume coordinates (l,n,m) for the current particle position (m).
        q_local : `LagElement` object
            Object that translates the plume model state space `t` and
            `q` into the comprehensive list of derived variables.

        Returns
        -------
        xp : ndarray
            Array of Cartesian coordinates (x,y,z) for the current particle position (m).

        """
        if self.integrate:
            # if particle is situated within the plume region

            # Compute the transformation matrix from local plume coordinates (l,n,m)
            # to Cartesian coordinates (x,y,z)
            A = local_coords(q_local)
            Ainv = inv(A)

            # Update the particle age
            self.t = t_p
            tp = self.t

            # Get the particle position
            xp = np.dot(Ainv, X_p) + X_cl
            self.x = xp[0]
            self.y = xp[1]
            self.z = xp[2]

            # Compute the particle offset from the plume centerline
            lp = np.sqrt(X_p[0] ** 2 + X_p[1] ** 2 + X_p[2] ** 2)

            # Compute the buoyant force reduction factor
            self.p_fac = (q_local.b - lp) ** 4 / q_local.b ** 4
            if self.p_fac < 0.:
                self.p_fac = 0.

            # Check if the particle exited the plume
            if lp > q_local.b:
                self.p_fac = 0.
                self.b_local = q_local.b

        else:
            # Return the time and position when the particle exited the plume
            tp = self.te
            xp = np.array([self.xe, self.ye, self.ze])
            self.p_fac = 0.

        # Return the particle position as a matrix
        return tp, xp


def ic_particle(profile, z0, FluidParticle, yk, q, de, T0=None):
    """
    Define initial conditions for a PlumeParticle based on certain flow rate

    Returns the standard variables describing a particle as needed to
    initialize a PlumeParticle object from specification of the dispersed phase flow rate.

    Parameters
    ----------
    profile : `Profile1d` object
        The ambient CTD object used by the simulation.
    z0 : float
        Depth of the release point (m)
    FluidParticle : FluidParticle
        Object describing the particle properties and behavior
    yk : ndarray
        mol fractions of each component of the dispersed phase particle.
    q : float
        Flux of the dispersed phase, either as the volume flux (m^3/s) at
        standard conditions, defined as 0 deg C and 1 bar, or as mass flux (kg/s).
    de : float
        Initial diameter (m) of the particle
    T0 : float, default = None
        Initial temperature of the of `FluidParticle` object (K).

    Returns
    -------
    m0 : ndarray
        Initial masses of the components of one particle in the `FluidParticle` particle object (kg)
    T0 : float
        Initial temperature of the of `FluidParticle` object (K)
    nb0 : float
        Initial number flux of particles at the release (--)
    P : float
        Local pressure (Pa)
    Sa : float
        Local salinity surrounding the particle (psu)
    Ta : float
        Local temperature surrounding the particle (K)

    """

    # Get the ambient conditions at the release
    P, Ta, Sa, Ua, Va, Wa = profile.get_values(z0)

    # Get the particle temperature
    if T0 is None: T0 = Ta

    # Compute the density at standard and in situ conditions
    mass_frac = FluidParticle.mass_frac(yk)
    rho_p = FluidParticle.density(mass_frac, T0, P)

    # Get the mass and number flux of particles
    # The input flux is the total mass flux (kg/s)
    m_dot = q

    # Get the source volume flux and particle number flux
    Q = m_dot / rho_p  # (m^3/s)
    nb0 = Q / (np.pi * de ** 3 / 6)  # (number of particles/s)

    # Get the mass of each component in the initial particle (kg)
    m0 = m_dot / nb0 * mass_frac

    return m0, T0, nb0, P, Sa, Ta


class ModelParams(object):
    """
    Fixed model parameters for the PlumeParticle

    Parameters
    ----------
    profile : `vertical_profile.Profile1d` object

    Attributes
    ----------
    rho_r : float
        Reference density (kg/m^3) evaluated at mid-depth of the water body.
    g : float
        Acceleration of gravity (m/s^2)
    Ru : float
        Ideal gas constant (J/mol/K)

    """

    def __init__(self, profile):
        super(ModelParams, self).__init__()

        # Store a reference density for the water column
        z_ave = profile.z_max - (profile.z_max - profile.z_min) / 2.
        P, T, S, Ua, Va, Wa = profile.get_values(z_ave)
        self.rho_r = seawater.density(T, S, P)

        # Store some physical constants
        self.g = 9.81
        self.Ru = 8.314510