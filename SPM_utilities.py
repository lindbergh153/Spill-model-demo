"""
SPM Utility Classes
===================

Data classes for particle cloud and oil slick state management.

Classes
-------
ParticleCloud : Subsurface particle cloud (oil droplets/gas bubbles)
Slick : Surface oil slick with weathering state

"""

from __future__ import annotations

from weathering_functions import *


class ParticleCloud(object):
    def __init__(self, particle, m0_Cloud, x, y, z):
        """
        Object describing the properties of the cloud (a collection of particles with the same size).
        It represents the underwater particle cloud consisting of
        oil droplets or gas bubbles having the same size.

        :param particle: particle_API.PlumeParticle
            Object combines TAMOC's PlumeParticle and bent_plume_model.Particle
        :param m0_Cloud: float
            The initial mass of the particle cloud released from the near-field (kg).
        :param x: float
            Longitude of cloud
        :param y: float
            Latitude of cloud
        :param z:
            Depth of cloud (m)

        Attributes
        ----------
        m : ndarray
            The mass of each component in a cloud (kg).
        mf : list
            The mass fraction of each component
        m_list : list
            A list to record the mass of cloud
        age : float, default 0
            The time since the cloud was released (hours).
        underwater : bool, default True
            A flag indicates whether cloud is underwater
        q_type : integer
            Defines the fluid type (0 = gas, 1 = liquid) that is expected to be
            contained in the particle (FluidParticle).
        time : float, default None
            Current time recorded for obtaining the surfacing time of cloud (hours)

        """
        self.particle = particle
        self.m0_Cloud = m0_Cloud
        self.m_Cloud = m0_Cloud
        self.m0_particle = self.particle.m0
        self.m_particle = self.particle.m
        self.mf = get_mf(self.m_Cloud)
        self.m_Cloud_list = []
        self.x_Cloud_list = []
        self.y_Cloud_list = []
        self.z_Cloud_list = []
        self.x = x
        self.y = y
        self.z = 0 if z < 0 else z
        self.age = 0
        self.underwater = True
        self.q_type = self.particle.FluidParticle.fp_type
        self.time = None

    def update(self, m_particle, x, y, z, dt):
        """
        update the status of 'ParticleCloud' object, including the mass of particles in 'ParticleCloud',
        location of particles, and the time step used for simulation.

        :param m_particle: ndarray
            Current masses (kg) of the particle components
        :param x: float
            Longitude of cloud
        :param y: float
            Latitude of cloud
        :param z: float
            Depth of cloud (m)
        :param dt: float
            The time step size used for simulation (hour)

        """

        # update the mass of particles in the cloud
        self.m_particle = m_particle

        # get the remaining fraction (compared to the initial mass) of each component's mass in a particle
        mr_particle = m_particle / self.m0_particle

        # update the current mass of cloud
        self.m_Cloud = self.m0_Cloud * mr_particle

        # add the new cloud mass into the list
        self.m_Cloud_list.append(sum(self.m_Cloud))
        self.x_Cloud_list.append(self.x)
        self.y_Cloud_list.append(self.y)
        self.z_Cloud_list.append(self.z)

        # update the composition of cloud
        self.mf = get_mf(self.m_Cloud)

        # update the location of cloud
        self.x = x
        self.y = y
        self.z = 0 if z < 0 else z

        # update the age of cloud
        self.age += dt


class Slick(object):
    def __init__(self, particle, m, x, y, z, A, Y, Ymax, mu, rho, sigma, t_emul, emulsion_type):
        """
        Object describing the properties of the oil slick.

        :param particle: particle_API.PlumeParticle
            Object combines TAMOC's PlumeParticle and bent_plume_model.Particle
            To obtain some properties of oil
        :param m: ndarray
            Current masses (kg) of each slick component
        :param x: float
            Longitude of a slick
        :param y: float
            Latitude of a slick
        :param z: float
            Depth of a slick (m), it should be zero. Here we use the minimal depth of the current data
        :param A: float
            The exposed area of an oil slick (m)
        :param Y: float
            The water content of an oil slick
        :param Ymax: float
            The maximum water content of an oil slick
        :param mu: float
            The dynamic viscosity of an oil slick (Pa s)
        :param rho: float
            The density of an oil slick (kg/m^3)
        :param t_emul: float
            The time required for the formation of emulsion
        :param emulsion_type: int
            The class of emulsions, including 1) 0--stable, 2) 1--Meso-stable,
            3) 2--entrained, 4) 3--unstable, 5) 10--stable.

        Attributes
        ----------
        m0 : ndarray
            Initial masses (kg) of each slick component (kg)
        m_Slick_list : ndarray
            Initial masses (kg) of each slick component (kg)
        mf : ndarray
            The mass fraction of each component
        mu0: float
            The initial dynamic viscosity of an oil slick (Pa s)
        age : float, default 0
            The time since the cloud was released (hours).
        delta : float
            The thickness of oil slick (m).
        evap_list: list
            A list records the mass of evaporated oil (kg)
        disp_list: list
            A list records the mass of dispersed oil (kg)
        time: list
            A list records time step of simulation (hours)
        iter_solver: list
            A list records the iteration times of solver
        iter_evap: list
            A list records the iteration times of evaporation module
        iter_disp: list
            A list records the iteration times of dispersion module

        """
        self.particle = particle
        self.composition = particle.composition
        self.m0 = m
        self.m = m
        self.m_Slick_list = []
        self.x_Slick_list = []
        self.y_Slick_list = []
        self.z_Slick_list = []
        self.mf = get_mf(self.m)
        self.x = x
        self.y = y
        self.z = z
        self.A = A
        self.Y = Y
        self.Ymax = Ymax
        self.mu = mu
        self.mu0 = mu
        self.rho = rho
        self.rho0 = rho
        self.nu0 = self.mu0 / self.rho0
        self.nu = self.nu0
        self.sigma = sigma
        self.sigma0 = sigma
        self.age = 0
        self.strand = False
        self.t_emul = t_emul
        self.emulsion_type = emulsion_type
        self.delta = get_thickness(self)
        self.evap_list = []
        self.disp_list = []
        self.time = []
        self.iter_solver = []
        self.iter_evap = []
        self.iter_disp = []

    def update(self, m, x, y, A, Y, p, dt=0):
        """
        update the status of 'Slick' object, including mass, location, area, and water content.

        :param m: ndarray
            Current masses (kg) of each slick component
        :param x: float
            Longitude of a slick
        :param y: float
            Latitude of a slick
        :param A: float
            The exposed area of an oil slick (m)
        :param Y: float
            The water content of an oil slick
        :param p: `ModelParams`
            Container for the fixed model parameters
        :param dt:  float
            The time step size used for simulation (hour)

        """

        self.m = m
        self.mf = get_mf(self.m)
        self.m_Slick_list.append(sum(self.m))
        self.x, self.y, z = get_location(x, y, 0, self.m)
        self.x_Slick_list.append(self.x)
        self.y_Slick_list.append(self.y)
        self.z_Slick_list.append(0)
        self.age += dt
        self.A = get_slick_area(A, self.m)
        self.mu = emulsion_vis(self)
        self.rho = emulsion_den(self, p)
        self.nu = get_nu(self.mu, self.rho, self.m)
        self.sigma = get_sigma(self.particle, self.m, p)
        self.Y = get_water_content(Y, self.m)
        self.delta = get_thickness(self)

    def update_fate(self, evap_rate, disp_rate, dt, t):
        """
        Get the mass of evaporated and dispersed oil at each simulation step

        :param evap_rate: float
            The rate of evaporation for each pseudo component
        :param disp_rate: float
            The rate of dispersion for each pseudo component
        :param dt: float
            The time step size used for simulation (hour)
        :param t: float
            Current simulation time (h)

        """

        # Get the mass of evaporated and dispersed oil in a simulation step
        oil_evap = sum(abs(evap_rate)) * dt
        oil_disp = sum(abs(disp_rate)) * dt

        # Record the evaporated and dispersed mass
        self.evap_list.append(oil_evap)
        self.disp_list.append(oil_disp)

        # Record the current time
        self.time.append(t)

    def get_k(self, iter_solver, iter_weather):
        """
        Obtain the exact iteration times used by solver or weathering module

        :param iter_solver: bool or int
            Determine whether recording the iteration times of solver
        :param iter_weather: bool
            Determine whether recording the iteration times of weathering module

        """

        # If the iteration times of solver is not False, record the iteration times of solver
        if iter_solver is not False:
            self.iter_solver.append(iter_solver)
        # otherwise, if the list 'iter_solver' is empty,  ; else,
        else:
            if len(self.iter_solver) == 0:
                self.iter_solver.append(1)
            else:
                self.iter_solver.append(self.iter_solver[-1])

        if iter_weather is None:
            pass
        else:
            if len(self.iter_evap) == 0:
                self.iter_evap.append(1)
                self.iter_disp.append(1)
            else:
                hold_evap = self.iter_evap[-1]
                hold_evap += 1
                self.iter_evap.append(hold_evap)

                hold_disp = self.iter_disp[-1]
                hold_disp += 1
                self.iter_disp.append(hold_disp)


def get_mb(iter_solver, evap_list, disp_list, time, y_sur):
    """
    Get the mass balance of a spill

    :param iter_solver: list
        A list records the iteration times of solver
    :param evap_list: list
        A list records the mass of evaporated oil (kg)
    :param disp_list: list
        A list records the mass of dispersed oil (kg)
    :param time: list
        A list records time step of simulation (hours)
    :param y_sur: ndarray
        A container of FFM solution

    :return evap_list: list
        A list records the mass of evaporated oil (kg)
    :return disp_list: list
        A list records the mass of dispersed oil (kg)
    :return time: list
        A list records time step of simulation (hours)
    :return iter_solver: list
        A list records the iteration times of solver

    """

    # Avoid invalidity due to denominator is zero
    np.seterr(invalid='ignore')

    # Remove the repeat element in a list that records the iteration times of solver
    iter_solver = remove_repeat(iter_solver)

    evap_list, disp_list, time, iter_solver = correct_fate(evap_list, disp_list, iter_solver, time)

    sur_mass = np.sum(y_sur[:, 4:-1], axis=1)

    diff_sur_mass = abs(np.diff(sur_mass, axis=0))
    frac_evap = evap_list / (evap_list + disp_list)
    frac_disp = disp_list / (evap_list + disp_list)
    evap_list = frac_evap * diff_sur_mass
    disp_list = frac_disp * diff_sur_mass
    evap_list[np.isnan(evap_list)] = 0
    disp_list[np.isnan(disp_list)] = 0

    evap_list = np.cumsum(evap_list)
    disp_list = np.cumsum(disp_list)

    return evap_list, disp_list, time, iter_solver


def get_location(x, y, z, mass):
    """
    Get the location of a slick

    :param x: float
        Longitude of slick
    :param y: float
        Latitude of slick
    :param z: float
        Depth of slick (m)
    :param mass: ndarray
        Current masses (kg) of each slick component
    :return x, y, z: float
        Oil location
    """
    if sum(mass) == 0:
        x, y, z = np.NAN, np.NAN, np.NAN

    return x, y, z


def get_slick_area(A, mass):
    """
    Get the exposed area of a slick

    :param A: float
        The exposed area of an oil slick (m)
    :param mass: ndarray
        The mass of each component in a cloud or a particle (kg).
    :return A: float
        The exposed area of an oil slick (m)

    """
    if sum(mass) == 0:
        A = np.NAN

    return A


def get_mf(mass):
    """
    Get the mass composition of a cloud or slick

    :param mass: ndarray
        The mass of each component in a cloud or a particle (kg).
    :return mf: ndarray
        The mass fraction of a cloud or a particle.

    """
    if sum(mass) == 0:
        mf = np.zeros_like(mass)
    else:
        mf = mass / sum(mass)

    return mf


def get_water_content(Y, mass):
    """
    Get the water content of a slick

    :param Y: float
        The water content of an oil slick
    :param mass: ndarray
        Masses (kg) of each slick component
    :return Y: float

    """
    if sum(mass) == 0:
        Y = np.NAN

    return Y


def get_nu(mu, rho, mass):
    if sum(mass) == 0:
        nu = np.NAN
    else:
        nu = mu / rho

    return nu


def get_sigma(particle, mass, p):
    if sum(mass) == 0:
        sigma = np.NAN
    else:
        sigma = particle.FluidParticle.interface_tension(mass, p.T_sur, p.S_sur, p.P_sur)

    return sigma


def remove_repeat(iter_solver):
    """

    :param iter_solver:
    :return:
    """
    if isinstance(iter_solver, list):
        it_solver = np.array(iter_solver)

    index_delete = []
    for i, (a, b) in enumerate(zip(iter_solver[:-1], iter_solver[1:])):
        if a != b:
            index_delete.append(i)
    it_solver = np.delete(iter_solver, index_delete)

    return it_solver


def correct_fate(evap_mass, disp_mass, it_solver, time):
    evap = []
    disp = []
    t = []
    solver = []

    for i, (a, b) in enumerate(zip(it_solver[:-1], it_solver[1:])):
        if a != b:
            evap.append(evap_mass[i])
            disp.append(disp_mass[i])
            t.append(time[i])
            solver.append(it_solver[i])

    evap.insert(0, evap_mass[0])
    disp.insert(0, disp_mass[0])
    t.insert(0, time[0])
    solver.insert(0, it_solver[0])

    evap = np.array(evap)
    disp = np.array(disp)
    t = np.array(t)
    solver = np.array(solver)

    return evap, disp, t, solver