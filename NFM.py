"""
Near-Field Model (NFM) API
==========================

High-level API for configuring and running near-field plume simulations.

Classes
-------
Blowout : Main interface for deep water blowout simulations

"""

from __future__ import annotations

import numpy as np

import particle_utilities as pu
from DSDmodel import DSD_model
from ambient_profile import Profile1d
from particle import FluidParticle
from particle_API import PlumeParticle, ic_particle
from plume_model import Model_plume


class Blowout(object):
    """
    The near-field model (NFM) aims to predict behaviors of plume.
    This class is an API of Plume model module.
    """

    def __init__(self, location, d0, substance, q_oil, gor,
                 phi_0, theta_0, num_gas_elements, num_oil_elements,
                 near_field_data, dt_max, SSDI, x0=0, y0=0):
        """

        :param z0: float
            The depth of the release point (m)
        :param d0: float
            Equivalent circular diameter of the release (m)
        :param substance: str
            An oil from the NOAA OilLibrary, this should be a string containing the Adios oil
            ID number (e.g., 'AD01554' for Louisiana Light Sweet).
        :param q_oil: float
            Release rate of the dead oil at the release point (bbl/day).
        :param gor: float, default=0
            Gas to oil ratio at standard surface conditions (ft^3/bbl)
        :param x0: float
            x-coordinate of the initial release point (m)
        :param y0: float
            y-coordinate of the initial release point (m)
        :param phi_0: float
            Vertical angle of the release relative to the horizontal plane; z is
            positive down so that -pi/2 represents a vertically upward flowing release (rad)
        :param theta_0: float
            Horizontal angle of the release relative to the x-direction (rad)
        :param num_gas_elements: int
            Number of gas bubble sizes to include in the gas bubble size distribution from DSD model
        :param num_oil_elements:  int
            Number of oil droplet sizes to include in the oil droplet size distribution from DSD model
        :param near_field_data: csv
            Vertical profile of ambient water temperature, current velocity, and salinity
        :param dt_max: float
            Maximum step size to take in the storage of the simulation solution (s)
        :param flash_calculation: bool
            A switch of the two-phase flash calculation

        """
        # Store the model parameters
        self.z0 = location[2]
        self.d0 = d0
        self.substance = substance
        self.q_oil = q_oil
        self.gor = gor
        self.x0 = x0
        self.y0 = y0
        self.phi_0 = phi_0
        self.theta_0 = theta_0
        self.num_gas_elements = num_gas_elements
        self.num_oil_elements = num_oil_elements
        self.near_field_data = near_field_data
        self.dt_max = dt_max
        self.SSDI = SSDI

        # Decide which phase flow rate is reported through q_oil
        if self.num_oil_elements > 0:
            # User is simulating oil; hence, oil flow rate should be given
            self.q_type = 1
        else:
            # User is simulating gas only; hence, gas flow rate should be given
            self.q_type = 0

        self._update()

    def _update(self):
        """
        Initialize plume_model for simulation run

        Set up the ambient profile, initial conditions, and model parameters
        for a new simulation run of the `plume_model`.

        """
        # Get an ambient Profile object
        self.profile = Profile1d(self.near_field_data)

        # Import the oil with the desired gas-to-oil ratio
        self.oil, self.mass_flux = pu.get_oil(substance=self.substance, q_oil=self.q_oil, gor=self.gor,
                                              fp_type=self.q_type)

        # Find the ocean conditions at the release
        self.P0, self.T0, self.S0, Ua, Va, Wa = self.profile.get_values(self.z0)

        # Define some the constant initial conditions
        self.Sj = 0.
        self.Tj = self.T0

        if self.gor > 0:
            # Compute the equilibrium mixture properties at the release
            m, xi, K = self.oil.equilibrium(self.mass_flux, self.Tj, self.P0)
        else:
            m = np.zeros((2, len(self.mass_flux)))
            m[self.q_type, :] = self.mass_flux

            xi = np.zeros((2, len(self.mass_flux)))
            xi[self.q_type, :] = (self.mass_flux / self.oil.M) / sum((self.mass_flux / self.oil.M))

        # Create the discrete bubble model objects for gas and liquid
        if self.gor > 0:
            self.gas = FluidParticle(self.oil.composition, fp_type=0, delta=self.oil.delta,
                                     user_data=self.oil.user_data)
        self.liq = FluidParticle(self.oil.composition, fp_type=1, delta=self.oil.delta,
                                 user_data=self.oil.user_data, oil_id=self.substance)

        # Compute the bubble and droplet volume size distributions
        self.breakup_model = DSD_model(self.profile, self.oil, self.mass_flux, self.z0,
                                       self.d0, self.substance, self.gor,
                                       self.num_oil_elements, self.num_gas_elements, self.SSDI)
        self.d_gas, self.vf_gas, d50_gas, self.d_liq, self.vf_liq, d50_liq = self.breakup_model.simulate()
        # print(self.d_gas)
        # self.breakup_model.plot(unit=1)
        plot_oil_gas_DSD(self.profile, self.oil, self.mass_flux, self.z0,
                         self.d0, self.substance, self.gor,
                         self.num_oil_elements, self.num_gas_elements)

        # Create the `plume_model` particle list
        self.disp_phases = []
        mass_gas = float(np.sum(m[0, :]))
        mass_liq = float(np.sum(m[1, :]))

        if self.gor > 0:
            self.disp_phases += assemble_particles(mass_gas, self.d_gas,
                                                   self.vf_gas, self.profile, self.gas,
                                                   xi[0, :], self.x0, self.y0, self.z0, self.Tj, 0.9)
        self.disp_phases += assemble_particles(mass_liq, self.d_liq,
                                               self.vf_liq, self.profile, self.liq,
                                               xi[1, :], self.x0, self.y0, self.z0, self.Tj, 0.98)

        # Set the stop criteria.
        # IF Distance along the plume centerline/Equivalent circular diameter (S/D)
        # is greater than the given sd_max, the simulation will cease.
        self.sd_max = 300 * self.z0 / self.d0

        # Create the initialized `plume_model` object
        self.plume_model = Model_plume(self.profile)

    def simulate(self):
        # Run the plume model
        self.plume_model.simulate(np.array([self.x0, self.y0, self.z0]),
                                  self.d0,
                                  self.phi_0,
                                  self.theta_0,
                                  self.Sj,
                                  self.Tj,
                                  self.disp_phases,
                                  self.dt_max,
                                  self.sd_max)


def assemble_particles(m_tot, d, vf, profile, oil, yk, x0, y0, z0, Tj, lambda_1):
    """
    Create a list of particle_API.PlumeParticle objects for the given particle properties,
    and then assemble them to add to a plume model simulation

    Parameters
    ----------
    m_tot : float
        Total mass flux of this fluid phase in the simulation (kg/s)
    d : ndarray
        Array of particle sizes for this fluid phase (m)
    vf : ndarray
        Array of volume fractions for each particle size for this fluid phase (--).
        This array should sum to 1.0.
    profile : vertical.Profile1d
        A vertical.Profile1d object with the ambient ocean water column data
    oil : FluidParticle
        A FluidParticle object that contains the desired oil database composition
    yk : ndarray
        Mole fractions of each compound in the chemical database of the FluidParticle object (--).
    x0, y0, z0 : floats
        Initial position of the particles in the simulation domain (m).
    Tj : float
        Initial temperature of the particles in the jet (K)
    lambda_1 : float
        Value of the dispersed phase spreading parameter of the jet integral model (--).

    Returns
    -------
    disp_phases : list of plume_model.Particle objects
        List of `particle_API.PlumeParticle` objects to be added in the plume model simulation
        based on the given input data.

    """
    # Create an empty list of particles embedded in the plume
    disp_phases = []

    # Add each particle in the distribution separately
    for i in range(len(d)):
        # Get the total mass flux of this fluid phase (kg/s) for the present particle size
        mb0 = vf[i] * m_tot

        # Get the properties of these particles at the source
        m0, T0, nb0, P, Sa, Ta = ic_particle(profile, z0, oil, yk, mb0, d[i], Tj)

        # Append these particles to the list of particles in the simulation
        disp_phases.append(PlumeParticle(x0, y0, z0, oil, m0, T0, nb0, lambda_1, P, Sa, Ta, fdis=1.e-6, t_hyd=0))

    # Return the list of particles
    return disp_phases


def plot_oil_gas_DSD(profile, oil, mass_flux, z0, d0, substance, gor, num_oil_elements, num_gas_elements):
    breakup_model = DSD_model(profile, oil, mass_flux, z0,
                              d0, substance, gor, num_oil_elements, num_gas_elements, SSDI=False)
    d_gas, vf_gas, d50_gas, d_liq, vf_liq, d50_liq = breakup_model.simulate()

    breakup_model_ssdi = DSD_model(profile, oil, mass_flux, z0,
                                   d0, substance, gor, num_oil_elements, num_gas_elements, SSDI=True)
    d_gas_ssdi, vf_gas_ssdi, d50_gas_ssdi, d_liq_ssdi, vf_liq_ssdi, d50_liq_ssdi = breakup_model_ssdi.simulate()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    # millimeters
    ax[0, 0].plot([i * 1e3 * 1.15 for i in d_liq], np.cumsum(vf_liq) * 100, label='DWOSM-DSD CDF')
    ax[0, 0].set_xlabel('Diameter (mm)')
    ax[0, 0].set_xscale("log")
    ax[0, 0].set_ylim(0, 100)
    ax[0, 0].set_xlabel('Diameter (mm)', fontsize=15)
    ax[0, 0].set_ylabel('Cumulative volume fraction (%)', fontsize=15)
    ax[0, 0].set_title('Untreated oil droplets', fontsize=15)
    legend0 = ax[0, 0].legend(loc='upper left', fontsize=12)
    legend0.set_frame_on(False)
    legend0.get_frame().set_alpha(0)

    ax[0, 1].plot([i * 1e3 for i in d_gas], np.cumsum(vf_gas) * 100, label='DWOSM-DSD CDF')
    ax[0, 1].set_xticks([i * 3 for i in range(7)])
    ax[0, 1].set_xticklabels([str(i * 3) for i in range(7)])
    ax[0, 1].set_xlim(0.01, 1e2)
    ax[0, 1].set_xscale("log")
    ax[0, 1].set_ylim(0, 100)
    ax[0, 1].set_xlabel('Diameter (mm)', fontsize=15)
    ax[0, 1].set_ylabel('Cumulative volume fraction (%)', fontsize=15)
    ax[0, 1].set_title('Untreated gas bubbles', fontsize=15)
    legend1 = ax[0, 1].legend(loc='upper left', fontsize=12)
    legend1.set_frame_on(False)
    legend1.get_frame().set_alpha(0)

    ax[1, 0].plot([i * 1e3 for i in d_liq_ssdi], np.cumsum(vf_liq_ssdi) * 100, label='DWOSM-DSD CDF')

    ax[1, 0].axhline(50, c='red', linestyle='dashed')
    ax[1, 0].annotate('50', xy=(.055, .277), xycoords='figure fraction', color='r',
                      horizontalalignment='left', verticalalignment='top', fontsize=12)
    obs_x = np.arange(2, 3, 0.2)
    obs_y = np.zeros_like(obs_x)
    obs_y[:] = 50
    ax[1, 0].scatter(obs_x, obs_y, marker='$o$', s=80, c='r', alpha=0.4, label='Observed $d_{50}$')
    ax[1, 0].set_xlim(0.01, 1e2)
    ax[1, 0].set_xscale("log")
    ax[1, 0].set_ylim(0, 100)
    ax[1, 0].set_xlabel('Diameter (mm)', fontsize=15)
    ax[1, 0].set_ylabel('Cumulative volume fraction (%)', fontsize=15)
    ax[1, 0].set_title('Treated oil droplets', fontsize=15)
    # legend2 = ax[1, 0].legend(loc='upper left', fontsize=14)
    legend2 = ax[1, 0].legend(loc='lower right', fontsize=10)

    legend2.set_frame_on(False)
    legend2.get_frame().set_alpha(0)

    ax[1, 1].plot([i * 1e3 for i in d_gas_ssdi], np.cumsum(vf_gas_ssdi) * 100, label='DWOSM-DSD CDF')
    ax[1, 1].axhline(50, c='red', linestyle='dashed')
    ax[1, 1].set_xscale("log")
    ax[1, 1].set_ylim(0, 100)
    ax[1, 1].set_xlabel('Diameter (mm)', fontsize=15)
    ax[1, 1].set_ylabel('Cumulative volume fraction (%)', fontsize=15)
    ax[1, 1].set_title('Treated gas bubbles', fontsize=15)
    legend3 = ax[1, 1].legend(loc='upper left', fontsize=12)
    legend3.set_frame_on(False)
    legend3.get_frame().set_alpha(0)

    for i in range(2):  # Loop over rows
        for j in range(2):  # Loop over columns
            ax[i, j].tick_params(axis='both', labelsize=11)


    notations = ['a', 'b', 'c', 'd']
    for i, ax in enumerate(ax.flat):
        ax.text(0.97, 0.97, notations[i], transform=ax.transAxes,
                fontsize=30, va='top', ha='right')

    plt.tight_layout()
    plt.show()
