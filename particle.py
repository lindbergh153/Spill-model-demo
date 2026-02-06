"""
Fluid Particle Classes
======================

Classes for representing oil droplets, gas bubbles, and fluid mixtures
with thermodynamic property calculations using Peng-Robinson EOS.

Classes
-------
FluidMixture : Multi-component fluid mixture with equilibrium calculations
FluidParticle : Single-phase fluid particle (gas or liquid)

References
----------
Peng, D.-Y., & Robinson, D.B. (1976). A new two-constant equation of state.
McCain, W.D. (1990). The Properties of Petroleum Fluids.

"""

from __future__ import annotations

import numpy as np
from oil_library import get_oil_props

import particle_functions as pf
import seawater
from chemical_data import tamoc_data


class FluidMixture(object):
    """
    Mixture

    Parameters
    ----------
    composition : string list, length nc
        Contains the names of the chemical components in the mixture
        using the same key names as in ./data/ChemData.csv
    delta : ndarray, size (nc, nc)
        Binary interaction coefficients for the Peng-Robinson equation of
        state.  If not passed at instantiation, Python will assume a
        full matrix of zeros.
    user_data : dict
        A dictionary of chemical property data.  If not specified, the data
        loaded from `/tamoc/data/ChemData.csv` by ``chemical_properties`` will
        be used.  To load a different properties database, use the
        ``chemical_properties.load_data`` function to load in a custom
        database, and pass that data to this object as `user_data`.
    sigma_correction : float
        Correction factor to adjust the interfacial tension value supplied by
        the default model to a value measured for the mixture of interest.
        The correction factor should be computed as sigma_measured /
        sigma_model at a single P and T value.  For the FluidParticle class,
        sigma_correction is a scalar applied to the phase defined by fp_type.
    """

    def __init__(self, composition, user_data={}, delta=None,
                 sigma_correction=np.array([[1.], [1.]])):
        if not isinstance(composition, list):
            composition = [composition]

        self.composition = composition
        self.nc = len(composition)
        self.delta = delta
        self.user_data = user_data

        self.chem_db, self.chem_units = tamoc_data()
        self.bio_db = pf.get_dict_k_bio()
        # Initialize the chemical composition variables used in TAMOC
        self.M = np.zeros(self.nc)
        self.Pc = np.zeros(self.nc)
        self.Tc = np.zeros(self.nc)
        self.Vc = np.zeros(self.nc)
        self.Tb = np.zeros(self.nc)
        self.Vb = np.zeros(self.nc)
        self.omega = np.zeros(self.nc)
        self.kh_0 = np.zeros(self.nc)
        self.neg_dH_solR = np.zeros(self.nc)
        self.nu_bar = np.zeros(self.nc)
        self.B = np.zeros(self.nc)
        self.dE = np.zeros(self.nc)
        self.K_salt = np.zeros(self.nc)
        self.k_bio = np.zeros(self.nc)

        # Fill the chemical composition variables from the chem database
        for i in range(self.nc):
            if composition[i] in user_data:
                # Get the properties from the user-specified dataset
                properties = user_data[composition[i]]
            else:
                if composition[i] in self.chem_db:
                    properties = self.chem_db[composition[i]].copy()
                    properties.update(self.bio_db[composition[i]])
                else:
                    print('\nERROR:  %s is not in the ' % composition[i] +
                          'Chemical Properties database\n')

            # Store the chemical properties in the object attributes
            self.M[i] = properties['M']
            self.Pc[i] = properties['Pc']
            self.Tc[i] = properties['Tc']
            self.Vc[i] = properties['Vc']
            self.Tb[i] = properties['Tb']
            if properties['Vb'] < 0.:
                # Use Tyn & Calus estimate in Poling et al. (2001)
                self.Vb[i] = (0.285 * (self.Vc[i] * 1.e6) ** 1.048) * 1.e-6
            else:
                self.Vb[i] = properties['Vb']
            self.omega[i] = properties['omega']
            self.kh_0[i] = properties['kh_0']
            self.neg_dH_solR[i] = properties['-dH_solR']
            if properties['nu_bar'] < 0.:
                # Check if pure compound is a gas
                if self.Tb[i] < 273.15 + 10.:
                    # Use Lyckman formula
                    self.nu_bar[i] = (0.095 + 2.35 * (298.15 * self.Pc[i] /
                                                      (2.2973e9 * self.Tc[i]))) * 8.314510 * self.Tc[i] / self.Pc[i]
                else:
                    # Use empirical equation from Jonas Gros
                    self.nu_bar[i] = (1.148236984 * self.M[i] * 1000 + 6.789136822) / 100. ** 3
            else:
                self.nu_bar[i] = properties['nu_bar']

            if properties['B'] < 0.:
                self.B[i] = 5.0 * 1.e-2 / 100. ** 2.
            else:
                self.B[i] = properties['B']

            if properties['dE'] < 0.:
                self.dE[i] = 4000. / 0.238846
            else:
                self.dE[i] = properties['dE']

            if properties['K_salt'] < 0.:
                self.K_salt[i] = (-1.345 * self.M[i] + 2799.4 * self.nu_bar[i] + 0.083556) / 1000.
            else:
                self.K_salt[i] = properties['K_salt']

            self.k_bio[i] = properties['k_bio']

        if delta is None:
            self.delta = np.zeros((self.nc, self.nc))

        self.delta_groups = np.zeros((self.nc, 15))
        self.Aij = np.zeros((15, 15))
        self.Bij = np.zeros((15, 15))

        # Ideal gas constant
        self.R = 8.314510  # (J/(kg K))

        # Store the interfacial tension correction factor
        self.sigma_correction = sigma_correction
        self.calc_delta = -1

    def masses(self, n):
        """
        Convert the moles of each component in a mixture to their masses (kg).

        Parameters
        ----------
        n : ndarray, size (nc)
            moles of each component in a mixture (--)

        Returns
        -------
        m : ndarray, size (nc)
            masses of each component in a mixture (kg)

        """
        m = n * self.M
        return m

    def mass_frac(self, n):
        """
        Calculate the mass fraction (--) from the number of moles of each
        component in a mixture.

        Parameters
        ----------
        n : ndarray, size (nc)
            moles of each component in a mixture (--)

        Returns
        -------
        mf : ndarray, size (nc)
            mass fractions of each component in a mixture (--)

        """
        m = self.masses(n)
        mf = m / np.sum(m)

        return mf

    def moles(self, m):
        """
        Convert the masses of each component in a mixture to their moles
        (mol).

        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a mixture (kg)

        Returns
        -------
        n : ndarray, size (nc)
            moles of each component in a mixture (--)
        """
        n = m / self.M

        return n

    def mol_frac(self, m):
        """
        Calculate the mole fraction (--) from the masses of each component in a mixture.

        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a mixture (kg)

        Returns
        -------
        yk : ndarray, size (nc)
            mole fraction of each component in a mixture (--)
        """
        n_moles = m / self.M
        yk = n_moles / sum(n_moles)

        return yk

    def partial_pressures(self, m, P):
        """
        Compute the partial pressure (Pa) of each component in a mixture.

        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a mixture (kg)
        P : float
            mixture pressure (Pa)

        Returns
        -------
        Pk : ndarray, size (nc)
            partial pressures of each component in a mixture (Pa)

        """
        yk = self.mol_frac(m)
        Pk = P * yk

        return Pk

    def density(self, m, T, P):
        """
        Compute the gas and liquid density (kg/m^3) of a fluid mixture at the given state.

        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a mixture (kg)
        T : float
            mixture temperature (K)
        P : float
            mixture pressure (Pa)

        Returns
        -------
        rho_p : ndarray, size (2,1)
            density of the gas phase (row 1) and liquid phase (row 2) of a
            fluid mixture (kg/m^3)
        """
        # Compute and return the density
        rho = pf.get_density(T, P, m, self.M, self.Pc, self.Tc, self.Vc, self.omega, self.delta)

        return rho

    def fugacity(self, m, T, P):
        """
        Compute the gas and liquid fugacity (Pa) of a fluid mixture at the
        given state.

        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a mixture (kg)
        T : float
            mixture temperature (K)
        P : float
            mixture pressure (Pa)

        Returns
        -------
        fk : ndarray, size (2, nc)
            fugacity coefficients of the gas phase (row 1) and liquid phase
            (row 2) for each component in a mixture (Pa)

        """
        fug = pf.get_fugacity(T, P, m, self.M, self.Pc, self.Tc, self.omega, self.delta)

        return fug

    def viscosity(self, m, T, P):
        """
        Computes the dynamic viscosity of the gas/liquid mixture.

        Computes the dynamic viscosity of gas and liquid using correlation
        equations in Pedersen et al. (2014) "Phase Behavior of Petroleum
        Reservoir Fluids", 2nd edition, chapter 10.  This function has been
        tested for non-hydrocarbon mixtures (oxygen, carbon dioxide) and
        shown to give reasonable results; hence, the same equations are used
        for all mixtures.

        Parameters size (nc)
            masses of each component in a mixture (kg)
        T : float
            mixture temperature (K)
        P : float
            mixture pressure (Pa)

        Returns
        -------
        mu_p : ndarray, size (2)
            dynamic viscosity for gas (row 1) and liquid (row 2) (Pa s)

        """
        mu = pf.get_viscosity(T, P, m, self.M, self.Pc, self.Tc)

        return mu

    def interface_tension(self, m, T, S, P):
        """
        Computes the interfacial tension between gas/liquid and water

        If `air` is False (thus, assume hydrocarbon), this method computes
        the interfacial tension between the gas and liquid phases of the
        mixture and water using equations in Danesh (1998) "PVT and Phase
        Behaviour Of Petroleum Reservoir Fluids," Chapter 8.  Otherwise, we
        treat the fluid like air and use the surface tension of seawater from
        Sharqawy et al. (2010), Table 6.

        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a mixture (kg)
        T : float
            mixture temperature (K)
        P : float
            mixture pressure (Pa)

        Returns
        -------
        sigma_p : ndarray, size (2)
            interfacial tension for gas (row 1) and liquid (row 2) (N/m)

        """
        # Compute the local density of water
        rho_w = seawater.density(T, S, P)

        # Compute the density of the mixture phases
        rho_p = FluidMixture.density(self, m, T, P)

        # Get the density difference in g/cm^3
        delta_rho = (rho_w - rho_p) / 1000.

        # Compute the pseudo critical temperature using mole fractions as weights
        xi = self.mol_frac(m)
        Tc = np.sum(self.Tc * xi)
        # print(rho_w, rho_p, T, Tc)
        # Get the interfacial tension
        sigma = 0.111 * delta_rho ** 1.024 * (T / Tc) ** -1.25

        # Adjust the interfacial tension to a measured value
        sigma = self.sigma_correction * sigma

        # Return the Interfacial tension
        return sigma

    def equilibrium(self, m, T, P):
        """
        Computes the equilibrium composition of a gas/liquid mixture.

        Computes the equilibrium composition of a gas/liquid mixture using the
        procedure in Michelson and Mollerup (2007) and McCain (1990).

        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a mixture (kg)
        T : float
            mixture temperature (K)
        P : float
            mixture pressure (Pa)

        Returns
        -------
        A tuple containing:

            m : ndarray, size(2, nc)
                masses of each component in a mixture at equilibrium between
                the gas and liquid phases (kg)
            xi : ndarray, size(2, nc)
                mole fractions of each component in the mixture at
                equilibrium between the gas and liquid phases (--)
            K : ndarray, size(nc)
                partition coefficients expressed as K-factor (--)

        Notes
        -----
        Uses the function equil_MM which uses the Michelsen and Mollerup
        (2007) procedure to find a stable solution

        """
        # Get the mole fractions and K-factors at equilibrium
        xi, beta, K = method_MM(m, T, P, self.M, self.Pc, self.Tc, self.omega, self.delta)

        # Get the total moles of each molecule (both phases together)
        n_tot = self.moles(m)

        # Get the total number of moles in gas phase using the first
        # non-zero component in the mixture (note that this is independent of
        # which component you pick):
        idx = 0
        while m[idx] <= 0.:
            idx += 1
        ng = np.abs((n_tot[idx] - (xi[1, idx] * np.sum(n_tot))) /
                    (xi[0, idx] - xi[1, idx]))

        # Get the moles of each component in gas (line 1) and liquid (line 2) phase
        n = np.zeros((2, self.nc))
        n[0, :] = xi[0, :] * ng
        n[1, :] = xi[1, :] * (np.sum(n_tot) - ng)

        # Finally converts to mass
        m = np.zeros((2, self.nc))
        for i in range(2):
            m[i, :] = self.masses(n[i, :])

        return m, xi, K

    def solubility(self, m, T, P, Sa):
        """
        Compute the solubility (kg/m^3) of each component of a mixture in both
        gas and liquid dissolving into seawater.

        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a mixture (kg)
        T : float
            mixture temperature (K)
        P : float
            mixture pressure (Pa)
        Sa : float
            salinity of the ambient seawater (psu)

        Returns
        -------
        Cs : ndarray, size (2, nc)
            solubility of the gas phase (row 1) and liquid phase (row 2) for
            each component in a mixture (kg/m^3)

        """

        # Compute the Henry's law coefficients using the temperature of the seawater
        kh = pf.kh_insitu(T, P, Sa, self.kh_0, self.neg_dH_solR, self.nu_bar, self.K_salt)

        # Compute the mixture fugacity using the temperature of the mixture
        f = FluidMixture.fugacity(self, m, T, P)
        Cs = f * kh

        return Cs

    def diffusivity(self, Ta, Sa, P):
        """
        Compute the diffusivity (m^2/s) of each component of a mixture into
        seawater at the given temperature.

        Parameters
        ----------
        Ta : float
            temperature of ambient seawater (K)
        Sa : float
            salinity of ambient seawater (psu)
        P : float
            pressure of ambient seawater (Pa)

        Returns
        -------
        D : ndarray, size (nc)
            diffusion coefficients for each component of a mixture into
            seawater (m^2/s)
        """
        # Compute the viscosity of seawater
        mu = seawater.mu(Ta, Sa, P)
        D = pf.get_diffusivity(mu, self.Vb)

        # Return the diffusivities
        return D


def get_VP(oil_id, composition, user_data, chem_db, temp):
    temp_trail = 273.15 + 15
    VP = get_oil_props(oil_id).vapor_pressure(temp_trail)
    if len(VP) == len(composition):
        VP = get_oil_props(oil_id).vapor_pressure(temp)
    else:
        gas = []
        count = 0
        for i in user_data:
            if composition[count] not in user_data:
                gas.append(composition[count])
            count += 1
        Tb = []
        for j in gas:
            Tb.append(chem_db[j]['Tb'])
        VP_gas = []
        for i in Tb:
            VP_gas.append(vapor_pressure(i, temp))
        VP_gas = np.array(VP_gas)
        VP_oil = get_oil_props(oil_id).vapor_pressure(temp)
        VP = np.concatenate((VP_gas, VP_oil), axis=0)

    return VP


def vapor_pressure(boiling_point, temp, atmos_pressure=101325.0):
    D_Zb = 0.97
    R_cal = 1.987  # calories

    D_S = 8.75 + 1.987 * np.log(boiling_point)
    C_2i = 0.19 * boiling_point - 18

    var = 1. / (boiling_point - C_2i) - 1. / (temp - C_2i)
    ln_Pi_Po = (D_S * (boiling_point - C_2i) ** 2 /
                (D_Zb * R_cal * boiling_point) * var)
    Pi = np.exp(ln_Pi_Po) * atmos_pressure

    return Pi


class FluidParticle(FluidMixture):
    def __init__(self, composition, fp_type=0, oil_id=None, delta=None, user_data={},
                 sigma_correction=1):
        super(FluidParticle, self).__init__(composition, user_data, delta, sigma_correction)

        # Store the input variables
        self.fp_type = int(fp_type)
        self.oil_id = oil_id
        self.user_data = user_data

        if fp_type == 1:
            self.VP = self.VP_oil_gas

    def VP_oil_gas(self, temp):
        return get_VP(self.oil_id, self.composition, self.user_data, self.chem_db, temp)

    def density(self, m, T, P):
        rho = FluidMixture.density(self, m, T, P)[self.fp_type, 0]

        return rho

    def fugacity(self, m, T, P):
        fug = FluidMixture.fugacity(self, m, T, P)[self.fp_type, :]

        return fug

    def viscosity(self, m, T, P):
        mu = FluidMixture.viscosity(self, m, T, P)[self.fp_type, 0]

        return mu

    def interface_tension(self, m, T, S, P):
        sigma = FluidMixture.interface_tension(self, m, T, S, P)[self.fp_type, 0]

        return sigma

    def solubility(self, m, T, P, Sa):
        Cs = FluidMixture.solubility(self, m, T, P, Sa)[self.fp_type, :]

        return Cs

    def masses_by_diameter(self, de, T, P, yk):
        """
        Find the masses (kg) of each component in a particle with equivalent
        spherical diameter `de` and mole fractions `yk`.

        Parameters
        ----------
        de : float
            equivalent spherical diameter (m)
        T : float
            particle temperature (K)
        P : float
            particle pressure (Pa)
        yk : ndarray, size (nc)
            mole fractions of each component in the particle (--)

        Returns
        -------
        m : ndarray, size (nc)
            masses of each component in a fluid particle (kg)

        """

        # Get the masses for one mole of fluid
        m = self.masses(yk)

        # Compute the actual total mass of the fluid particle using the
        # density computed from one mole of fluid
        m_tot = 1 / 6 * np.pi * de ** 3 * self.density(m, T, P)

        # Determine the number of moles in the fluid particle
        n = yk * m_tot / np.sum(m)

        return self.masses(n)

    def diameter(self, m, T, P):
        """
        Compute the equivalent spherical diameter (m) of a single fluid particle.

        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a particle (kg)
        T : float
            particle temperature (K)
        P : float
            particle pressure (Pa)

        Returns
        -------
        de : float
            equivalent spherical diameter of a fluid particle (m)

        """
        de = (6 * np.sum(m) / (np.pi * self.density(m, T, P))) ** (1 / 3)
        de = float(de)

        return de

    def particle_shape(self, m, T, P, Sa, Ta):
        """
        Determine the shape of a fluid particle from the properties of the
        particle and surrounding fluid.

        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a particle (kg)
        T : float
            particle temperature (K)
        P : float
            particle pressure (Pa)
        Sa : float
            salinity of ambient seawater (psu)
        Ta : float
            temperature of ambient seawater (K)

        Returns
        -------
        A tuple containing:

            shape : integer
                1 - sphere, 2 - ellipsoid, 3 - spherical cap
            de : float
                equivalent spherical diameter (m)
            rho_p : float
                particle density (kg/m^3)
            rho : float
                ambient seawater density (kg/m^3)
            mu : float
                ambient seawater dynamic viscosity (Pa s)
            mu_p : float
                dispersed phase dynamic viscosity (Pa s)
            sigma : float
                interfacial tension (N/m)

        Notes
        -----
        As for the solubility calculation, we use the particle temperature to
        calculate properties at the interface (e.g., to calculate the
        interfacial tension) and the ambient temperature for properties of
        the bulk continuous phase (e.g., density and viscosity).

        Uses the Fortran subroutines in ``./src/dbm_phys.f95``.

        """
        # Compute the fluid particle and ambient properties
        de = self.diameter(m, T, P)
        rho_p = self.density(m, T, P)
        rho = seawater.density(Ta, Sa, P)
        mu = seawater.mu(Ta, Sa, P)
        mu_p = self.viscosity(m, T, P)
        sigma = self.interface_tension(m, T, Sa, P)
        shape = pf.get_particle_shape(de, rho_p, rho, mu, sigma)

        return shape, de, rho_p, rho, mu_p, mu, sigma

    def slip_velocity(self, m, T, P, Sa, Ta, status=-1):
        """
        Compute the slip velocity (m/s) of a fluid particle.

        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a particle (kg)
        T : float
            particle temperature (K)
        P : float
            particle pressure (Pa)
        Sa : float
            salinity of ambient seawater (psu)
        Ta : float
            temperature of ambient seawater (K)
        status : int
            flag indicating whether the particle is clean (status = 1) or
            dirty (status = -1).  Default value is -1.

        Returns
        -------
        us : float
            slip velocity of the fluid particle (m/s)

        """
        # Get the particle properties
        shape, de, rho_p, rho, mu_p, mu, sigma = self.particle_shape(m, T, P, Sa, Ta)

        if shape == 1:
            us = pf.us_sphere(de, rho_p, rho, mu)
        elif shape == 2:
            us = pf.us_ellipsoid(de, rho_p, rho, mu_p, mu, sigma, status)
        else:
            us = pf.us_spherical_cap(de, rho_p, rho)

        return us

    def surface_area(self, m, T, P, Sa, Ta):
        """
        Compute the surface area (m^2) of a fluid particle.

        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a particle (kg)
        T : float
            particle temperature (K)
        P : float
            particle pressure (Pa)
        Sa : float
            salinity of ambient seawater (psu)
        Ta : float
            temperature of ambient seawater (K)

        Returns
        -------
        A : float
            surface area of the fluid particle (m^2)
        """
        # Get the particle properties
        shape, de, rho_p, rho, mu_p, mu, sigma = self.particle_shape(m, T, P, Sa, Ta)

        if shape == 3:
            # Compute the surface area of a spherical cap bubble
            us = self.slip_velocity(m, T, P, Sa, Ta)
            theta_w = pf.theta_w_sc(de, us, rho, mu)
            A = pf.surface_area_sc(de, theta_w)
        else:
            # Compute the area of the equivalent sphere:
            A = np.pi * de ** 2

        return A

    def mass_transfer(self, m, T, P, Sa, Ta, status=-1):
        """
        Compute the mass transfer coefficients (m/s) for each component in a
        fluid particle

        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a particle (kg)
        T : float
            particle temperature (K)
        P : float
            particle pressure (Pa)
        Sa : float
            salinity of ambient seawater (psu)
        Ta : float
            temperature of ambient seawater (K)
        status : int
            flag indicating whether the particle is clean (status = 1) or
            dirty (status = -1).  Default value is -1.

        Returns
        -------
        beta : ndarray, size (nc)
            mass transfer coefficient for each component in a fluid particle
            (m/s)
        """
        # Get the particle properties
        shape, de, rho_p, rho, mu_p, mu, sigma = self.particle_shape(m, T, P, Sa, Ta)

        # Compute the slip velocity
        us = self.slip_velocity(m, T, P, Sa, Ta, status)

        # Get the diffusivities
        D = self.diffusivity(Ta, Sa, P)

        # Compute the appropriate mass transfer coefficients
        if shape == 1:
            beta = pf.xfer_sphere(de, us, rho, mu, D, sigma, mu_p, self.fp_type, status)
        elif shape == 2:
            beta = pf.xfer_ellipsoid(de, us, rho, mu, D, sigma, mu_p, self.fp_type, status)
        else:
            beta = pf.xfer_spherical_cap(de, us, rho, rho_p, mu, D, status)

        return beta

    def heat_transfer(self, m, T, P, Sa, Ta, status=-1):
        """
        Compute the heat transfer coefficients (m/s) for an inert fluid
        particle.

        Parameters
        ----------
        m : float
            mass of the inert fluid particle (kg)
        T : float
            particle temperature (K)
        P : float
            particle pressure (Pa)
        Sa : float
            salinity of ambient seawater (psu)
        Ta : float
            temperature of ambient seawater (K)
        status : int
            flag indicating whether the particle is clean (status = 1) or
            dirty (status = -1).  Default value is -1.

        Returns
        -------
        beta_T : float
            heat transfer coefficient for the inert particle (m/s)

        Notes
        -----
        Uses the Fortran subroutines in ``./src/dbm_phys.f95``.

       """
        # Get the particle properties
        shape, de, rho_p, rho, mu_p, mu, sigma = self.particle_shape(m, T, P, Sa, Ta)

        # Get the thermal conductivity of seawater
        k = seawater.k(Ta, Sa, P) / (seawater.density(Ta, Sa, P) * seawater.cp())

        # Compute the slip velocity
        us = self.slip_velocity(m, T, P, Sa, Ta, status)

        # Compute the appropriate heat transfer coefficients.  Assume the
        # heat transfer has the same form as the mass transfer with the
        # diffusivity replaced by the thermal conductivity
        if shape == 1 or shape == 4:
            beta = pf.xfer_sphere(de, us, rho, mu, k, sigma, mu_p, self.fp_type, status)
        elif shape == 2:
            beta = pf.xfer_ellipsoid(de, us, rho, mu, k, sigma, mu_p, self.fp_type, status)
        else:
            beta = pf.xfer_spherical_cap(de, us, rho, rho_p, mu, k, status)

        return beta

    def return_all(self, m, T, P, Sa, Ta, status=-1):
        """
        Compute all of the dynamic properties of the bubble in an efficient
        manner (e.g., minimizing replicate calls to functions).

        This method repeats the calculations in the individual property
        methods, and does not call the methods already defined.  This is done
        so that multiple calls to functions (e.g., slip velocity) do not
        occur.

        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a particle (kg)
        T : float
            particle temperature (K)
        P : float
            particle pressure (Pa)
        Sa : float
            salinity of ambient seawater (psu)
        Ta : float
            temperature of ambient seawater (K)
        status : int
            flag indicating whether the particle is clean (status = 1) or
            dirty (status = -1).  Default value is -1.

        Returns
        -------
        A tuple containing:

            shape : integer
                1 - sphere, 2 - ellipsoid, 3 - spherical cap
            de : float
                equivalent spherical diameter (m)
            rho_p : float
                particle density (kg/m^3)
            us : float
                slip velocity (m/s)
            A : float
                surface area (m^2)
            Cs : ndarray, size (nc)
                solubility (kg/m^3)
            beta : ndarray, size (nc)
                mass transfer coefficient (m/s)
            beta_T : float
                heat transfer coefficient (m/s)
        """
        # Particle density, equivalent diameter and shape
        shape, de, rho_p, rho, mu_p, mu, sigma = self.particle_shape(m, T, P, Sa, Ta)

        # Solubility
        Cs = self.solubility(m, T, P, Sa)

        # Shape-specific properties
        us = self.slip_velocity(m, T, P, Sa, Ta)
        A = self.surface_area(m, T, P, Sa, Ta)
        beta = self.mass_transfer(m, T, P, Sa, Ta, status)
        beta_T = self.heat_transfer(m, T, P, Sa, Ta, status)

        return shape, de, rho_p, us, A, Cs, beta, beta_T


def method_MM(m, T, P, M, Pc, Tc, omega, delta):
    # Compute the some constant properties of the mixture
    moles = m / M
    zi = moles / np.sum(moles)
    f_zi = pf.get_fugacity(T, P, zi * M, M, Pc, Tc, omega, delta)[0, :]
    phi_zi = f_zi / (zi * P)
    di = np.log(zi) + np.log(phi_zi)

    def gibbs_energy(K):
        """
        Compute the Gibbs energy difference between the feed and the current
        composition given by K using equation (41) on page 266

        """
        # Use the current K to compute the equilibrium
        xi, beta = pf.gas_liq_eq(m, M, K)

        # Compute the fugacities of the new composition
        f_gas = pf.get_fugacity(T, P, xi[0, :] * M, M, Pc, Tc, omega, delta)[0, :]
        f_liq = pf.get_fugacity(T, P, xi[1, :] * M, M, Pc, Tc, omega, delta)[1, :]

        # Get the fugacity coefficients
        phi_gas = f_gas / (xi[0, :] * P)
        phi_liq = f_liq / (xi[1, :] * P)

        # Compute the reduced tangent plane distances
        tpdx = np.nansum(xi[1, :] * (np.log(xi[1, :]) + np.log(phi_liq) - di))
        tpdy = np.nansum(xi[0, :] * (np.log(xi[0, :]) + np.log(phi_gas) - di))

        # Compute the change in the total Gibbs energy between the feed
        # and this present composition
        DG_RT = (1. - beta) * tpdx + beta * tpdy

        # Return the results
        return DG_RT, tpdx, tpdy, phi_liq, phi_gas

    # Get an initial estimate for the K-factors
    # Use equation (26) on page 259 of Michelson and Mollerup (2007)
    K = np.exp(5.37 * (1. + omega) * (1 - Tc / T)) / (P / Pc)

    # Follow the procedure on page 266ff of Michelsen and Mollerup (2007).
    # Start with three iterations of successive substitution
    K, beta, xi, exit_flag, steps = pf.successive_substitution(m, T, P, 3, M, Pc, Tc, omega, delta, K)

    # Continue iterating if necessary until the solution converges
    while exit_flag <= 0:
        # Test the total Gibbs energy to decide how to proceed.
        Delta_G_RT, tpdx, tpdy, phi_liq, phi_gas = gibbs_energy(K)

        if exit_flag == 0:

            if Delta_G_RT < 0.:
                # The current composition is converging on a lower total Gibbs
                # energy than the feed: continue successive substitution
                phases = 2

            elif tpdy < 0.:
                # The feed is unstable, but we need a better estimate of K
                K = phi_zi / phi_gas
                phases = 2

            elif tpdx < 0.:
                # The feed is unstable, but we need a better estimate of K
                K = phi_liq / phi_zi
                phases = 2

            else:
                # We are not sure of the stability of the feed:  do stability analysis.
                K, phases = pf.stability_analysis(m, T, P, M, Pc, Tc, omega, delta, K, zi, di)
        else:
            # Successive substitution thinks this is single-phase...check
            # with stability analysis
            K_st, phases = pf.stability_analysis(m, T, P, M, Pc, Tc, omega, delta, K, zi, di)

            # Keep the K-values with the lowest Gibbs energy
            Delta_G_RT_st, tpdx_st, tpdy_st, phi_liq_st, phi_gas_st = \
                gibbs_energy(K)
            if Delta_G_RT_st < Delta_G_RT:
                K = K_st

        if phases > 1:
            # The mixture is unstable and unconverged, continue with successive substitution
            K, beta, xi, exit_flag, steps = \
                pf.successive_substitution(m, T, P, np.inf, M, Pc, Tc, omega, delta, K, steps)
        else:
            # The mixture is single-phase and converged
            exit_flag = 1
            xi = np.zeros((2, len(zi)))
            if beta > 0.5:
                # Pure gas
                beta = 1.
                xi[0, :] = zi
                K = np.zeros(K.shape) + np.nan
            else:
                # Pure liquid
                beta = 0.
                xi[1, :] = zi
                K = np.zeros(K.shape) + np.nan

    # Return the optimized mixture composition
    return xi, beta, K

