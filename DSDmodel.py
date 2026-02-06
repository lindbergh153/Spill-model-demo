"""
Droplet Size Distribution (DSD) Model
=====================================

Predicts bubble and droplet size distributions at deep water releases
using SINTEF and Wang et al. models with Rosin-Rammler distributions.

References
----------
Wang, B., et al. (2018). Gas bubble breakup model for subsea blowouts.
Johansen, Ø. (2003). Development of deep-water blowout models.

"""

from __future__ import annotations
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

import seawater
from particle import FluidParticle, FluidMixture


class DSD_model(object):
    def __init__(self, profile1d, oil, mass_flux, z0, d_pipe, substance,
                 gor, bins_oil, bins_gas, SSDI):
        self.profile = profile1d
        self.P, self.Tj, self.S, Ua, Va, Wa = profile1d.get_values(z0)
        self.oil = oil
        self.mass_flux = mass_flux
        self.rho_c = seawater.density(self.Tj, self.S, self.P)
        self.mu_c = seawater.mu(self.Tj, self.S, self.P)
        self.d0 = d_pipe
        self.nbins_oil = bins_oil
        self.nbins_gas = bins_gas
        self.gor = gor

        # Compute the gas/liquid equilibrium
        if self.gor > 0:
            # Compute the equilibrium mixture properties at the release
            m_eq, xi, K = self.oil.equilibrium(self.mass_flux, self.Tj, self.P)
        else:
            m_eq = np.zeros((2, len(self.mass_flux)))
            m_eq[1, :] = self.mass_flux

        # Compute the gas phase properties
        if np.sum(m_eq[0, :]) == 0:
            self.gas = None
            self.m_gas = m_eq[0, :]
            self.rho_gas = None
            self.mu_gas = None
            self.sigma_gas = None
        else:
            self.gas = FluidParticle(self.oil.composition, fp_type=0, delta=oil.delta,
                                     user_data=oil.user_data)
            self.m_gas = m_eq[0, :]
            self.rho_gas = self.gas.density(self.m_gas, self.Tj, self.P)
            self.mu_gas = self.gas.viscosity(self.m_gas, self.Tj, self.P)
            self.sigma_gas = self.gas.interface_tension(self.m_gas, self.Tj, self.S, self.P)
            if SSDI:
                self.sigma_gas = self.sigma_gas / 5.4  # 5.4 8 10
                # if self.sigma_gas < 40 / 1000:
                #     self.sigma_gas = 40 / 1000
        # Compute the liquid phase properties
        if np.sum(m_eq[1, :]) == 0:
            self.oil = None
            self.m_oil = m_eq[1, :]
            self.rho_oil = None
            self.mu_oil = None
            self.sigma_oil = None
        else:
            self.oil = FluidParticle(self.oil.composition, fp_type=1, delta=oil.delta,
                                     user_data=oil.user_data, oil_id=substance)
            self.m_oil = m_eq[1, :]
            self.rho_oil = self.oil.density(self.m_oil, self.Tj, self.P)
            self.mu_oil = self.oil.viscosity(self.m_oil, self.Tj, self.P)
            self.sigma_oil = self.oil.interface_tension(self.m_oil, self.Tj, self.S, self.P)
            if SSDI:
                self.sigma_oil = self.sigma_oil / 5.4  #5.4 8 10 100

    def simulate(self):
        self.d50_oil, self.de_max_oil, self.k_oil, self.alpha_oil = \
            sintef(self.d0, self.m_gas, self.rho_gas, self.m_oil,
                   self.rho_oil, self.mu_oil, self.sigma_oil, self.rho_c, self.mu_c)
        self.de_oil, self.vf_oil = rosin_rammler(self.nbins_oil, self.d50_oil, self.k_oil, self.alpha_oil)
        if self.gor > 0:
            self.d50_gas, m_gas, m_oil, self.de_max_gas, self.sigma_ln_gas = \
                wang(self.d0, self.m_gas, self.rho_gas, self.mu_gas,
                     self.sigma_gas, self.rho_c, self.mu_c, m_l=self.m_oil,
                     rho_l=self.rho_oil, P=self.P, T=self.Tj)
            # Convert lognormal parameters to Rosin-Rammler
            self.d50_gas, self.k_gas, self.alpha_gas = ln2rr(self.d50_gas, self.sigma_ln_gas)
            self.de_gas, self.vf_gas = rosin_rammler(self.nbins_gas, self.d50_gas, self.k_gas, self.alpha_gas)
        else:
            self.de_gas = np.zeros_like(self.de_oil)
            self.vf_gas = np.zeros_like(self.vf_oil)

        return self.de_gas, self.vf_gas, self.d50_gas, self.de_oil, self.vf_oil, self.d50_oil

    def plot(self, unit: int = 0):
        fig, ax = plt.subplots(figsize=(7, 5))

        if unit == 0:
            # micrometers
            ax.plot([i * 1e6 for i in self.de_oil], np.cumsum(self.vf_oil) * 100, label='DWOSM-DSD CDF')
            ax.set_xlabel('Diameter (μm)')
        elif unit == 1:
            # millimeters
            ax.plot([i * 1e3 for i in self.de_oil], np.cumsum(self.vf_oil) * 100, label='DWOSM-DSD CDF')
            x_vdrop = [1.0142, 2.0047, 2.9481, 4.0016, 4.945, 5.8884, 6.8396, 7.9481, 9.00, 10.0]
            y_vdrop = [0.00607, 0.08462, 0.26671, 0.54497, 0.75221, 0.88401, 0.98622, 0.99822, 0.99985, 1]
            x_vdrop, y_vdrop = np.array(x_vdrop), np.array(y_vdrop) * 100
            ax.plot(x_vdrop, y_vdrop, linestyle='dashed', c='y', label='Vdrop-J CDF')
            ax.scatter(self.d50_oil * 1e3, 50, marker='v', s=80, label='DWOSM-DSD d50')
            ax.scatter(3.88, 50, marker='v', s=80, c='y', label='Vdrop-J d50')
            ax.scatter(2.7, 50, marker='v', s=80, c='g', label='''Spaulding's d50''')
            ax.scatter(3.0, 50, marker='v', s=80, c='g')
            ax.set_xlabel('Diameter (mm)')

            # 2.7 - 3.0 mm
            ax.axhline(50, c='red', linestyle='dashed')
        elif unit == 2:
            # meters
            ax.plot(self.de_oil, np.cumsum(self.vf_oil) * 100, label='DWOSM-DSD')
            ax.set_xlabel('Diameter (m)')
        else:
            raise Exception('wrong input of unit in DSDmodel.plot')

        ax.annotate('50', xy=(.09, .511), xycoords='figure fraction', color='r',
                    horizontalalignment='left', verticalalignment='top', fontsize=12)
        ax.set_xlim(0, )
        ax.set_ylim(0, 100)
        ax.set_ylabel('Volume fraction')
        plt.legend()

        plt.show()


def wang(d0, m_g, rho_g, mu_g, sigma_g, rho, mu, m_l, rho_l, P, T):
    """
    Compute characteristic values for gas jet breakup

    Computes the characteristic gas bubble sizes for jet breakup using the
    equations in Wang et al. (2018) (wang_etal model equations).

    Parameters
    ----------
    d0 : float
        Equivalent circular diameter of the release (m)
    m_g : np.array
        Mass fluxes of each pseudo-component of the gas-phase fluid at the
        release (kg/s)
    rho_g : float
        Density of the gas-phase  fluid at the release (kg/m^3)
    mu_g : float
        Dynamic viscosity of the gas-phase fluid at the release (Pa s)
    sigma_g : float
        Interfacial tension between the gas-phase fluid and water at the
        release (N/m)
    rho : float
        Density of seawater at the release (kg/m^3)
    mu : float
        Dynamic viscosity of seawater at the release (Pa s)
    m_l : np.array
        Mass fluxes of each pseudo-component of the liquid-phase fluid at the
        release (kg/s)
    rho_l : float
        Density of the liquid-phase fluid at the release (kg/m^3)
    P : float, default=4.e6
        Pressure in the receiving fluid (Pa); used to compute the speed of
        sound in the released gas.
    T : float, default=288.15
        Temperature of the gas phase at the release (K); used to compute the
        speed of sound in the released gas.

    Returns
    -------
    d50_gas : float
        Volume median diameter of the gas bubbles (m)
    m_gas : float
        Mass fluxes of each pseudo-component of the gas-phase fluid at the
        release (kg/s).  This may be different from the input value in the
        case of choked flow at the orifice.
    m_oil : float
        Mass fluxes of each pseudo-component of the liquid-phase fluid at the
        release (kg/s).  This may be different from the input value in the
        case of choked flow at the orifice.
    de_max : float
        Maximum stable particle size of the fluid phase of interest (m)
    sigma : float
        Standard deviation of the Log-normal distribution in logarithmic
        units.

    """
    # Convert mass-flux to volume flux
    Qg = mass2vol(m_g, rho_g)
    if np.sum(m_l) == 0.:
        Ql = 0.
    else:
        Ql = mass2vol(m_l, rho_l)

    # Compute the exit velocity assuming no choked flow and single exit
    # velocity
    n = Qg / (Qg + Ql)
    A = np.pi * d0 ** 2 / 4.
    Ug = (Qg + Ql) / A
    # Check for choked flow using methane for speed of sound
    ch4 = FluidMixture(['methane'])
    delta_rho = ch4.density(np.array([1.]), T, P)[0, 0] - ch4.density(np.array([1.]), T, 1.01 * P)[0, 0]
    a = np.sqrt((P - 1.01 * P) / delta_rho)

    if 10. * Ug < a:
        U_E = Ug
    else:
        # Compute the cp / cv ratio
        cp_ch4 = 35.69  # J/mol/K;  CO2 = 37.13
        cv_ch4 = cp_ch4 - 8.31451  # From Poling et al. for ideal gases
        kappa = cp_ch4 / cv_ch4  # Assume approximately ok for petroleum

        # Get the Mach number
        Ma = Ug / a

        # Correct the exit velocity for choked flow
        if Ma < np.sqrt((kappa + 1.) / 2.):
            U_E = a * (-1. + np.sqrt(1. + 2. * (kappa - 1.) * Ma ** 2.)) / \
                  ((kappa - 1.) * Ma)
        else:
            U_E = a * np.sqrt(2. / (kappa + 1.))

    # Update the gas and oil exit velocities
    if Qg > 0:
        Ug = U_E
    else:
        Ug = 0
    if Ql > 0:
        Ul = U_E
    else:
        Ul = 0

    # Compute the particle size distribution for gas
    d50_gas, m_gas, m_oil, de_max, sigma = wang_model(A, n, Ug, rho_g,
                                                      mu_g, sigma_g, Ul,
                                                      rho_l, rho, mu)

    return d50_gas, m_gas, m_oil, de_max, sigma


def wang_model(A, n, Ug, rho_g, mu_g, sigma_g, Ul, rho_l, rho, mu):
    """
    Computes the particle size for the Wang et al. equation

    Evaluates the parameters of the gas bubble size distribution for the Wang
    et al. equation and implements the d_95 rule. This function returns the
    parameters of the log-normal size distribution with the spreading
    rates as reported in Wang et al.

    Returns
    -------
    d50_gas : float
        Volume median diameter of the gas bubbles (m)
    m_gas : float
        Mass fluxes of each pseudo-component of the gas-phase fluid at the
        release (kg/s).  This may be different from the input value in the
        case of choked flow at the orifice.
    m_oil : float
        Mass fluxes of each pseudo-component of the liquid-phase fluid at the
        release (kg/s).  This may be different from the input value in the
        case of choked flow at the orifice.
    de_max : float
        Maximum stable particle size of the fluid phase of interest (m)
    sigma_ln : float
        Standard deviation of the Log-normal distribution in logarithmic
        units.

    Notes
    -----
    This function is called by the `wang_etal()` function after several
    intermediate parameters are computed.  This function should not be
    called directly.

    """
    if Ug > 0:

        # Compute d50 from the model
        (d, m_gas, m_oil) = wang_etal_d50(A, n, Ug, rho_g, mu_g, sigma_g, Ul,
                                          rho_l, rho, mu)

        # Compute the maximum stable bubble size
        de_max = grace(rho, rho_g, mu, mu_g, sigma_g, fp_type=0)
        print('\nMax stable bubble size:  %g (m)' % de_max)

        # Get the adjusted particle size distribution
        print('    ---> Predicted median bubble size = %g (m)' % d)
        d50_gas, sigma_ln = log_normal_fit(d, de_max, sigma=0.27)

    else:

        # Return an empty set of particles
        m_gas = 0.
        m_oil = rho_l * A * Ul
        de_max = None
        d50_gas, sigma_ln = log_normal_fit(0., de_max)

    return d50_gas, m_gas, m_oil, de_max, sigma_ln


def wang_etal_d50(A, n, Ug, rho_g, mu_g, sigma_g, Ul, rho_l, rho, mu):
    """
    Compute d_50 from the Wang et al. equations

    Returns
    -------
    d50 : float
        Volume median diameter of the gas bubbles (m)

    Notes
    -----
    This function is called by the `wang_etal()` function after several
    intermediate parameters are computed.  This function should not be
    called directly.

    """
    # Compute the total dynamic momentum and buoyancy fluxes
    Ag = A * n
    Al = A * (1. - n)
    mg = rho_g * Ag * Ug ** 2
    bg = (rho - rho_g) * 9.81 * Ag * Ug
    if n == 1:
        ml = 0.
        bl = 0.
    else:
        ml = rho_l * Al * Ul ** 2
        bl = (rho - rho_l) * 9.81 * Al * Ul
    mo = mg + ml
    bo = bg + bl

    # The kinematic momentum and buoyancy fluxes are
    M = mo / rho
    B = bo / rho

    # Jet-to-plume transition length scale
    l_M = M ** (3. / 4.) / B ** (1. / 2.)

    # Characteristic velocity scale
    Ua = np.sqrt(mo / (rho * A))

    # Compute the mixture density
    if n == 1:
        rho_l = 0.
    rho_m = n * rho_g + (1. - n) * rho_l

    # Get the modified Weber number
    We_m = rho_m * Ua ** 2 * l_M / sigma_g

    # Compute the characteristic droplet size
    d = 4.3 * We_m ** (-3. / 5.) * l_M

    # Compute the actual gas and oil flow rate
    m_g = rho_g * Ag * Ug
    m_l = rho_l * Al * Ul

    # Return the characteristic size
    return d, m_g, m_l


def sintef(d0, m_gas, rho_gas, m_oil, rho_oil, mu_p, sigma, rho, mu):
    # Convert mass-flux to volume flux
    if np.sum(m_gas) > 0:
        q_gas = mass2vol(m_gas, rho_gas)
    else:
        q_gas = 0.
    if np.sum(m_oil) > 0:
        q_oil = mass2vol(m_oil, rho_oil)
    else:
        q_oil = 0.

    # Get the void-fraction adjusted characteristic exit velocity
    n = q_gas / (q_gas + q_oil)
    if q_oil == 0.:
        # This is gas only
        Un = 4 * q_gas / (np.pi * d0 ** 2)
        rho_m = rho_gas
    elif q_gas == 0:
        # This is oil only
        Un = 4 * q_oil / (np.pi * d0 ** 2)
        rho_m = rho_oil
    else:
        # This is oil and gas
        Un = 4 * q_oil / (np.pi * d0 ** 2) / (1 - n) ** (1 / 2)
        rho_m = rho_oil * (1 - n) + rho_gas * n

    Fr = Un / (9.81 * (rho - rho_m) / rho * d0) ** (1 / 2)
    Uc = Un * (1 + 1 / Fr)

    # Compute the particle size distribution parameters
    d50, de_max, k, alpha = sintef_model(Uc, d0, rho_oil, mu_p, sigma, rho)

    return d50, de_max, k, alpha


def sintef_model(Uc, d0, rho_p, mu_p, sigma, rho):
    """
    Computes the particle size for the Sintef equation

    Evaluates the parameters of the particle size distribution for the
    SINTEF equation and implements the d_95 rule as appropriate.  This
    function returns the parameters of the Rosin-Rammler size distribution
    with the spreading rates as reported in Johansen et al.

    Returns
    -------
    d50 : float
        Volume median diameter of the fluid phase of interest (m)
    de_max : float
        Maximum stable particle size of the fluid phase of interest (m)
    k : float
        Scale parameter for the Rosin-Rammler size distribution (--)
    alpha : float
        Shape parameter for the Rosin-Rammler size distribution (--)

    Notes
    -----
    This function is called by the `sintef()` function after several
    intermediate parameters are computed.  This function should not be
    called directly.

    """

    # Compute d_50 from the We model
    d50 = sintef_d50(Uc, d0, rho_p, mu_p, sigma)

    # Get an estimate of de_max
    de_max = de_max_oil(rho_p, sigma, rho)

    # Get the adjusted particle size distribution
    print('    ---> Predicted median droplet size = %g (m)' % d50)
    d50_from95, k, alpha = rosin_rammler_fit(d50, de_max)

    # Return the desired value for d50
    d50 = d50_from95

    if d50 > de_max:
        # Truncate the distribution
        d50 = de_max

    return d50, de_max, k, alpha


def sintef_d50(u0, d0, rho_p, mu_p, sigma):
    """
    Compute d_50 from the SINTEF equations

    Returns
    -------
    d50 : float
        Volume median diameter of the fluid phase of interest (m)

    Notes
    -----
    This function is called by the `sintef()` function after several
    intermediate parameters are computed.  This function should not be
    called directly.

    """
    # Compute the non-dimensional constants
    We = rho_p * u0 ** 2 * d0 / sigma
    Oh = mu_p / (rho_p * sigma * d0) ** 0.5
    r, p, q = 1.791, 0.46, -0.518   #　1.791　9.67　14.05

    if We > 350.:
        d50 = d0 * r * (1 + 10 * Oh) ** p * We ** q
    else:
        # Sinuous wave breakup...use the pipe diameter
        d50 = 1.2 * d0

    # Return the result
    return d50


def log_normal_fit(d50, d_max, sigma=0.27):
    """
    Return d_50 and sigma for the log-normal distribution

    Parameters
    ----------
    d50 : float
        Volume median diameter (m)
    d_max : float
        Maximum stable diameter (m)
    sigma : float, default=0.27
        Standard deviation of the Log-normal distribution in logarithmic
        units.

    Returns
    -------
    d_50 : float
        Volume median diameter (m)
    sigma : float
        Standard deviation of the Log-normal distribution in logarithmic
        units.

    Notes
    -----
    This function follows the idea of Sintef to not let d_95 of the particle
    size distribution exceed the maximum stable particle size.  If the
    original d_50 and d_max result in d_95 exceeding d_max, the d_50 is
    shifted downward such that d_95 will equal d_max.  Otherwise, the original
    d_50 is preserved.

    """
    # Adjust down if d50 exceeds the de_max
    if d_max is None:
        # Do not adjust the fit
        d50 = d50

    else:
        # Comnpute d95 for the given d50 and sigma
        mu = np.log(d50)
        mu_95 = mu + 1.6449 * sigma
        d95 = np.exp(mu_95)

        # Adjust d_50 so that d_95 does not exceed d_max
        if d95 > d_max:
            print('\nPredicted size distribution exceeds d50...')
            print('    ---> Adjusting size distribution down.\n')
            d50 = np.exp(np.log(d_max) - 1.6449 * sigma)

    # Return the final distribution fit
    return d50, sigma


def rosin_rammler_fit(d50, d_max, alpha=1.8):
    """
    Return d_50, k, and alpha for the Rosin-Rammler distribution

    Parameters
    ----------
    d50 : float
        Volume median diameter (m)
    d_max : float
        Maximum stable diameter (m)
    alpha : float, default=1.8

    Returns
    -------
    d_50 : float
        Volume median diameter (m)
    k : float
        Scale parameter for the Rosin-Rammler size distribution (--)
    alpha : float
        Shape parameter for the Rosin-Rammler size distribution (--)

    Notes
    -----
    This function follows the idea of Sintef to not let d_95 of the Rosin-
    Rammler distribution exceed the maximum stable particle size.  If the
    original d_50 and d_max result in d_95 exceeding d_max, the d_50 is
    shifted downward such that d_95 will equal d_max.  Otherwise, the original
    d_50 is preserved.

    """
    # k parameter for d50
    k = np.log(0.5)

    # Adjust down if d95 exceeds the de_max
    if isinstance(d_max, type(None)):
        d50 = d50

    else:
        # Compute d95 for the given d50
        d95 = d50 * (np.log(1. - 0.95) / k) ** (1. / alpha)

        # Adjust d50 so that d95 does not exceed d_max
        if d95 > d_max:
            print('\nPredicted size distribution exceeds d50...')
            print('    ---> Adjusting size distribution down.\n')
            d95 = d_max
            k95 = np.log(0.05)
            d50 = d95 * (np.log(1. - 0.5) / k95) ** (1. / alpha)

    # Return the final distribution fit
    return d50, k, alpha


def find_de(de, rho_d, rho_c, mu_d, mu_c, sigma, nu_d, nu_c, g, dp, K,
            lam_crit, c_0):
    """
    Search for the critical stable bubble size

    Search for the maximum stable bubble size of a gas bubble in water using
    the method in Grace et al.

    Returns
    -------
    t_min : float
        The minimum time required for a disturbance of the given size to
        break the fluid particle.

    Notes
    -----
    This function is used by the `grace()` function for maximum stable
    particle size.  It should not be called directly.

    """
    # Select the best available equations of state module
    from particle_functions import get_particle_shape, us_sphere, us_ellipsoid, us_spherical_cap
    from scipy.optimize import minimize

    # Time available for growth, t_a
    # The travel time from the position where disturance starts to the equator
    # Compute the rise velocity of this bubble size
    shape = get_particle_shape(de, rho_d, rho_c, mu_c, sigma)
    if shape == 1:
        U = us_sphere(de, rho_d, rho_c, mu_c)
    elif shape == 2:
        U = us_ellipsoid(de, rho_d, rho_c, mu_d, mu_c, sigma, -1)
    else:
        U = us_spherical_cap(de, rho_d, rho_c)

    # lam_max is upper limit on leading interface disturbance size
    lam_max = np.pi * de / 2.

    # Find the wave length that corresponds to the maximum growth rate
    delta = 2. * np.finfo(float).eps
    lam = minimize(grow_time, lam_crit, args=(de, U, nu_c, nu_d,
                                              sigma, g, dp, rho_c, rho_d, K, c_0),
                   bounds=[((1. + delta) * lam_crit, lam_max)]).x[0]

    t_min = grow_time(lam, de, U, nu_c, nu_d, sigma, g, dp, rho_c, rho_d, K, c_0)

    # Return the growth time
    return t_min


def grow_rate(n, k, nu_c, nu_d, sigma, g, dp, rho_c, rho_d, K):
    """
    Compute the instability growth rate on a gas bubble

    Write instability growth rate equation in Grace et al. as a root
    problem for n = f(k)

    Returns
    -------
    res : float
        The residual of the growth-rate equation expressed as a root-finding
        problem.

    Notes
    -----
    This function is used by the `grace()` function for maximum stable
    particle size.  It should not be called directly.

    """
    # Compute the kinematic viscosity from the dynamic viscosity
    mu_c = nu_c * rho_c

    # Compute more derived variables
    if n < 0.:
        m_c = k
        m_d = k
    else:
        m_c = np.sqrt(k ** 2 + n / nu_c)
        m_d = np.sqrt(k ** 2 + n / nu_d)

    # Compute the residual of the root function
    res = (sigma * k ** 3 - g * k * dp + n ** 2 * (rho_c + rho_d)) * \
          (k + m_c + K * (k + m_d)) + 4 * n * k * mu_c * (k + K * m_d) * \
          (K * k + m_c)

    # Return the residual
    return res


def grow_time(lam, de, U, nu_c, nu_d, sigma, g, dp, rho_c, rho_d, K, c_0):
    """
    Compare the available and needed disturbance growth times for instability

    Compares the time available for a disturbance to grow to the time needed
    for that disturbance to break a bubble.

    Returns
    -------
    t_cr : float
        The critical time (s) for which the required grow time equals the
        available time

    Notes
    -----
    This function is used by the `grace()` function for maximum stable
    particle size.  It should not be called directly.

    """
    # Compute the derived variables
    k = 2. * np.pi / lam

    # Consider disturbances with a node at the nose
    theta_1 = lam / (2. * de)

    # Compute the available time for disturbance growth
    t_a = de / 2 / U * (1. + 3. / 2. * K) * np.log(1. / np.tan(theta_1 / 2.))

    # Compute the grwoth rate of this disturbance
    n0 = 1. / t_a  # Value at the critical point
    n = fsolve(grow_rate, 5. * n0, args=(k, nu_c, nu_d, sigma, g, dp, rho_c, rho_d, K))[0]

    # Relate n to t_e
    t_e = 1. / n

    # Return the critical growth time
    return c_0 * t_e - t_a


def grace(rho_c, rho_d, mu_c, mu_d, sigma, fp_type=0):
    """
    Implement the Grace et al. algorithm for maximum stable particle size

    Computes the maximum stable particle size of an immiscible particle
    rising in stagnant water following a method in Grace et al.

    Parameters
    ----------
    rho_c : float
        Density of the continuous-phase ambient fluid (kg/m^3)
    rho_d : float
        Density of the immiscible fluid particle subject to breakup (kg/m^3)
    mu_c : float
        Dynamic viscosity of the continuous-phase ambient fluid (Pa s)
    mu_d : float
        Dynamic viscosity of the immiscible fluid particle subject to breakup
        (Pa s)
    sigma : float
        Interfacial tension between the continuous phase ambient fluid and
        the immiscible fluid particle subject to breakup (N/m)
    fp_type : int, default=0
        Phase of the immiscible fluid particle; 0 = gas, 1 = liquid.

    Returns
    -------
    de_max : float
        Equivalent spherical diameter of the maximum stable fluid particle
        subject to breakup in stagnant water (m)

    See Also
    --------
    grow_rate, grow_time, find_de

    Notes
    -----
    Implements the method in * Grace, J.R., Wairegi, T., Brophy, J., (1978)
    "Break-up of drops and bubbles in stagnant media," Can. J. Chem. Eng. 56
    (1), 3-8.

    """
    # Set the fit parameter
    if fp_type == 0:
        # This is gas
        c_0 = 3.8
    else:
        # This is liquid
        c_0 = 1.4

    # Compute the derived properties
    dp = np.abs(rho_c - rho_d)
    K = mu_d / mu_c
    nu_c = mu_c / rho_c
    nu_d = mu_d / rho_d

    # Region of instability.
    # lam_crit is lower limit on unstable wavelengths
    lam_crit = 2. * np.pi * np.sqrt(sigma / (9.81 * dp))

    # Lower limit on maximum stable diameter is lam_crit = lam_max
    de_max_star = 2. / np.pi * lam_crit

    # Choose Initialize the search near this minimum
    de = 1.01 * de_max_star

    # Find the maximum stable bubble size
    de_max = fsolve(find_de, de, args=(rho_d, rho_c, mu_d, mu_c, sigma, nu_d,
                                       nu_c, 9.81, dp, K, lam_crit, c_0))[0]

    # Return the result
    return de_max


def mass2vol(m, rho):
    """
    Convert a mass or mass flux to a volume or volume flux

    Parameters
    ----------
    m : ndarray
        Array of masses (kg) or mass fluxes (kg/s) for each component of a
        given fluid
    rho : float
        In-situ density of a given fluid (kg/m^3)

    Returns
    -------
    q : float
        Corresponding volume (m^3) or volume flux (m^3/s) of a given fluid

    """
    # Compute volume handling zero-fluxes correctly
    if np.sum(m) > 0:
        q = np.sum(m) / rho
    else:
        q = 0.

    return q


def de_max_oil(rho_p, sigma, rho):
    """
    Calculate the maximum stable oil droplet size

    Predicts the maximum stable liquid particle size per Clift et al. (1978)
    via the equation:

    d_max = 4. * np.sqrt(sigma / (g (rho - rho_p)))

    Parameters
    ----------
    rho_p : float
        Density of the phase undergoing breakup (kg/m^3)
    sigma : float
        Interfacial tension between the phase undergoing breakup and the
        ambient receiving continuous phase (N/m)
    rho : float
        Density of the ambient receiving continuous phase (kg/m^3)

    Returns
    -------
    d_max : float
        Maximum stable particle size (m)

    """
    return 4. * np.sqrt(sigma / (9.81 * (rho - rho_p)))


def rosin_rammler(nbins, d50, k, alpha):
    """
    Return the volume size distribution from the Rosin-Rammler distribution

    Returns the fluid particle diameters in the selected number of bins on
    a volume-fraction basis from the Rosin Rammler distribution with
    parameters d_50, k, and alpha.

    Parameters
    ----------
    nbins : int
        Desired number of size bins in the particle volume size distribution
    d50 : float
        Volume mean particle diameter (m)
    k : float
        Scale parameter of the Rosin-Rammler distribution (=log(0.5) for d_50)
    alpha : float
        Shape parameter of the Rosin-Rammler distribution

    Returns
    -------
    de : ndarray
        Array of particle sizes at the center of each bin in the distribution
        (m)
    vf : ndarray
        Volume fraction in each bin (--)

    """
    # Get the de/d50 ratio for the edges of each bin in the distribution
    # using a log-spacing
    a99 = (np.log(1. - 0.995) / k) ** (1. / alpha)
    a01 = (np.log(1. - 0.01) / k) ** (1. / alpha)
    bin_edges = np.logspace(np.log10(a01), np.log10(a99), nbins + 1)

    # Find the logarithmic average location of the center of each bin
    bin_centers = np.zeros(len(bin_edges) - 1)

    for i in range(len(bin_centers)):
        bin_centers[i] = np.exp(np.log(bin_edges[i]) +
                                (np.log(bin_edges[i + 1]) - np.log(bin_edges[i])) / 2.)

    # Compute the actual diameters of each particle
    de = d50 * bin_centers

    # Get the volume fraction within each bin
    if d50 == 0:
        vf = np.zeros(len(bin_centers))
    else:
        vn = 1. - np.exp(k * bin_edges ** alpha)
        vf = np.zeros(len(bin_centers))
        for i in range(len(bin_edges) - 1):
            vf[i] = vn[i + 1] - vn[i]
        vf = vf / np.sum(vf)

    # Return the particle size distribution
    return de, vf


def ln2rr(d50, sigma):
    """
    Convert the parameters of a log-normal distribution to Rosin-Rammler

    Parameters
    ----------
    d50 : float
        The median particle size of a volume distribution
    sigma : float
        Standard deviation of the Log-normal distribution in logarithmic
        units.

    Returns
    -------
    d50 : float
        Volume mean particle diameter (m)
    k : float
        Scale parameter of the Rosin-Rammler distribution (=log(0.5) for d_50)
    alpha : float
        Shape parameter of the Rosin-Rammler distribution

    """
    # Compute d95 of the log-normal distribution
    mu = np.log(d50)
    mu_95 = mu + 1.6449 * sigma
    d95 = np.exp(mu_95)

    # Find parameters of Rosin-Rammler with same d50 and d95
    k = np.log(0.5)
    alpha = np.log(np.log(1. - 0.95) / k) / np.log(d95 / d50)

    return d50, k, alpha