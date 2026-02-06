"""
Oil Weathering Processes
========================

Algorithms for oil weathering including evaporation, emulsification,
natural dispersion, and spreading.

References
----------
Fingas, M., & Fieldhouse, B. (2004). Formation of water-in-oil emulsions.
    Journal of Hazardous Materials.
Johansen, Ø. (2003). Development and verification of deep-water blowout
    models. Marine Pollution Bulletin, 47(9-12), 360-368.

"""

from __future__ import annotations

import numpy as np
from scipy import stats

import seawater

# Physical constants
g = 9.81  # Gravitational acceleration (m/s²)


def spreading_factors(slick, p):
    """
    Compute the factors required by spreading algorithm

    :param slick: SPM_utilities.Slick
        Object describing the properties of the oil slick.
    :param p: SPM.ModelParams
        Container for the fixed model parameters

    :return C: float
        See Technical manual of GNOME
    :return K: float
        See Technical manual of GNOME

    """

    # The Spreading Law coefficient
    k_v = 1.45

    # The kinematic viscosity of water (m^2/s)
    v_w = 1e-6

    # The oil-water density difference (in kg/m^3)
    delta_rho = (seawater.density(p.T_sur, p.S_sur, p.P_sur) - slick.rho) / \
                seawater.density(p.T_sur, p.S_sur, p.P_sur)

    # The volume of oil (m^3)
    volume = sum(slick.m) / slick.rho

    # Recasting as a differential equation
    C = np.pi * k_v ** 2 * (delta_rho * p.g * volume ** 2 / v_w ** (1 / 2)) ** (1 / 3)
    K = 4 * np.pi * 2 * 0.033

    return C, K


def spreading_rate(slick, p):
    """
    Compute the rate of oil spreading

    :param slick: SPM_utilities.Slick
        Object describing the properties of the oil slick.
    :param p: SPM.ModelParams
        Container for the fixed model parameters

    :return rate: float
        The rate of oil spreading (m^2/s)
    """

    if sum(slick.m) == 0 or slick.A == 0:
        rate = 0
    else:
        C, K = spreading_factors(slick, p)
        rate = 1 / 2 * C ** 2 / slick.A + 7 / 6 * K * (slick.A / K) ** (1 / 7) * p.time_interval

    return rate


def wind_coefficient(wind_speed):
    """
    Compute the mass transfer coefficient

    :param wind_speed: float
        Current wind speed (m/s)

    :return coeff: float
        Mass transfer coefficient (m/s)

    """
    c = 0.025
    if wind_speed <= 10:
        coeff = c * wind_speed ** 0.78
    else:
        coeff = 0.06 * c * wind_speed ** 2

    return coeff


def initial_area(cloud, T, S, P):
    """
    Compute the initial area of slick

    :param cloud: SPM_utilities.ParticleCloud
        Object describing the properties of the cloud (a collection of particles with the same size).
    :param T: float
        Surface temperature (K)
    :param S: float
        Surface salinity (PSU)
    :param P: float
        Surface pressure (Pa)

    :return area: float
        Initial area of slick

    """

    # Seawater density
    rho_seawater = seawater.density(T, S, P)

    # The oil-water density difference (in kg/m^3)
    delta_rho = (rho_seawater - cloud.particle.FluidParticle.density(cloud.m_particle, T, P)) / rho_seawater

    # The initial volume of slick
    volume = sum(cloud.m_Cloud) / cloud.particle.FluidParticle.density(cloud.m_particle, T, P)

    # The initial area of slick
    area = 42.09 * delta_rho ** (1 / 6) * volume ** (5 / 6)

    return area


def correct_area_mass(integrator, slick):
    """
    A function to correct the slick area

    :param integrator: scipy.integrate.ode
        An integrator object
    :param slick: SPM_utilities.Slick
        Object describing the properties of the oil slick.

    """
    mass = integrator.y[4:-1]
    # # Loop all the PC's masses, avoid them become negative
    mass[mass < 1e-4] = 0

    # Compute the maximum area of a slick
    area_max = sum(slick.m0) / slick.rho0 / thickness_limit(slick)

    if integrator.y[3] > area_max:
        integrator.y[3] = area_max
    elif sum(mass) == 0:
        integrator.y[3] = 0


def thickness_limit(slick):
    """
    A function to compute the minimum thickness of a slick

    :param slick: SPM_utilities.Slick
        Object describing the properties of the oil slick.

    :return: float
        The terminal thickness of a slick (m)

    """

    # convert dynamic viscosity (mu, Pa s) to kinematic viscosity (nu, m^2/s)

    if slick.nu0 > 10 ** -4:
        delta_min = 10 ** -4
    elif 10 ** -6 <= slick.nu0 <= 10 ** -4:
        delta_min = 10 ** -5 + 0.9091 * (slick.nu0 - 10 ** -6)
    else:
        delta_min = 10 ** -5

    return delta_min


def get_thickness(slick):
    """
    A function to compute the slick thickness

    :param slick: SPM_utilities.Slick
        Object describing the properties of the oil slick.

    :return delta: float
        The slick thickness (m)

    """

    mass = sum(slick.m)

    if mass == 0 or slick.A == 0:
        delta = np.NAN
    else:
        volume = mass / slick.rho
        delta = volume / slick.A
        mini_delta = thickness_limit(slick)
        if delta < mini_delta:
            delta = mini_delta

    return delta


def max_water_fraction(cloud):
    """
    Compute the maximum water content

    :param cloud: SPM_utilities.ParticleCloud
        Object describing the properties of the cloud (a collection of particles with the same size).

    :return y_max: float
        The maximum water fraction ranging from 0 ot 1

    """

    # Get the mass of resins and asphaltenes
    resins, asph = cloud.m_Cloud[-2], cloud.m_Cloud[-1]

    # Get the asphaltene/resin ratio
    z_ar = asph / resins

    # Compute the maximum water content based on Fingas and Fieldhouse (2004)
    y_max = 0.61 + 0.5 * z_ar - 0.28 * z_ar ** 2

    # The maximum water content is always less than 0.9
    if y_max > 0.9:
        y_max = 0.9

    return y_max


def evaporation_rate(slick, p, wind_speed):
    """
    Compute the evaporation rate for each PC corrected by step size

    :param slick: SPM_utilities.Slick
        Object describing the properties of the oil slick.
    :param p: SPM.ModelParams
        Container for the fixed model parameters
    :param wind_speed: float
        The wind speed (m/s).
    :return rate: ndarray
        The evaporation rate for each PC

    """

    # Get mass transfer coefficient
    fraction = slick.m / slick.particle.FluidParticle.M
    mass_term = np.sum(fraction)

    if mass_term == 0:
        rate = np.zeros_like(slick.m)
    else:
        k = wind_coefficient(wind_speed)
        VP = slick.particle.FluidParticle.VP(float(p.T_sur))
        M = slick.particle.FluidParticle.M
        rate = -(1 - slick.Y) * slick.A * k * M * VP / (p.R * p.T_sur) * fraction / mass_term * p.time_interval

    return rate


def emulsification_rate(slick, p, wind_speed, kemul=1e-6):
    """
    Compute the emulsification rate corrected by step size

    :param slick: SPM_utilities.Slick
        Object describing the properties of the oil slick.
    :param p: SPM.ModelParams
        Container for the fixed model parameters.
    :param wind_speed: float
        The wind speed (m/s).
    :param kemul: float, default 1e-6
        The water uptake coefficient.

    :return rate: float
        The emulsification rate

    """

    # when 1) the emulsion type is unstable or 2) oil mass weathers out or 3) didn't reach the onset,
    # assume no emulsification
    if slick.emulsion_type == 3 or sum(slick.m) == 0 or slick.age <= slick.t_emul:
        rate = 0
    else:
        rate = kemul * slick.nu0 / slick.nu * wind_speed ** 2 * (slick.Ymax - slick.Y) * p.time_interval
        print(slick.nu0 / slick.nu, wind_speed, rate)
    return rate


def dispersion_rate_GNOME(slick, p, wind_speed):
    if sum(slick.m) == 0:
        Qe = np.zeros_like(slick.m)
    else:
        Ventrain = 3.9e-8
        wind_stress = 0.71 * wind_speed ** 1.23  # wind stress factor (unit:m/s)
        H0 = 0.0248 * wind_stress ** 2  # significant wave height (unit:m)
        Hrms = 0.707 * H0  # root-mean wave height (unit: meters)
        De = 0.0034 * seawater.density(p.T_sur, p.S_sur, p.P_sur) * p.g * Hrms ** 2
        fbw = WCC(wind_speed)
        cRoy = 2400 * np.exp(-73.682 * slick.nu ** (1 / 2))
        Cdisp = De ** 0.57 * fbw
        Q = - cRoy * Cdisp * Ventrain * (1 - slick.Y) * slick.A * p.time_interval
        Qe = Q * slick.mf

    return Qe


def dispersion_rate_Fingas(slick, p, wind_speed):
    if sum(slick.m) == 0:
        Qe = np.zeros_like(slick.m)
    else:
        wind_stress = 0.71 * wind_speed ** 1.23  # wind stress factor (unit:m/s)
        H0 = 0.0248 * wind_stress ** 2  # significant wave height (unit:m)
        Hrms = 0.707 * H0  # root-mean wave height (unit: meters)
        fbw = Fbw(wind_speed)
        Fd = 6.3 * 10 ** -3 / (slick.mu * 1e3) ** 1.5 * (34.4 * Hrms ** 2) ** 0.57 * fbw
        Qe = -Fd * slick.A * p.time_interval * slick.mf

    return Qe


def Fbw(wind_speed, wind_speed_th=5):
    wind_stress = 0.71 * wind_speed ** 1.23  # wind stress factor (unit:m/s)
    peak_wave_period = 0.83 * wind_stress

    if wind_speed < wind_speed_th:
        fbw = 0
    else:
        fbw = 0.032 * (wind_speed - wind_speed_th) / peak_wave_period

    return fbw


def dispersion_rate_Johansen(slick, p, wind_speed):
    if sum(slick.m) == 0:
        Q = np.zeros_like(slick.m)
    else:
        # Tm = 0.812 * np.pi * wind_speed / g
        Tm = 0.52 * wind_speed
        W = WCC(wind_speed)
        wind_stress = 0.71 * wind_speed ** 1.23  # wind stress factor (unit:m/s)
        H0 = 0.0248 * wind_stress ** 2  # significant wave height (unit:m)
        d50 = new_median_droplet_size(slick.delta, slick.sigma, slick.mu, H0, slick.rho)
        de, vf = log_normal(d50, nbins=20)
        P_prime = get_P_prime(vf, de, D_prime=107e-6)
        alpha = P_prime * W / Tm
        Q = -alpha * slick.m * p.time_interval

    return Q


def dispersion_rate_Li(slick, p, wind_speed, a=4.604 * 1e-10, b=1.805, c=-1.023):
    wind_stress = 0.71 * wind_speed ** 1.23  # wind stress factor (unit:m/s)
    H0 = 0.0248 * wind_stress ** 2  # significant wave height (unit:m)
    rho_w = 1030
    d0 = 4 * (slick.sigma / (rho_w - slick.rho) / g)
    We = rho_w * g * H0 * d0 / slick.sigma
    Oh = slick.mu / (slick.rho * slick.sigma * d0) ** 0.5
    fbw = Fbw(wind_speed)
    Q0 = a * We ** b * Oh ** c
    Q = Q0 * slick.rho * slick.delta * fbw

    return Q


def WCC(wind_speed):
    # white cap coverage
    if wind_speed <= 3.7:
        W = 0
    elif 3.7 < wind_speed < 11.25:
        W = 3.18 * 1e-3 * (wind_speed - 3.7) ** 3
    elif 9.25 <= wind_speed < 23.09:
        W = 4.82 * 1e-4 * (wind_speed + 1.98) ** 3
    else:
        W = 4.82 * 1e-4 * (23.1 + 1.98) ** 3
    W = W / 100

    return W


def median_droplet_size(delta, sigma, mu, H, rho_p):
    a = 0.6
    A = 2.251
    B_p = 0.027
    Uh = (2 * g * H) ** (1 / 2)
    Vi = mu * Uh / sigma
    We = rho_p * Uh ** 2 * delta / sigma
    d50 = A * We ** -a * (1 + B_p * Vi ** a) * delta

    return d50


def new_median_droplet_size(delta, sigma, mu, H, rho_p):
    Uh = (2 * g * H) ** (1 / 2)
    d50 = (18.41 * ((rho_p * Uh ** 2 * delta) / sigma) ** -0.6 +
           0.64 * ((rho_p * Uh * delta) / mu) ** -0.6) * delta

    return d50


def log_normal(d50, nbins=10, sigma=0.403):  # 0.38  0.4*np.log(10)
    a0 = np.exp(np.log(d50) - 2.8 * sigma) / d50
    a1 = np.exp(np.log(d50) + 2.3 * sigma) / d50
    bin_edges = np.logspace(np.log10(a0), np.log10(a1), nbins + 1)

    # Find the logarithmic average location of the center of each bin
    bin_centers = np.zeros(len(bin_edges) - 1)

    for i in range(len(bin_centers)):
        bin_centers[i] = np.exp(np.log(bin_edges[i]) +
                                (np.log(bin_edges[i + 1]) - np.log(bin_edges[i])) / 2.)

    # Compute the actual diameters of each particle
    de = d50 * bin_edges[1:]

    # Compute log-normal parameters for de/d50 distribution
    mu = np.log(d50)

    Y = stats.lognorm(s=sigma, scale=np.exp(mu))
    cum_vf = [Y.cdf(i) for i in de]
    vf = np.diff(cum_vf)
    vf = np.append(np.diff(cum_vf), 1 - sum(vf))

    return de, vf


def get_P_prime(vf, de, D_prime=107e-6):
    de = np.flip(de)
    vf = np.flip(vf)

    if max(de) <= D_prime:
        P_prime = 1
    elif min(de) >= D_prime:
        P_prime = 0
    else:
        for idx, i in enumerate(de):
            if i <= D_prime:
                P_prime = np.sum(vf[idx:])
                break
            else:
                P_prime = 0

    if P_prime > 1: P_prime = 1

    return P_prime


def emulsion_vis(slick):
    if sum(slick.m) == 0:
        mu = np.NAN
    else:
        mu = slick.mu0 * (1 + 1.15 * slick.Y / (1.187 - 1.15 * slick.Y)) ** 2.49

    return mu


def emulsion_den(slick, p):
    if sum(slick.m) == 0:
        rho = np.NAN
    else:
        rho = slick.Y * seawater.density(p.T_sur, p.S_sur, p.P_sur) + (1 - slick.Y) * slick.rho0

    return rho


def get_time_emulsion(mf, vis, rho, wave_height):
    content_resin = mf[-2] * 100
    content_asph = mf[-1] * 100
    content_satu = sum((mf[0], mf[2], mf[4], mf[6], mf[8],
                        mf[10], mf[12], mf[14], mf[16], mf[18])) * 100
    vis = vis * 1e6
    rho = rho / 1000

    def transformed_resin():
        if content_resin == 0:
            coeff = 20
        else:
            coeff = abs(content_resin - 10)

        return coeff

    def transformed_asph():
        if content_asph == 0:
            coeff = 20
        else:
            coeff = abs(content_asph - 4)

        return coeff

    def transformed_satu():
        coeff = abs(content_satu - 45)

        return coeff

    def get_emul_time(emulsion_type):
        if emulsion_type == 0 or 10:
            # emulsion type is stable
            a = 27.1
            b = 7520
            y = (a + b / (wave_height * 100) ** 1.5) / 60
        elif emulsion_type == 1:
            # emulsion type is Meso-stable
            a = 47
            b = 49100
            y = (a + b / (wave_height * 100) ** 1.5) / 60
        elif emulsion_type == 2:
            # emulsion type is entrained
            a = 30.8
            b = 18300
            y = (a + b / (wave_height * 100) ** 1.5) / 60
        elif emulsion_type == 3:
            # emulsion type is unstable
            y = np.NAN

        return y

    A = np.exp(rho)
    B = np.log(vis)
    C = transformed_resin()
    D = transformed_asph()
    E = content_asph / content_resin
    F = np.exp(A)
    G = np.exp(E)
    H = np.log(A)
    I = np.log(B)
    J = np.log(transformed_satu())
    K = np.log(transformed_resin())
    L = np.log(transformed_asph())
    M = np.log(E)

    sta = -15.3 + 1010 * A - 3.66 * B + 0.174 * C - 0.579 * D + 34.4 * E + 1.02 * F - 7.91 * G \
          - 2740 * H + 12.2 * I - 0.334 * J - 3.17 * K + 0.99 * L - 2.29 * M

    if -20 <= sta <= 3 and rho > 0.94 and vis > 6000:
        emul_type = 2
    elif -18 <= sta <= -4 and (rho < 0.85 or rho > 1) and (vis < 100 or vis > 800000) and \
            (content_resin < 1 or content_asph < 1):
        emul_type = 3
    elif 4 <= sta <= 29:
        emul_type = 0
    elif -10 <= sta <= 5:
        emul_type = 1
    else:
        emul_type = 10

    y = get_emul_time(emul_type)

    return y, emul_type


def get_wave_height(profile, y, t, p):
    time_wind = t + p.dt02_wind
    u_wind, v_wind = wind_velocity(profile[2], profile[3], y, time_wind)
    wind_speed = np.sqrt(u_wind ** 2 + v_wind ** 2)
    wind_stress = 0.71 * wind_speed ** 1.23  # wind stress factor (unit:m/s)
    H0 = 0.0248 * wind_stress ** 2  # significant wave height (unit:m)

    return H0


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
        # u, v = 0, 0
        raise Exception('wind_velocity in weathering_functions is wrong')

    return u, v