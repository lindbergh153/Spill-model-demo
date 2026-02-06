"""
Particle Thermodynamic Functions
================================

Peng-Robinson equation of state calculations, viscosity correlations,
and phase equilibrium functions for oil/gas mixtures.

References
----------
Peng, D.-Y., & Robinson, D.B. (1976). Industrial & Engineering Chemistry.
Lin, H., & Duan, Y.Y. (2005). Fluid Phase Equilibria.

"""

from __future__ import annotations

import numpy as np
from numba import njit

# Physical constants
R = 8.314510  # Universal gas constant (J/mol/K)
G = 9.81      # Gravitational acceleration (m/s²)


def kh_insitu(T, P, S, kh_0, dH_solR, nu_bar, K_salt):
    P_ATM = 101325
    M_SEA = 0.06835

    # Adjust from STP to ambient temperature, the ambient pressure, the salting out effect of salinity.
    kh = kh_0 * np.exp(dH_solR * (1 / T - 1 / 298.15)) * \
         np.exp((P_ATM - P) * nu_bar / (R * T)) * 10 ** (-S / M_SEA * K_salt)
    kh[kh < 0] = 0

    return kh


def z_pr(nc, T, P, m, M, Pc, Tc, omega, delta):

    A, B, Ap, Bp, yk = coefs(nc, T, P, m, M, Pc, Tc, omega, delta)
    if isinstance(A, np.ndarray):
        A, B = A[0], B[0]
    p_coefs = [1, B - 1, A - 2 * B - 3 * B ** 2, B ** 3 + B ** 2 - A * B]  # z-factor of liquid and gas
    z_roots = np.roots(p_coefs)

    # Extract the correct z-factors
    z_max = 0
    for i in range(len(z_roots)):
        if z_roots[i].imag == 0:
            if z_roots[i].real > z_max:
                z_max = z_roots[i].real

    z_min = z_max
    for i in range(len(z_roots)):
        if z_roots[i].imag == 0.0:
            if z_min > z_roots[i].real > 0:
                z_min = z_roots[i].real

    # Return the z-factors in z
    z = np.zeros((2,1))
    z[0,0] = z_max # gas
    z[1,0] = z_min # liq

    return z, A, B, Ap, Bp, yk


@njit
def coefs(nc, T, P, mass, M, Pc, Tc, omega, delta):
    n_moles = mass / M
    yk = n_moles / sum(n_moles)
    # Compute the coefficient values for each gas in the mixture. Use the modified Peng-Robinson (1978) equations for mu
    m = np.zeros(nc)
    for i in range(nc):
        if omega[i] > 0.49:
            m[i] = 0.379642 + 1.48503 * omega[i] - 0.164423 * omega[i] ** 2 + 0.016666 * omega[i] ** 3
        else:
            m[i] = 0.37464 + 1.54226 * omega[i] - 0.26992 * omega[i] ** 2
    alpha = (1 + m * (1 - (T / Tc) ** (1 / 2))) ** 2
    ac = 0.45724 * R ** 2 * Tc ** 2 / Pc
    aTc = ac * alpha
    bc = 0.07780 * R * Tc / Pc

    # Use the mixing rules in McCain (1990)
    b = sum(yk * bc)
    aT = 0
    for j in range(nc):
        for i in range(nc):
            aT = aT + yk[i] * yk[j] * (aTc[i] * aTc[j]) ** (1 / 2) * (1 - delta[i, j])

    A = aT * P / (R ** 2 * T ** 2)
    B = b * P / (R * T)
    Bp = bc / b
    Ap = np.zeros(nc)
    for i in range(nc):
        Ap[i] = 1.0 / aT * (2 * aTc[i] ** (1 / 2) * sum(yk * aTc ** (1 / 2) * (1 - delta[:, i])))

    return A, B, Ap, Bp, yk


@njit
def volume_trans(T, Pc, Tc, Vc):
    # Compute the compressibility factor(--) for each component of the mixture
    Zc = Pc * Vc / (R * Tc)

    #  Calculate the parameters in the Lin and Duan(2005) paper: beta is from equation (12)
    beta = -2.8431 * np.exp(-64.2184 * (0.3074 - Zc)) + 0.1735

    # gamma is from Equation (13)
    gamma = -99.2558 + 301.6201 * Zc

    # Account for the temperature dependence (equation 10)
    f_Tr = beta + (1 - beta) * np.exp(gamma * np.abs(1 - T / Tc))

    # Compute the volume translation for the critical point (equation 9)
    cc = (0.3074 - Zc) * R * Tc / Pc

    # Finally, the volume translation at the given state is (equation 8)
    vt = f_Tr * cc

    return vt


def get_diffusivity(mu, Vb):
    # Use the Hayduk and Laudie formula
    D = 13.26e-9 / ((mu * 1.0e3) ** 1.14 * (Vb * 1e6) ** 0.589)
    D[D < 0] = 0

    return D


def get_density(t, p, mass, mol_wt, pc, tc, vc, omega, delta):
    nc = len(mass)
    z, A, B, Ap, Bp, yk = z_pr(nc, t, p, mass, mol_wt, pc, tc, omega, delta)

    # Convert the masses to mole fraction
    n_moles = mass / mol_wt
    yk = n_moles / sum(n_moles)

    # Compute the volume translation coefficient
    vt = volume_trans(t, pc, tc, vc)

    # Compute the molar volume
    nu = z * R * t / p - sum(yk * vt)

    # Compute and return the density
    rho = sum(yk * mol_wt) / nu

    return rho


def get_viscosity(T, P, mass, Mol_wt, Pc, Tc):
    # Enter the parameter values from Table 10.1
    GV = [-2.090975e5, 2.647269e5, -1.472818e5, 4.716740e4, -9.491872e3, 1.219979e3, -9.627993e1,
          4.274152, -8.141531e-2]
    A, B = 1.696985927, -0.133372346
    C, F = 1.4, 168.0
    jc = [-10.3506, 17.5716, -3019.39, 188.730, 0.0429036, 145.290, 6127.68]
    kc = [-9.74602, 18.0834, -4126.66, 44.6055, 0.976544, 81.8134, 15649.9]

    # Enter the properties for the reference fluid (methane)
    M0 = np.array([16.043e-3])
    Tc0 = np.array([190.56])
    Pc0 = np.array([4599000.0])
    omega0 = np.array([0.011])
    Vc0 = np.array([9.86e-5])
    delta0 = np.array([0.0]).reshape(1, 1)
    rho_c0 = np.array([162.84])

    #  1. Prepare the variables to determine the corresponding states between the given mixture and the
    #  reference fluid (methane)
    #  Get the mole fraction of the components of the mixture
    n_moles = mass / Mol_wt
    z = n_moles / sum(n_moles)
    #  Compute equation(10.19)
    nc = len(mass)
    # Tc_mix = numerator / denominator
    Tc_mix, numerator, denominator = get_Tc_mix(nc, z, Tc, Pc)

    #  Compute equation(10.22)
    Pc_mix = 8 * numerator / denominator ** 2

    #  Get the density of methane at TTc0 / Tc_mix and PPc0 / Pc_mix

    # t, p, mass, mol_wt, pc, tc, vc, omega, delta, aij, bij, delta_groups, calc_delta
    rho0 = get_density(T * Tc0 / Tc_mix, P * Pc0 / Pc_mix, np.array([1]), M0, Pc0, Tc0, Vc0,
                       omega0, delta0)

    #  Compute equation(10.27)
    rho_r = np.zeros((2,1))
    rho_r[:,0] = rho0[:,0] / rho_c0

    #  Compute equation(10.23), where M is in g / mol
    M = Mol_wt * 1e3
    M_bar_n = sum(z * M)
    M_bar_w = sum(z * M ** 2) / M_bar_n
    M_mix = 1.304e-4 * (M_bar_w ** 2.303 - M_bar_n ** 2.303) + M_bar_n

    #  Compute equation(10.26), where M is in g / mol
    M0 = M0 * 1e3
    alpha_mix = np.zeros((2,1))
    alpha0 = np.zeros((2,1))
    alpha_mix[:,0] = 1. + 7.378e-3 * rho_r[:,0]**1.847 * M_mix**0.5173
    alpha0[:,0] = 1. + 7.378e-3 * rho_r[:,0]**1.847 * M0**0.5173

    #  2. Compute the viscosity of methane at the corresponding state

    #  Corresponding state
    T0 = T * Tc0 / Tc_mix * alpha0 / alpha_mix
    P0 = P * Pc0 / Pc_mix * alpha0 / alpha_mix

    # Compute each state separately
    eta_ch4 = np.zeros((2,1))

    for i in range(2):
        #  Get the density of methane at T0 and P0. Be sure to use molecular weight in kg / mol
        rho0 = get_density(T0[i], P0[i], np.array([1]), M0 * 1e-3, Pc0, Tc0, Vc0, omega0, delta0)

        #  Compute equation(10.10)
        theta = (rho0 - rho_c0) / rho_c0

        #  Equation (10.9) with T in K and rho in g/cm^3
        rho0 = rho0 * 1e-3

        delta_eta_p = np.exp(jc[0] + jc[3] / T0[i]) * (np.exp(rho0 ** 0.1 * (jc[1] + jc[2] / T0[i] ** 1.5)
                                                           + theta * rho0 ** 0.5 * (
                                                                   jc[4] + jc[5] / T0 + jc[6] / T0[i] ** 2)) - 1.0)

        #  Equation (10.28)
        delta_eta_pp = np.exp(kc[0] + kc[3] / T0[i]) * (np.exp(rho0 ** 0.1 * (kc[1] + kc[2] / T0[i] ** 1.5)
                                                            + theta * rho0 ** 0.5 * (
                                                                    kc[4] + kc[5] / T0[i] + kc[6] / T0[i] ** 2)) - 1.0)

        #  Equation (10.7)
        eta_0 = GV[0] / T0[i] + GV[1] / T0[i] ** (2 / 3) + GV[2] / T0[i] ** (1 / 3) + GV[3] + GV[4] * T0[i] ** (1 / 3) + \
                GV[5] * T0[i] ** (2 / 3) + GV[6] * T0[i] + GV[7] * T0[i] ** (4 / 3) + GV[8] * T0[i] ** (5 / 3)

        #  Equation(10.8)
        eta_1 = A + B * (C - np.log(T0[i] / F)) ** 2

        #  Equation(10.32)
        delta_T = T0[i] - 91.0

        #  Equation(10.31)
        htan = (np.exp(delta_T) - np.exp(-delta_T)) / (np.exp(delta_T) + np.exp(-delta_T))

        #  Viscosity of methane (Equation 10.29) -- reported in (Pa s)
        F1, F2 = (htan + 1) / 2, (1 - htan) / 2.0
        eta_ch4 = (eta_0 + eta_1 + F1 * delta_eta_p + F2 * delta_eta_pp) * 1e-7

    #  Compute the viscosity of the mixture at the given T and P
    mu = (Tc_mix / Tc0) ** (-1 / 6) * (Pc_mix / Pc0) ** (2 / 3) * (M_mix / M0) ** 0.5 * alpha_mix / alpha0 * eta_ch4

    return mu


@njit
def get_Tc_mix(nc, z, Tc, Pc):
    numerator = 0.0
    denominator = 0.0
    for i in range(nc):
        for j in range(nc):
            numerator = numerator + \
                        z[i] * z[j] * \
                        ((Tc[i] / Pc[i]) ** (1 / 3) + (Tc[j] / Pc[j]) ** (1 / 3)) ** 3 * (Tc[i] * Tc[j]) ** (1 / 2)

            denominator = denominator + \
                          z[i] * z[j] * ((Tc[i] / Pc[i]) ** (1 / 3) + (Tc[j] / Pc[j]) ** (1 / 3)) ** 3
    Tc_mix = numerator / denominator

    return Tc_mix, numerator, denominator


def get_fugacity(T, P, mass, Mol_wt, Pc, Tc, omega, delta):
    nc = len(mass)

    z, A, B, Ap, Bp, yk = z_pr(nc, T, P, mass, Mol_wt, Pc, Tc, omega, delta)

    fug = np.zeros((2,len(mass)))
    for i in range(2):
        term1 = - np.log(z[i,0] - B)
        term2 = (z[i,0] - 1) * Bp
        term3 = z[i,0] + (2 ** 0.5 + 1) * B
        term4 = z[i,0] - (2 ** 0.5 - 1) * B
        term5 = np.log(term3 / term4)
        term6 = - A / (2 ** 1.5 * B) * (Ap - Bp)

        fug[i,:] = np.exp(term1 + term2 + term6 * term5) * yk * P

    return fug


def get_particle_shape(de, rho_p, rho, mu, sigma):
    # Calculate the non-dimensional variables
    Eo = eotvos(de, rho_p, rho, sigma)
    M = morton(rho_p, rho, mu, sigma)
    H = h_parameter(Eo, M, mu)

    # Select the appropriate shape classification
    if H < 2:
        shape_p = 1
    elif Eo < 40.0 and M < 0.001 and H < 1000.0:
        shape_p = 2
    else:
        shape_p = 3

    return shape_p


def eotvos(de, rho_p, rho, sigma):
    Eo = G * (rho - rho_p) * de ** 2 / sigma
    return Eo


def morton(rho_p, rho, mu, sigma):
    M = G * mu ** 4 * (rho - rho_p) / (rho ** 2 * sigma ** 3)
    return M


def h_parameter(Eo, M, mu):
    H = 4 / 3 * Eo * M ** -0.149 * (mu / 0.0009) ** -0.14
    return H


def reynolds(de, us, rho_c, mu_c):
    Re = rho_c * de * us / mu_c
    return Re


def xfer_sphere(de, us, rho, mu, D, sigma, mu_p, fp_type, status):
    if isinstance(D, np.float64):
        D = np.array([D])
    if isinstance(D, float):
        D = np.array([D])

    # Compute the correct mass transfer coefficients
    nc = len(D)
    beta = np.zeros(nc)
    if status > 0:
        if fp_type > 0:
            #  This is a liquid particle: use Kumar and Hartland
            beta = xfer_kumar_hartland(de, us, rho, mu, D, sigma, mu_p)
        else:
            #  This is gas: use larger of Johnson or Clift
            beta_j = xfer_johnson(de, us, D)
            beta_c = xfer_clift(de, us, rho, mu, D)

            #  For small particles, the Johnson formula is too low
            for i in range(nc):
                if beta_j[i] > beta_c[i]:
                    #  Johnson is ok
                    beta[i] = beta_j[i]
                else:
                    #  Clift is better
                    beta[i] = beta_c[i]

    else:
        beta = xfer_clift(de, us, rho, mu, D)

    return beta


def xfer_ellipsoid(de, us, rho, mu, D, sigma, mu_p, fp_type, status):
    if isinstance(D, np.float64):
        D = np.array([D])
    if isinstance(D, float):
        D = np.array([D])

    nc = len(D)
    beta = np.zeros(nc)
    if status > 0:
        if fp_type > 0:
            #  This is a liquid particle: use Kumar and Hartland
            beta = xfer_kumar_hartland(de, us, rho, mu, D, sigma, mu_p)
        else:
            #  This is gas: use larger of Johnson or Clift
            beta_j = xfer_johnson(de, us, D)
            beta_c = xfer_clift(de, us, rho, mu, D)

            #  For small particles, the Johnson formula is too low
            for i in range(nc):
                if beta_j[i] > beta_c[i]:
                    #  Johnson is ok
                    beta[i] = beta_j[i]
                else:
                    #  Clift is better
                    beta[i] = beta_c[i]

    else:
        beta = xfer_clift(de, us, rho, mu, D)

    return beta


def xfer_spherical_cap(de, us, rho, rho_p, mu, D, status):
    #  Compute the correct mass transfer coefficients
    if status > 0:
        # Use the Johnson et al.(1969) equation for clean bubbles
        beta = xfer_johnson(de, us, D)
    else:
        # Use the Clift et al.(1978) equation for spherical cap bubbles
        # Compute the wake angle for the partial sphere model (equation 8-1)
        theta_w = theta_w_sc(de, us, rho, mu)

        # Compute the surface area of the spherical cap and equivalent sphere
        A = surface_area_sc(de, theta_w)
        Ae = 4 * np.pi * (de / 2.0) ** 2

        # Compute the mass transfer(equation 8 - 28)
        beta = (1.25 * (G * (rho - rho_p) / rho) ** 0.25 * np.sqrt(D) / de ** 0.25) * Ae / A

    return beta


def xfer_clift(de, us, rho, mu, D):
    #  Compute the non - dimensional governing parameters
    Sc = mu / (rho * D)
    Pe = us * de / D
    Re = reynolds(de, us, rho, mu)
    nc = len(D)
    Sh = np.zeros(nc)

    #  Compute the Sherwood Number
    for i in range(nc):
        if D[i] > 0.0:
            if Re < 1.0:
                Sh[i] = 1 + (1.0 + Pe[i]) ** (1 / 3)
            elif Re < 100.0:
                Sh[i] = 1 + (1 + 1 / Pe[i]) ** (1 / 3) * Re ** 0.41 * Sc[i] ** (1 / 3)
            elif Re < 2000.0:
                Sh[i] = 1.0 + 0.724 * Re ** 0.48 * Sc[i] ** (1 / 3)
            else:
                Sh[i] = 1.0 + 0.425 * Re ** 0.55 * Sc[i] ** (1 / 3)
        else:
            Sh[i] = 0.0
    #  Return the mass transfer coefficient
    beta = Sh * D / de

    return beta


def xfer_johnson(de, us, D):
    #  Compute equation (42) in Johnson et al. (1969)
    beta = 1.13 * np.sqrt(D * us * 100.0 ** 3 / (0.45 + 0.2 * de * 100.0)) / 100.0

    return beta


def xfer_kumar_hartland(de, us, rho, mu, D, sigma, mu_p):
    # Compute the Reynolds, Schmidt, and Peclet numbers
    Re = de * us * rho / mu
    Sc = mu / (rho * D)
    Pe = de * us / D

    # Constants for the formulas
    C1 = 50
    C2 = 5.26e-2
    n1 = (1 / 3) + 6.59e-2 * Re ** 0.25
    n2 = 1.0 / 3.0
    n3 = 1.0 / 3.0
    n4 = 1.1

    # Compute equation 16
    Sh_rigid = 2.43 + 0.775 * Re ** 0.5 * Sc ** n2 + 0.0103 * Re * Sc ** n2

    # Compute equation 50
    Sh_infty = C1 + 2.0 / np.pi ** 0.5 * Pe ** 0.5

    # Compute lambda as the RHS of equation 51
    lam = C2 * Re ** n1 * Sc ** n2 * (us * mu / sigma) ** n3 * 1.0 / (1.0 + (mu_p / mu) ** n4)

    # Compute the in situ Sherwood number
    Sh = (Sh_infty * lam + Sh_rigid) / (1.0 + lam)

    # Convert Sherwood number to mass transfer coefficient
    beta = Sh * D / de

    return beta


def us_sphere(de, rho_p, rho_c, mu):
    Nd = 4.0 * rho_c * np.abs(rho_c - rho_p) * 9.81 * de ** 3 / (3.0 * mu ** 2)
    W = np.log10(Nd)
    if Nd <= 73.0:
        Re = Nd / 24.0 - 1.7569 * 10 ** -4 * Nd ** 2 + 6.9252 * 10 ** -7 * Nd ** 3 - 2.3027 * 10 ** -10 * Nd ** 4
    elif Nd <= 580.0:
        Re = 10.0 ** (-1.7095 + 1.33438 * W - 0.11591 * W ** 2)
    elif Nd <= 1.55 * 10 ** 7:
        Re = 10.0 ** (-1.81391 + 1.34671 * W - 0.12427 * W ** 2 + 0.006344 * W ** 3)
    elif Nd <= 5.0 * 10 ** 10:
        Re = 10.0 ** (5.33283 - 1.21728 * W + 0.19007 * W ** 2 - 0.007005 * W ** 3)
    else:
        Re = 0.0
    us = mu / (rho_c * de) * Re
    us = float(us)

    return us


def us_ellipsoid(de, rho_p, rho_c, mu_p, mu, sigma, status):
    Eo = eotvos(de, rho_p, rho_c, sigma)
    M = morton(rho_p, rho_c, mu, sigma)
    H = h_parameter(Eo, M, mu)

    if H > 59.3:
        J = 3.42 * H ** 0.441
    else:
        J = 0.94 * H ** 0.757
    Re = M ** -0.149 * (J - 0.857)
    us_dirty = mu / (rho_c * de) * Re

    # Return the correct slip velocity
    if status > 0:
        # Compute the clean-bubble correction from Figure 7.7 and Eqn. 7-10 in Clift et al. (1978)
        kappa = mu_p / mu
        xi = Eo * (1.0 + 0.15 * kappa) / (1.0 + kappa)
        gamma = 2.0 * np.exp(-(np.log10(xi) + 0.6383) ** 2 / (0.2598 + 0.2 * (np.log10(xi) + 1.0)) ** 2)
        us = us_dirty * (1 + gamma / (1.0 + kappa))
    else:
        us = us_dirty
    us = float(us)

    return us


def us_spherical_cap(de, rho_p, rho_c):
    us = 0.711 * np.sqrt(G * de * (rho_c - rho_p) / rho_c)
    us = float(us)

    return us


def theta_w_sc(de, us, rho_c, mu_c):
    # Get the Reynolds number
    Re = reynolds(de, us, rho_c, mu_c)
    # Return the wake angle
    theta_w = np.pi * (50.0 + 190.0 * np.exp(-0.62 * Re ** 0.4)) / 180.0

    return theta_w


def surface_area_sc(de, theta_w):
    # Match the volume
    V = 4 / 3 * np.pi * (de / 2.0) ** 3

    # Find the radius at the bottom of the partial sphere
    r_sc = (V / np.pi / (2 / 3 - np.cos(theta_w) + np.cos(theta_w) ** 3 / 3)) ** (1 / 3)

    # Surface area of the frontal sphere
    Af = 2.0 * np.pi * r_sc ** 2 * (1 - np.cos(theta_w))

    # Surface area of the real bottom of the partial sphere
    Ar = np.pi * (r_sc * np.sin(theta_w)) ** 2

    # Return the surface area
    area = Af + Ar

    return area

def successive_substitution(m, T, P, max_iter, M, Pc, Tc, omega, delta, K, steps=0):
    def update_K(K):
        # Get the mixture composition for the current K-factor
        xi, beta = gas_liq_eq(m, M, K)

        # Get tha gas and liquid fugacities for the current composition
        f_gas = get_fugacity(T, P, xi[0, :] * M, M, Pc, Tc, omega, delta)[0, :]
        f_liq = get_fugacity(T, P, xi[1, :] * M, M, Pc, Tc, omega, delta)[1, :]

        # Update K using K = (phi_liq / phi_gas)
        K_new = (f_liq / (xi[1, :] * P)) / (f_gas / (xi[0, :] * P))
        K_new[np.isnan(K_new)] = 0.

        if steps == 0.:
            moles = m / M
            zi = moles / np.sum(moles)
            if np.sum(zi * K_new) - 1. <= 0.:  # Condition 4 page 252
                xi[0, :] = K_new * zi / np.sum(K_new * zi)
                xi[1, :] = zi

                f_gas = get_fugacity(T, P, xi[0, :] * M, M, Pc, Tc, omega, delta)[0, :]
                f_liq = get_fugacity(T, P, xi[1, :] * M, M, Pc, Tc, omega, delta)[1, :]

                # Update K using K = (phi_liq / phi_gas)
                K_new = (f_liq / (xi[1, :] * P)) / (f_gas / (xi[0, :] * P))
                K_new[np.isnan(K_new)] = 0.

            elif (1. - np.sum(zi / K_new)) >= 0.:  # % Condition 5 page 252
                xi[0, :] = zi
                xi[1, :] = (zi / K_new) / np.sum(zi / K_new)

                f_gas = get_fugacity(T, P, xi[0, :] * M, M, Pc, Tc, omega, delta)[0, :]
                f_liq = get_fugacity(T, P, xi[1, :] * M, M, Pc, Tc, omega, delta)[1, :]

                # Update K using K = (phi_liq / phi_gas)
                K_new = (f_liq / (xi[1, :] * P)) / (f_gas / (xi[0, :] * P))
                K_new[np.isnan(K_new)] = 0.

        # Return an updated value for the K factors
        return K_new, beta

    # Set up the iteration parameters
    tol = 1.49012e-8  # Suggested by McCain (1990)
    err = 1.

    # Iterate to find the final value of K factor using successive
    # substitution
    stop = False
    while err > tol and steps < max_iter and not stop:
        # Save the current value of K factor
        K_old = K

        # Update the estimate of K factor using the present fugcaities
        K, beta = update_K(K)

        steps += 1
        if steps > 3 and (beta == 0. or beta == 1.):
            stop = True

        # Compute the current error based on the squared relative error
        # suggested by McCain (1990) and update the iteration counter
        err = np.nansum((K - K_old) ** 2 / (K * K_old))

    # Determine the exit condition
    if stop:
        # Successive substitution thinks this is single-phase, stability analysis is required
        flag = -1
    elif steps < max_iter:
        # This solution is converged
        flag = 1
    else:
        # No decision has been reached
        flag = 0

    # Update the equilibrium and return the last value of K-factor
    xi, beta = gas_liq_eq(m, M, K)

    return K, beta, xi, flag, steps


def stability_analysis(m, T, P, M, Pc, Tc, omega, delta, K, zi, di):
    moles = m / M
    zi = moles / np.sum(moles)

    # Generate the update equation for finding W that minimizes tm
    def update_W(W, phase):
        f_W = get_fugacity(T, P, W*M, M, Pc, Tc, omega, delta)[phase,:]

        # Get the fugacity coefficients
        phi_W = f_W / (W / np.sum(W) * P)

        W_new = np.exp(di - np.log(phi_W))

        return W_new

    # Compute the modified tangent plane distance
    def compute_tm(W, phase):
        # Compute the fugacity at the composition W
        f_W = get_fugacity(T, P, W * M, M, Pc, Tc, omega, delta)[phase, :]

        # Get the fugacity coefficients
        phi_W = f_W / (W / np.sum(W) * P)

        tm = 1. + np.sum(W * (np.log(W) + np.log(phi_W) - di - 1.))

        return tm

    # Solve for W that minimizes tm
    def find_W(W, phase):
        # Set up the iteration parameters
        tol = 1.49012e-8  # Use same value as for K-factor iteration
        err = 1.

        # Iterate to find the final value of W
        while err > tol:
            # Save the current value of W
            W_old = W

            # Update the estimate of W using the update equation
            W = update_W(W, phase)

            # Compute the current error based on the squared relative error suggested by McCain (1990)
            err = np.nansum((W - W_old) ** 2 / (W * W_old))

        # Compute the modified tangent plane distance
        tm = compute_tm(W, phase)

        # Determine if we found a trivial solution
        trivial = True
        for i in range(len(W)):
            if np.abs(W[i] - zi[i]) > 1.e-5:
                trivial = False

        # Evaluate the stability of the outcome
        if tm < 0. and not trivial:
            phases = 2
        else:
            # This is a single-phase gas
            phases = 1

        # Return the results
        return W, tm, phases

    # First, do a test vapor-like composition
    W = K * zi
    W_gas, tm_gas, phases_gas = find_W(W, 0)
    K_gas = W_gas / zi

    # Second, to be conservative, do a test liquid-like composition
    W = zi / K
    W_liq, tm_liq, phases_liq = find_W(W, 1)
    K_liq = zi / W_liq

    if phases_gas > 1 and phases_liq > 1:
        if tm_gas < tm_liq:
            # This is probably a gas-like mixture
            K = K_gas
            phases = 2
        else:
            # This is probably a liquid-like mixture
            K = K_liq
            phases = 2
    elif phases_gas > 1:
        # This is proably a gas-like mixture
        K = K_gas
        phases = 2
    elif phases_liq > 1:
        # This is probably a liquid-like mixture
        K = K_liq
        phases = 2
    else:
        # This is a single-phase mixture
        K = np.ones(K.shape)
        phases = 1

    # Return the results
    return K, phases


def gas_liq_eq(m, M, K):
    # Compute the mole fraction of the total mixture (Michelsen and Mollerup, 2007)
    moles = m / M
    zi = moles / np.sum(moles)

    # Define the Rachford-Rice equation for beta as gas fraction.
    def f_gas(beta_g):
        return np.sum(zi * (K - 1.) / (1. + beta_g * (K - 1.)))

    def f_gas_prime(beta_g):
        return -np.sum(zi * (K - 1.) ** 2 / (1. + beta_g * (K - 1.)) ** 2)

    def f_liq(beta_l):
        return np.sum(zi * (K - 1.) / (K - beta_l * (K - 1.)))

    def f_liq_prime(beta_l):
        return np.sum(zi * (K - 1.) ** 2 / (K - beta_l * (K - 1.)) ** 2)

    if np.sum(zi * K) - 1. <= 0.:
        # This is subcooled liquid, beta = 0.
        beta = 0.

    elif 1. - np.sum(zi / K) > 0.:
        # This is superheated gas, beta = 1.
        beta = 1.

    else:
        # This is a two-phase mixture
        beta_min = 0.
        beta_max = 1.

        for i in range(len(K)):
            if K[i] >= 1.:
                beta_min = np.max([beta_min, (K[i] * zi[i] - 1.) / (K[i] - 1.)])
            else:
                beta_max = np.min([beta_max, (1. - zi[i]) / (1. - K[i])])

        beta = 0.5 * (beta_min + beta_max)

        if f_gas(beta) > 0.:
            # Solution will have excess gas
            eqn = 1.
            beta_var = beta

        else:
            # Solution will have excess liquid
            eqn = 0.
            beta_var = 1. - beta
            beta_min_hold = beta_min
            beta_min = 1. - beta_max
            beta_max = 1. - beta_min_hold

        # Set up an iterative solution to find beta_var using the optimal root-finding equation
        tol = 1.e-8
        err = 1.

        while err > tol:
            # Store the current value of beta
            beta_old = beta_var

            # Step iv on page 254:  Perform one iteration of Newton's method
            # and narrow the possible range of the solution for beta
            if eqn > 0.:
                # Use the equations for excess gas
                g = f_gas(beta_var)
                gp = f_gas_prime(beta_var)
                beta_new = beta_var - g / gp

                # Update bounds on beta per criteria in step iv on page 254
                if g > 0:
                    beta_min = beta_var
                else:
                    beta_max = beta_var

            else:
                # Use the equations for excess liqiud
                g = f_liq(beta_var)
                gp = f_liq_prime(beta_var)
                beta_new = beta_var - g / gp

                # Update bounds on beta per criteria in step iv on page 254
                if g > 0:
                    beta_max = beta_var
                else:
                    beta_min = beta_var

            # Step v on page 254:  Select best update for beta
            if beta_max >= beta_new >= beta_min:
                beta_var = beta_new
            else:
                beta_var = 0.5 * (beta_min + beta_max)

            # Step vi on page 254:  Check for convergence.
            err = np.abs(beta_var - beta_old)

        # Get the final value of beta, the gas mole fraction
        if eqn > 0.:
            # We found beta
            beta = beta_var
        else:
            # We found beta_l
            beta = 1. - beta_var

    # Return the solution for gas and liquid mole fractions based on the converged value of beta
    return np.array([zi * K / (1. + beta * (K - 1.)), zi / (1. + beta * (K - 1.))]), beta


def get_k_bio(composition):
    dict_k_bio = {'Aromatics1': 0.23 / 24, 'Saturates1': 0.24 / 24,
                  'Aromatics2': 0.29 / 24, 'Saturates2': 0.12 / 24,
                  'Aromatics3': 0.28 / 24, 'Saturates3': 0.06 / 24,
                  'Aromatics4': 0.06 / 24, 'Saturates4': 0.06 / 24,
                  'Aromatics5': 0.28 / 24, 'Saturates5': 0.06 / 24,
                  'Aromatics6': 0.18 / 24, 'Saturates6': 0.05 / 24,
                  'Aromatics7': 0.15 / 24, 'Saturates7': 0.04 / 24,
                  'Aromatics8': 0.10 / 24, 'Saturates8': 0.04 / 24,
                  'Aromatics9': 0.10 / 24, 'Saturates9': 0.04 / 24,
                  'Aromatics10': 0.10 / 24, 'Saturates10': 0.04 / 24,
                  'Resins': 0, 'Asphaltenes': 0,
                  'methane': 1 / 24, 'ethane': 0.95 / 24, 'propane': 0.9 / 24,
                  'isobutane': 0.85 / 24, 'n-butane': 0.8 / 24,
                  'isopentane': 0.75 / 24, 'n-pentane': 0.7 / 24
                  }

    k_bio = []

    for i in composition:
        k_bio.append(dict_k_bio[i])

    return k_bio


def get_dict_k_bio():
    dict_k_bio = {'Aromatics1': {'k_bio': 0.23 / 24}, 'Saturates1': {'k_bio': 0.24 / 24},
                  'Aromatics2': {'k_bio': 0.29 / 24}, 'Saturates2': {'k_bio': 0.12 / 24},
                  'Aromatics3': {'k_bio': 0.28 / 24}, 'Saturates3': {'k_bio': 0.06 / 24},
                  'Aromatics4': {'k_bio': 0.06 / 24}, 'Saturates4': {'k_bio': 0.06 / 24},
                  'Aromatics5': {'k_bio': 0.28 / 24}, 'Saturates5': {'k_bio': 0.06 / 24},
                  'Aromatics6': {'k_bio': 0.18 / 24}, 'Saturates6': {'k_bio': 0.05 / 24},
                  'Aromatics7': {'k_bio': 0.15 / 24}, 'Saturates7': {'k_bio': 0.04 / 24},
                  'Aromatics8': {'k_bio': 0.10 / 24}, 'Saturates8': {'k_bio': 0.04 / 24},
                  'Aromatics9': {'k_bio': 0.10 / 24}, 'Saturates9': {'k_bio': 0.04 / 24},
                  'Aromatics10': {'k_bio': 0.10 / 24}, 'Saturates10': {'k_bio': 0.04 / 24},
                  'Resins': {'k_bio': 0}, 'Asphaltenes': {'k_bio': 0},
                  'methane': {'k_bio': 1 / 24}, 'ethane': {'k_bio': 0.95 / 24}, 'propane': {'k_bio': 0.9 / 24},
                  'isobutane': {'k_bio': 0.85 / 24}, 'n-butane': {'k_bio': 0.8 / 24},
                  'isopentane': {'k_bio': 0.75 / 24}, 'n-pentane': {'k_bio': 0.7 / 24}
                  }

    return dict_k_bio

