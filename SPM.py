"""
Single Parcel Model (SPM)
=========================

Far-field transport model for tracking oil parcels (underwater particle
clouds and surface slicks) through advection-diffusion processes.

Classes
-------
Model_parcel : Main simulation class for particle cloud and slick transport
ModelParams : Container for fixed model parameters

References
----------
Johansen, Ø. (2003). Development and verification of deep-water blowout
    models. Marine Pollution Bulletin, 47(9-12), 360-368.

"""

from __future__ import annotations

from SPM_functions import *
from SPM_utilities import get_mb
from ambient_profile import Profile1d


class Model_parcel(object):
    def __init__(self, profile, cloud, X0, near_data, dt02_current, dt02_wind):
        """
        Single Parcel Model (SPM)
        Class to calculate the trajectory and fate of both particle cloud
        in the water column and slick on the sea surface.

        :param profile: list
            A data ensemble required by far-field modeling, including:
            1) interpolation_water_u, 2) interpolation_water_v, 3) interpolation_wind_u,
            4) interpolation_wind_v, 5) min_time_current, 6) max_time_current, 7) min_depth_current,
            8) max_depth_current, 9) min_lat_current,  10) max_lat_current, 11) min_lon_current,
            12) max_lon_current, 13) min_time_wind, 14) max_time_wind, 15) min_lat_wind,
            16) max_lat_wind, 17) min_lon_wind, 18) max_lon_wind, 19) time_interval,
            20) profile1d.get_values: P, Ta, Sa, Ua, Va, Wa.
        :param cloud: SPM_utilities.ParticleCloud
            Object describing the properties of the cloud (a collection of particles with the same size).
        :param X0: list
            The initial location of a particle cloud: [Lat, lon, depth]
        :param near_data: str
            A path of a csv file;
            Vertical profile of ambient water temperature, current velocity, and salinity

        Attributes
        ----------
        t_sub :  ndarray
            Times (h) associated with the state space of the particle cloud
        y_sub :  ndarray
            State space along the trajectory of the particle cloud
        t_sur :  ndarray
            Times (h) associated with the state space of the slick
        y_sur :  ndarray
            State space along the trajectory of the slick

        """

        self.profile = profile
        self.p = ModelParams(self.profile, near_data, dt02_current, dt02_wind)
        self.dt02_wind = dt02_wind
        self.cloud = cloud
        self.X0 = X0

    def simulate(self, start_time_FF, dt23, Dxyz_sur, Dxyz_sub, dt_sub, dt_sur,
                 para_current, para_wind, shore_polygon):
        """
        Simulate the trajectory and fate of a particle cloud from given initial conditions

        :param start_time_FF: float
            Start time of far-field simulation (hours). This time is related to the initial time of the current data, e.g.,
            a start time equaling 0 means the input of time is the initial value of time properties in NetCDF data.
        :param dt_end_time: float
            Hours between Start time and End time of simulation.
        :param Dxyz_sur: float
            Diffusion coefficients for surface particles in xyz-direction
        :param Dxyz_sub: float
            Diffusion coefficients for subsurface cloud in xyz-direction
        :param dt_sub: float
            Time step of far-field model for underwater oil (hours)
        :param dt_sur: float
            Time step of far-field model for surfaced oil (hours)
        :param para_current: float, default=1
            current drift coefficient
        :param para_wind: float, default=0.03
            wind drift coefficient

        :return FF_result: list
            A list contains the output of SPM
            If the FF_result has 1) two elements: [state_variables, particle cloud];
            2) three elements: [state_variables, particle cloud, slick].

        """

        # Get the initial conditions and time for the subsurface oil simulation
        t0_sub, y0_sub = ic_underwater(self.cloud, start_time_FF, self.X0)

        print('\n--- DWOSM Far-field model ---')
        print('--- Particle cloud in the water column ---')

        # Run the simulation for a particle cloud
        self.t_sub, self.y_sub, self.cloud = calculation_underwater(self.profile, self.cloud,
                                                                    self.p, t0_sub, y0_sub,
                                                                    dt23, dt_sub, Dxyz_sub)

        # Add times (h) associated with the state space of the particle cloud to cloud's attribute
        self.cloud.time = self.t_sub

        # When Cloud reaches the surface and the particles in Cloud are liquid phase,
        # continue to run the simulation for Slick
        if not self.cloud.underwater and self.cloud.q_type == 1:
            # Get the inputs required by the surface oil simulation,
            # i.e., initial time and location
            t0_sur = self.t_sub[-1]
            loc_sur = self.y_sub[-1, 0:3]

            # Get the initial conditions for the surface oil simulation
            t0_sur, y0_sur, slick = ic_surface(self.cloud, t0_sur, self.p, self.profile, loc_sur)
            print('--- Oil slick at the sea surface ---')

            # Run the simulation of the surface oil
            self.t_sur, self.y_sur, self.slick = calculation_surface(self.profile, slick, self.p, t0_sur, y0_sur,
                                                                     dt23, dt_sur,
                                                                     Dxyz_sur, para_current, para_wind,
                                                                     shore_polygon=shore_polygon)

            # Process the records of Slick and get the record of the
            # evaporated and dispersed mass along with time
            self.slick.evap_list, self.slick.disp_list, self.slick.time, self.slick.iter_solver = \
                get_mb(self.slick.iter_solver, self.slick.evap_list, self.slick.disp_list, self.slick.time, self.y_sur)

        # Combine the simulation results of FFM
        FF_result = self.merge_results()

        return FF_result

    def merge_results(self):
        """
        Merge the outputs of simulations for Cloud and Slick into a list

        :return FF_result: list
            A list contains the output of SPM
            If the FF_result has 1) two elements: [state_variables, particle cloud];
            2) three elements: [state_variables, particle cloud, slick].

        """

        # When Cloud reaches the surface and the particles in Cloud are liquid phase,
        # store both cloud and slick into a list since slick is transited from cloud
        if not self.cloud.underwater and self.cloud.q_type == 1:
            # Extract times associated with the state space of the particle cloud
            t_sub = self.t_sub

            # Extract the state space of the particle cloud
            X_sub = self.y_sub[:, 0:3]
            m_sub = self.y_sub[:, 3:-1]

            # Extract times associated with the state space of the slick
            t_sur = self.t_sur

            # Extract the state space of the slick
            X_sur = self.y_sur[:, 0:3]
            m_sur = self.y_sur[:, 4:-1]

            # Vertically concatenate times, locations, and mass for subsurface and surface oils
            t = np.hstack((t_sub, t_sur)).reshape(-1, 1)
            X = np.vstack((X_sub, X_sur))
            m = np.vstack((m_sub, m_sur))

            # Horizontally concatenate the above-mentioned data
            state_variables = np.hstack((t, X, m))

            # Add state variables, cloud, and slick into a list
            FF_result = [state_variables, self.cloud, self.slick]

        else:
            # When Cloud didn't reach the surface or the particles in Cloud are gas phase,
            # no need to concatenate them since only Cloud get involved
            t_sub = self.t_sub.reshape(-1, 1)
            X_sub = self.y_sub[:, 0:3]
            m_sub = self.y_sub[:, 3:-1]

            # Horizontally concatenate the above-mentioned data
            state_variables = np.hstack((t_sub, X_sub, m_sub))

            # Add state variables and cloud into a list
            FF_result = [state_variables, self.cloud]

        return FF_result


class ModelParams(object):
    def __init__(self, profile, near_data, dt02_current, dt02_wind):
        """
        Fixed model parameters for SPM

        :param profile: list
            A data ensemble required by far-field modeling, including:
            1) interpolation_water_u, 2) interpolation_water_v, 3) interpolation_wind_u,
            4) interpolation_wind_v, 5) min_time_current, 6) max_time_current, 7) min_depth_current,
            8) max_depth_current, 9) min_lat_current,  10) max_lat_current, 11) min_lon_current,
            12) max_lon_current, 13) min_time_wind, 14) max_time_wind, 15) min_lat_wind,
            16) max_lat_wind, 17) min_lon_wind, 18) max_lon_wind, 19) time_interval,
            20) P, 21) Ta, 22) Sa, 23) Ua, 24) Va, 25) Wa.
        :param near_data: csv
            Vertical profile of ambient water temperature, current velocity, and salinity

        Attributes
        ----------
        t_min_current : float
            Minimal time in current data
        t_max_current : float
            Maximum time in current data
        z_min : float
            Minimal depth in current data
        z_max : float
            Maximum depth in current data
        y_min_current : float
            Minimal latitude in current data
        y_max_current : float
            Maximum latitude in current data
        x_min_current : float
            Minimal longitude in current data
        x_max_current : float
            Maximum longitude in current data
        time_interval : float
            The value for transiting hour to second (3600)
        left_bound : float
            Mutual left boundary of velocity fields
        right_bound : float
            Mutual right boundary of velocity fields
        upper_bound : float
            Mutual upper boundary of velocity fields
        lower_bound : float
            Mutual lower boundary of velocity fields
        rho_r : float
            Reference density (kg/m^3) evaluated at mid-depth of the water body.
        P_sur : float
            Surface pressure (Pa)
        T_sur : float
            Surface temperature (K)
        S_sur : float
            Surface salinity (PSU)
        Ua_sur : float
            Surface current velocity in x-direction (m/s)
        Va_sur : float
            Surface current velocity in y-direction (m/s)
        g : float
            Acceleration of gravity (m/s^2)
        R : float
            Ideal gas constant (J/mol/K)

        """

        # Extract information from the data collection
        self.t_min_current = profile[4]
        self.t_max_current = profile[5]
        self.z_min = profile[6]
        self.z_max = profile[7]
        self.y_min_current = profile[8]
        self.y_max_current = profile[9]
        self.x_min_current = profile[10]
        self.x_max_current = profile[11]
        self.t_min_wind = profile[12]
        self.t_max_wind = profile[13]
        self.y_min_wind = profile[14]
        self.y_max_wind = profile[15]
        self.x_min_wind = profile[16]
        self.x_max_wind = profile[17]
        self.time_interval = profile[18]
        self.dt02_current = dt02_current
        self.dt02_wind = dt02_wind

        # Set the mutual boundary of current and wind datasets
        self.left_bound = max(self.x_min_current, self.x_min_wind)
        self.right_bound = min(self.x_max_current, self.x_max_wind)
        self.upper_bound = min(self.y_max_current, self.y_max_wind)
        self.lower_bound = max(self.y_min_current, self.y_min_wind)

        # Compute the density at mid-depth of the water body.
        vertical_profile = Profile1d(near_data)
        z_ave = self.z_max - (self.z_max - self.z_min) / 2.
        P, T, S, Ua, Va, Wa = vertical_profile.get_values(z_ave)
        self.rho_r = seawater.density(T, S, P)

        # Store the properties of surface seawater
        self.P_sur, self.T_sur, self.S_sur, self.Ua_sur, self.Va_sur, Wa_sur = \
            vertical_profile.get_values(float(self.z_min))

        self.g = 9.81
        self.R = 8.314510