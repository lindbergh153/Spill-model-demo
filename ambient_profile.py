"""
Ambient Profile Data Handling
=============================

Classes for loading and interpolating oceanographic ambient data
from CSV files (1D profiles) and NetCDF files (3D/4D fields).

Classes
-------
Profile1d : 1D vertical CTD profile from CSV
Profile3d : 3D/4D spatiotemporal data from NetCDF with RegularGridInterpolator

"""

from __future__ import annotations

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.interpolate import interp1d, RegularGridInterpolator

import seawater


class Profile1d(object):
    def __init__(self, file):
        """
        This module is to read ambient data files (e.g., CTD profiles) in csv format

        :param file: csv
            A file contains water column data, including temperature, salinity,
            and current velocity in x and y directions
        Attributes
        ----------
        data: pandas.core.frame.DataFrame
            data derived from csv file
        z_min: float
            minimal depth of the data range
        z_max: float
            maximal depth of the data range

        """
        self.data = pd.read_csv(file)
        self.z_min = min(self.data['water_depth(m)'])
        self.z_max = max(self.data['water_depth(m)'])

    def _compute_pressure(self, z, T, S, fs_loc):
        """
        Compute the pressure profile

        :param z: ndarray
            Array of depths (m)
        :param T: ndarray
            Array of temperatures (K) at the corresponding depths in `z`.
        :param S: ndarray
            Array of salinities (psu) at the corresponding depth in `z`.
        :param fs_loc: integer (0 or -1)
            Index to the location of the free-surface in the `z`-array.
            0 corresponds to the first element of `z`, -1 corresponds to the last element.

        :return: P : ndarray
            Array of pressures (Pa) at the corresponding depth in `z`.

        """
        # Get the sign of the z-data for the midpoint of the dataset
        z_sign = int(np.sign(z[len(z) // 2]))

        # Initialize an array for storing the pressures
        P0 = 101325.0
        g = 9.81
        P = np.zeros(z.shape)

        # Find the free surface in the z-data
        if fs_loc == -1:
            depth_idxs = range(len(z) - 2, -1, -1)
            idx_0 = len(z) - 1
        else:
            depth_idxs = range(1, len(z))
            idx_0 = 0

        # Compute the pressure at the free surface
        P[idx_0] = P0 + seawater.density(T[0], S[0], P0) * g * z_sign * z[idx_0]

        # Compute the pressure at the remaining depths
        for i in depth_idxs:
            P[i] = P[i - z_sign] + seawater.density(T[i - z_sign], S[i - z_sign],
                                                    P[i - z_sign]) * g * (z[i] - z[i - z_sign]) * z_sign

        return P

    def get_values(self, current_z):
        """
        get the water column data required by near-field modeling

        :param current_z: float
            Depth (m) at which data are desired.

        :return: P: float
            ambient pressure (Pa)
        :return: Ta: float
            ambient temperature (C)
        :return: Sa: float
            ambient salinity (1e-3)
        :return: Ua: float
            ambient water velocity in x direction (m/s)
        :return: Va: float
            ambient velocity in y direction (m/s)
        :return: Wa: float
            ambient velocity in z direction (m/s)

        """
        u = self.data['eastward water velocity(m s-1)']
        v = self.data['northward water velocity(m s-1)']
        temp = self.data['temperature(K)']
        so = self.data['salinity(1e-3)']
        depth = self.data['water_depth(m)']
        pressure = self._compute_pressure(depth, temp, so, 0)
        ua = interp1d(depth, u.transpose())
        va = interp1d(depth, v.transpose())
        temp_a = interp1d(depth, temp.transpose())
        sa = interp1d(depth, so.transpose())
        pressure_a = interp1d(depth, pressure)

        try:
            P = pressure_a(current_z)
            Ta = temp_a(current_z)
            Sa = sa(current_z)
            Ua = ua(current_z)
            Va = va(current_z)
        except:
            if current_z < self.z_min:
                P = 101325
                Ta = temp_a(self.z_min)
                Sa = sa(self.z_min)
                Ua = ua(self.z_min)
                Va = va(self.z_min)
            else:
                P = pressure_a(self.z_max)
                Ta = temp_a(self.z_max)
                Sa = sa(self.z_max)
                Ua = ua(self.z_max)
                Va = va(self.z_max)
                print('the depth desired is deeper than the maximum depth of the current data, '
                      'here the profile in {} meters is used'.format(round(self.z_max, 2)))

        P, Ta, Sa, Ua, Va, Wa = float(P), float(Ta), float(Sa), float(Ua), float(Va), 0.

        return P, Ta, Sa, Ua, Va, Wa


class Profile3d(Profile1d):
    def __init__(self, data_2d, data_3d, data_wind):
        super(Profile3d, self).__init__(file=data_2d)
        self.data_2d = data_2d
        if isinstance(data_3d, str):
            self.current_3d, self.wind_2d, = Dataset(data_3d), Dataset(data_wind)
        elif isinstance(data_3d, Dataset):
            self.current_3d, self.wind_2d, = data_3d, data_wind
        else:
            raise TypeError("Input data type is wrong")

        self.time_interval = self.get_time_interval()

    def profile_1d(self, z):
        Pa, Ta, Sa, Ua, Va, Wa = Profile1d.get_values(self, z)
        return Pa, Ta, Sa, Ua, Va, Wa

    def velocity_field(self):

        time_current = self.current_3d.variables['time'][:]
        depth_current = self.current_3d.variables['depth'][:]
        lat_current = self.current_3d.variables['lat'][:]
        lon_current = self.current_3d.variables['lon'][:]
        raw_water_u = self.current_3d.variables['water_u'][:]
        raw_water_v = self.current_3d.variables['water_v'][:]
        interpolation_water_u = RegularGridInterpolator((time_current, depth_current, lat_current, lon_current),
                                                        raw_water_u)
        interpolation_water_v = RegularGridInterpolator((time_current, depth_current, lat_current, lon_current),
                                                        raw_water_v)

        time_wind = self.wind_2d.variables['time'][:]
        lat_wind = self.wind_2d.variables['latitude'][:]
        lon_wind = self.wind_2d.variables['longitude'][:]
        raw_wind_u = self.wind_2d.variables['u10'][:]
        raw_wind_v = self.wind_2d.variables['v10'][:]

        interpolation_wind_u = RegularGridInterpolator((time_wind, lat_wind, lon_wind), raw_wind_u)
        interpolation_wind_v = RegularGridInterpolator((time_wind, lat_wind, lon_wind), raw_wind_v)

        fields = [interpolation_water_u, interpolation_water_v, interpolation_wind_u, interpolation_wind_v]

        return fields

    def data_range(self):
        min_time_current = float(self.current_3d.variables['time'][0])
        max_time_current = float(self.current_3d.variables['time'][-1])
        min_depth_current = float(self.current_3d.variables['depth'][0])
        max_depth_current = float(self.current_3d.variables['depth'][-1])
        min_lat_current = min(self.current_3d.variables['lat'][:])
        max_lat_current = max(self.current_3d.variables['lat'][:])
        min_lon_current = min(self.current_3d.variables['lon'][:])
        max_lon_current = max(self.current_3d.variables['lon'][:])

        min_time_wind = float(self.wind_2d.variables['time'][0])
        max_time_wind = float(self.wind_2d.variables['time'][-1])
        min_lat_wind = min(self.wind_2d.variables['latitude'][:])
        max_lat_wind = max(self.wind_2d.variables['latitude'][:])
        min_lon_wind = min(self.wind_2d.variables['longitude'][:])
        max_lon_wind = max(self.wind_2d.variables['longitude'][:])

        profile = [min_time_current, max_time_current, min_depth_current, max_depth_current, min_lat_current,
                      max_lat_current, min_lon_current, max_lon_current, min_time_wind, max_time_wind, min_lat_wind,
                      max_lat_wind, min_lon_wind, max_lon_wind]

        return profile

    def get_time_interval(self):
        time_unit = self.current_3d.variables['time'].units[0:5]
        if time_unit == 'hours':
            time_interval = 3600
        else:
            raise Exception('time interval is wrong')

        return time_interval

    def return_all(self):
        profile1d = Profile1d(self.data_2d)
        fields = self.velocity_field()
        profile = self.data_range()
        time_interval = self.get_time_interval()
        all_profiles = fields + profile + [time_interval, profile1d.get_values]

        return all_profiles