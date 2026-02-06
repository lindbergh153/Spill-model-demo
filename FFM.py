"""
Far-Field Model (FFM) Main Module
=================================

Ensemble simulation framework for far-field oil transport using
multiprocessing parallelization.

Classes
-------
Far_Field : Main simulation controller with parallel execution

"""

from __future__ import annotations
import copy
import csv
import multiprocessing
from multiprocessing import Pool

import matplotlib.pyplot as plt
import matplotlib
import geopandas as gpd
from netCDF4 import Dataset
from matplotlib_scalebar.scalebar import ScaleBar

from SPM import Model_parcel
from SPM_utilities import *
from ambient_profile import Profile3d
from conversion_functions import convert_loc, get_delta_t02
from gridmesh import plot_cuboid
from shore_module import get_shore_polygon, Point
import pandas as pd
from FFM_utilities import (separate_oil_gas, get_exit_time, get_release_time,
                           assemble_clouds, initial_cloud_mass, duplicate_gas_results)


class FFM_API(object):
    """
    The Far-Field Model (FFM) aims to predict behaviors of petroleum particles.
    This class is an API of Plume model module.

    """

    def __init__(self, near_data, current_data, wind_data, shore_data, particles, initial_location, SSDI, spill_image):
        """
        :param near_data: str
            A path of a csv file;
            Vertical profile of ambient water temperature, current velocity, and salinity
        :param current_data: str
            A path of a NetCDF file;
            3D current velocity field
        :param wind_data: str
            A path of a NetCDF file;
            2D wind velocity field
        :param shore_data: str
            A path of a NetCDF file;
            2D wind velocity field
        :param particles: list
            A list contains multiple PlumeParticle exiting from the plume
        :param initial_location: list
            Coordinates and depth of the release point, [latitude, longitude, depth (m)]
        :param SSDI: bool, default=False
            A flag used to indicate whether to apply SSDI
        Attributes
        ----------
        FF_results : list
            A list contains the output of SPM.
            If the FF_result has 1) two elements: [state_variables, particle cloud];
            2) three elements: [state_variables, particle cloud, slick].

        """
        self.near_data = near_data
        self.current_data = current_data
        self.wind_data = wind_data
        self.shore_data = shore_data
        self.initial_location = initial_location
        self.particles = particles
        self.SSDI = SSDI
        self.spill_image = spill_image
        self.FF_results = []

    def simulate(self, start_time, end_time, release_duration, release_interval,
                 Dxy_sub, Dz_sub, Dxy_sur, para_current, para_wind, dt_sub, dt_sur):
        """
        Run an SPM simulation for the present conditions

        :param start_time: datetime.datetime
            Start time of simulation (Y/M/D/H/M).
        :param end_time: datetime.datetime
            End time of simulation (Y/M/D/H/M).
        :param release_duration: datetime.datetime
            Duration of an oil/gas blowout.
        :param release_interval: float
            The time interval that transfers the output of the near-field model
            as the input to the far-field model (hours).
        :param Dxy_sub: float
            Horizontal diffusion coefficients for subsurface particles in the x- and y-direction
        :param Dz_sub: float
            Vertical diffusion coefficients for subsurface particles in the z-direction
        :param Dxy_sur: float
            Horizontal diffusion coefficients for oil slicks in the x- and y-direction
        :param para_current: float, default=1
            current drift coefficient
        :param para_wind: float, default=0.03
            wind drift coefficient
        :param dt_sub: float
            Time step of far-field model for underwater oil (hours)
        :param dt_sur: float
            Time step of far-field model for surfaced oil (hours)

        """
        self.start_time = start_time
        self.end_time = end_time
        self.release_duration = release_duration
        self.dt_sur = dt_sur
        self.dt_sub = dt_sub
        self.Dxy_sur = Dxy_sur
        self.para_current = para_current
        self.para_wind = para_wind
        self.release_interval = release_interval

        gas_bubbles, oil_droplets = separate_oil_gas(self.particles)

        all_gas_inputs = assemble_gas_inputs(self.near_data, self.current_data, self.wind_data, gas_bubbles,
                            self.initial_location, Dxy_sub, Dz_sub,
                            start_time, end_time, release_interval, dt_sub)

        # create process pools
        pool = Pool(multiprocessing.cpu_count())

        if len(gas_bubbles) != 0:
            gas_bubbles_results = pool.starmap(self.SPM_API, all_gas_inputs)
            self.FF_results += gas_bubbles_results
        # generate a list, each element contains the inputs required by far-field model
        all_oil_inputs = assemble_oil_inputs(self.near_data, self.current_data, self.wind_data,
                                     oil_droplets, self.initial_location, Dxy_sub, Dz_sub, self.Dxy_sur,
                                     self.start_time, self.release_duration, self.end_time, self.release_interval,
                                     self.para_current, self.para_wind, self.dt_sub, self.dt_sur, self.shore_data)

        # # receive the results from process pools
        self.FF_results += pool.starmap(self.SPM_API, all_oil_inputs)
        # from plot_gas_depth import extract_gas_data
        # extract_gas_data(self.FF_results, self.particles)

        pool.close()
        pool.join()

    def add_gas_result(self):
        gas_results = []
        for element in self.FF_results:
            if len(element) == 2:
                gas_results.append(element)
        if len(gas_results) != 0:
            new_gas_results = duplicate_gas_results(gas_results, self.start_time, self.end_time, self.release_interval)
            print('Before adding gas components, the total number of particles {0}'.format(len(self.FF_results)))
            self.FF_results += new_gas_results
            print('After adding gas components, the total number of particles {0}'.format(len(self.FF_results)))

    def SPM_API(self, cloud, profiles, initial_location, Dxy_sub, Dz_sub, Dxy_sur,
                particle_release_time, dt_end_time, para_current, para_wind,
                dt_sub, dt_sur, dt02_current, dt02_wind, shore_polygon=None):
        """
        An API to run an SPM simulation (aims to a single cloud/slick)

        :param cloud: SPM_utilities.ParticleCloud
             Object describing the properties of the cloud (a collection of particles with the same size).
        :param profiles: list
            A data ensemble required by far-field modeling, including:
            1) interpolation_water_u, 2) interpolation_water_v, 3) interpolation_wind_u,
            4) interpolation_wind_v, 5) min_time_current, 6) max_time_current, 7) min_depth_current,
            8) max_depth_current, 9) min_lat_current,  10) max_lat_current, 11) min_lon_current,
            12) max_lon_current, 13) min_time_wind, 14) max_time_wind, 15) min_lat_wind,
            16) max_lat_wind, 17) min_lon_wind, 18) max_lon_wind, 19) time_interval,
            20) profile1d.get_values: P, Ta, Sa, Ua, Va, Wa.
        :param initial_location: list
            The initial location of a particle cloud: [Lat, lon, depth]
        :param Dxy_sub: float
            Horizontal diffusion coefficients for subsurface particles in the x- and y-direction
        :param Dz_sub: float
            Vertical diffusion coefficients for subsurface particles in the z-direction
        :param Dxy_sur: float
            Horizontal diffusion coefficients for oil slicks in the x- and y-direction
        :param particle_release_time: float
            The time (hour) that oil released from the near field after start time
        :param dt_end_time: float
            Hours between Start time and End time of simulation.
        :param para_current: float, default=1
            current drift coefficient
        :param para_wind: float, default=0.03
            wind drift coefficient
        :param dt_sub: float
            Time step of far-field model for underwater oil (hours)
        :param dt_sur: float
            Time step of far-field model for surfaced oil (hours)

        :return FF_result: list
            A list contains the output of SPM
            If the FF_result has 1) two elements: [state_variables, particle cloud];
            2) three elements: [state_variables, particle cloud, slick].

        """

        # initialize the SPM
        SPM = Model_parcel(profiles, cloud, initial_location,
                           self.near_data, dt02_current, dt02_wind)

        # assemble diffusion coefficients in three direction
        Dxyz_sub, Dxyz_sur = [Dxy_sub, Dxy_sub, Dz_sub], [Dxy_sur, Dxy_sur, 0]

        # run the simulation of SPM
        FF_result = SPM.simulate(particle_release_time, dt_end_time, Dxyz_sur, Dxyz_sub,
                                 dt_sub, dt_sur, para_current, para_wind, shore_polygon)

        return FF_result

    def plot_2d(self):
        """
        Plot the 2D figures for far-field modeling

        """

        # Initialize the containers of clouds and slicks' location
        x_sur, y_sur, z_sur = [], [], []
        x_sub, y_sub, z_sub = [], [], []

        # Get the location of the spill site
        x0, y0, z0 = self.initial_location[1], self.initial_location[0], self.initial_location[2]

        # Extract the results from the outputs of FFM
        for i in self.FF_results:
            # Extract the results when FF_results contain state variables and ParticleClouds
            if len(i) == 2:
                x_sub.append(i[1].x)
                y_sub.append(i[1].y)
                z_sub.append(i[1].z)
            # Extract the results when FF_results contain state variables, ParticleClouds, and Slicks
            elif len(i) == 3:
                x_sur.append(i[2].x)
                y_sur.append(i[2].y)
                z_sur.append(i[2].z)
            else:
                raise Exception('something wrong with FF_results (FFM_API.plot_2d)')

        plt.style.use('ggplot')
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 6))

        # x-z view
        axes[0, 0].scatter(x_sur, z_sur, s=5, color='r')
        axes[0, 0].scatter(x_sub, z_sub, s=5, color='b')
        axes[0, 0].scatter(x0, z0, s=50, marker='*', color='r')
        axes[0, 0].set_xlabel('Lon')
        axes[0, 0].invert_yaxis()
        axes[0, 0].set_ylabel('Depth (m)')

        # y-z view
        axes[0, 1].scatter(y_sur, z_sur, s=5, color='r')
        axes[0, 1].scatter(y_sub, z_sub, s=5, color='b')
        axes[0, 1].scatter(y0, z0, s=50, marker='*', color='red')
        axes[0, 1].set_xlabel('Lat')
        axes[0, 1].invert_yaxis()
        axes[0, 1].set_ylabel('Depth (m)')

        # x-y view
        axes[1, 0].scatter(x_sur, y_sur, s=5, color='r')
        axes[1, 0].scatter(x_sub, y_sub, s=5, color='b', alpha=0.8)
        axes[1, 0].scatter(x0, y0, s=50, marker='*', color='red')
        axes[1, 0].set_xlabel('Lat')
        axes[1, 0].set_ylabel('Lon')
        axes[1, 0].set_xlim(-50, -49.)
        axes[1, 0].set_ylim(49.4, 50.2)

        plt.show()

    def plot_xy_2d_fc(self):
        """
        Plot a 2D figure for far-field modeling

        """
        # Initialize the containers of clouds and slicks' location
        x_sur, y_sur, A = [], [], []
        lat, lon, z0 = self.initial_location

        # Extract the results from the outputs of FFM
        for i in self.FF_results:
            if len(i) == 3:
                if sum(i[2].m) > 1e-4:
                    x_sur.append(i[2].x)
                    y_sur.append(i[2].y)
                    A.append(i[2].A)

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.set_title('{0} (CDT time)'.format(self.end_time), fontsize=15)

        mergedPolys = get_shore_polygon('data/coast_GOM_full.bna')
        gpd.GeoSeries(mergedPolys).plot(ax=ax)

        ax.scatter(x_sur, y_sur, s=0.2, color='r', alpha=0.7)
        ax.scatter(lon, lat, color='black', s=80, marker="X")

        gdf = gpd.read_file(self.spill_image)
        target_crs = 'EPSG:4326'
        gdf = gdf.to_crs(target_crs)
        for j in gdf.axes[1]:
            cap_j = j.upper()
            if cap_j == 'FORECAST':
                column_name = j
        category_mapping = {
            'FORECASTHEAVY': 'Heavy',
            'FORECASTMEDIUM': 'Medium',
            'FORECASTLIGHT': 'Light',
            'FORECASTUNCERTAINTY': 'Uncertainty'
        }
        gdf[column_name] = gdf[column_name].replace(category_mapping)

        gdf.plot(column=column_name, ax=ax, categorical=True, alpha=0.5, cmap=matplotlib.colormaps['viridis'],
                 legend=True, legend_kwds={"loc": "lower left", 'fontsize': 12})

        points = gpd.GeoSeries([Point(-86, 27), Point(-85, 27)], crs=4326)
        points = points.to_crs(32619)
        distance_meters = points[0].distance(points[1])
        ax.add_artist(ScaleBar(distance_meters, location="lower right", box_color="grey"))

        ax.set_xlim(-91, -85)
        ax.set_ylim(27, 30.5)
        ax.set_xlabel('Lon', fontsize=14)
        ax.set_ylabel('Lat', fontsize=14)
        ax.set_title('{0} (CDT time)'.format(self.end_time), fontsize=16)

        plt.tight_layout()
        # plt.show()
        # time_name = ''.join(char for char in str(self.end_time) if char.isdigit())
        time_name = ''.join(char for char in str(self.end_time) if char.isdigit())[4:-4]
        Dxy_sur = "{:.1e}".format(self.Dxy_sur)
        plt.savefig('trajectory0921/fc/{0}cur{1}wind{2}D{3}.jpg'.
                    format(time_name, self.para_current, self.para_wind, Dxy_sur), dpi=900)

        df = pd.DataFrame({"lon": x_sur, "lat": y_sur, 'area': A})
        df.to_csv('trajectory0921/fc/{0}cur{1}wind{2}D{3}.csv'.
                    format(time_name, self.para_current, self.para_wind, Dxy_sur), index=False)

    def plot_xy_2d_fp(self):
        """
        Plot a 2D figure for far-field modeling

        """
        # Initialize the containers of clouds and slicks' location
        x_sur, y_sur, A = [], [], []
        lat, lon, z0 = self.initial_location

        # Extract the results from the outputs of FFM
        for i in self.FF_results:
            if len(i) == 3:
                if sum(i[2].m) > 0:
                    x_sur.append(i[2].x)
                    y_sur.append(i[2].y)
                    A.append(i[2].A)

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.set_title('{0} (CDT time)'.format(self.end_time), fontsize=15)

        mergedPolys = get_shore_polygon('data/coast_GOM_full.bna')
        gpd.GeoSeries(mergedPolys).plot(ax=ax)

        ax.scatter(x_sur, y_sur, s=0.5, color='r', alpha=0.8)
        ax.scatter(lon, lat, color='black', s=80, marker="X")

        gdf = gpd.read_file(self.spill_image)
        gdf.plot(ax=ax, color='black', alpha=0.5)

        points = gpd.GeoSeries([Point(-86, 27), Point(-85, 27)], crs=4326)
        points = points.to_crs(32619)
        distance_meters = points[0].distance(points[1])
        ax.add_artist(ScaleBar(distance_meters, location="lower right", box_color="grey"))

        ax.set_xlim(-91, -85)
        ax.set_ylim(27, 30.5)

        plt.tight_layout()
        plt.show()
        # time_name = ''.join(char for char in str(self.end_time) if char.isdigit())[4:-4]
        # Dxy_sur = "{:.1e}".format(self.Dxy_sur)
        # plt.savefig('trajectory0925/fp/{0}cur{1}wind{2}D{3}.jpg'.
        #             format(time_name, self.para_current, self.para_wind, Dxy_sur), dpi=900)

        # df = pd.DataFrame({"lon": x_sur, "lat": y_sur, 'area': A})
        # df.to_csv('trajectory0925/fp/{0}cur{1}wind{2}D{3}.csv'.
        #             format(time_name, self.para_current, self.para_wind, Dxy_sur), index=False)

    def plot_xy_2d_shore(self):
        """
        Plot a 2D figure for far-field modeling

        """
        # Initialize the containers of clouds and slicks' location
        x_sur, y_sur, A = [], [], []
        lat, lon, z0 = self.initial_location

        # Extract the results from the outputs of FFM
        for i in self.FF_results:
            if len(i) == 3:
                if sum(i[2].m) > 0 and i[2].strand:
                    x_sur.append(i[2].x)
                    y_sur.append(i[2].y)
                    A.append(i[2].A)

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.set_title('{0} (CDT time)'.format(self.end_time), fontsize=15)

        mergedPolys = get_shore_polygon('data/coast_GOM_full.bna')
        gpd.GeoSeries(mergedPolys).plot(ax=ax)

        ax.scatter(x_sur, y_sur, s=0.5, color='r', alpha=0.8)
        ax.scatter(lon, lat, color='black', s=80, marker="X")

        gdf = gpd.read_file(self.spill_image)
        gdf.plot(ax=ax, color='black', alpha=0.5)

        points = gpd.GeoSeries([Point(-86, 27), Point(-85, 27)], crs=4326)
        points = points.to_crs(32619)
        distance_meters = points[0].distance(points[1])
        ax.add_artist(ScaleBar(distance_meters, location="lower right", box_color="grey"))

        ax.set_xlim(-91, -85)
        ax.set_ylim(27, 30.5)

        plt.tight_layout()
        # plt.show()
        time_name = ''.join(char for char in str(self.end_time) if char.isdigit())[4:-4]
        Dxy_sur = "{:.1e}".format(self.Dxy_sur)
        plt.savefig('trajectory/trajectory1005/{0}cur{1}wind{2}D{3}.jpg'.
                    format(time_name, self.para_current, self.para_wind, Dxy_sur), dpi=900)

        df = pd.DataFrame({"lon": x_sur, "lat": y_sur, 'area': A})
        df.to_csv('trajectory/trajectory1005/{0}cur{1}wind{2}D{3}.csv'.
                    format(time_name, self.para_current, self.para_wind, Dxy_sur), index=False)

    def plot_3d(self, show_mesh=False):
        """
        Plot a 3D figure for far-field modeling

        """

        # Get the location of the spill site
        x0, y0, z0 = self.initial_location[1], self.initial_location[0], self.initial_location[2]

        # Initialize the containers of clouds and slicks' location
        x_oil, y_oil, z_oil = [], [], []
        x_gas, y_gas, z_gas = [], [], []

        # # Extract the locations from the outputs of FFM (state variables)
        # # [0, 1:3]: last time's location
        for i in self.FF_results:
            if len(i) == 2 and i[1].q_type == 0:
                x_gas.append(i[1].x)
                y_gas.append(i[1].y)
                z_gas.append(i[1].z)
            elif len(i) == 2 and i[1].q_type == 1:
                x_oil.append(i[1].x)
                y_oil.append(i[1].y)
                z_oil.append(i[1].z)
            elif len(i) == 3:
                x_oil.append(i[2].x)
                y_oil.append(i[2].y)
                z_oil.append(i[2].z)

        ax = plt.axes(projection='3d')
        plt.gca().invert_zaxis()

        if len(x_gas) != 0:
            ax.scatter3D(x_gas, y_gas, z_gas, s=3, facecolors='none', edgecolors='b')
        ax.scatter3D(x_oil, y_oil, z_oil, s=3, color='red')
        ax.scatter3D(x0, y0, z0, marker='*', color='red', s=50)
        ax.set_xlabel('Lon', fontdict={'size': 18})
        ax.set_ylabel('Lat', fontdict={'size': 18})
        ax.set_zlabel('Depth', fontdict={'size': 18})
        ax.xaxis.labelpad = 20
        ax.yaxis.labelpad = 20
        ax.zaxis.labelpad = 20
        ax.set_xlim(-92, -84)
        ax.set_ylim(26, 30.5)
        ax.set_zlim(1500, 0)

        if show_mesh:
            lon_min, lon_max, lat_min, lat_max, depth_min, depth_max \
                = min(min(x_gas), min(x_oil)), \
                max(max(x_gas), max(x_oil)), \
                min(min(y_gas), min(y_oil)), \
                max(max(y_gas), max(y_oil)), \
                z0, 0
            bound = [lon_min, lon_max, lat_min, lat_max, depth_min, depth_max]
            plot_cuboid(ax, bound, bin_size=20, line_width=0.5)

        # ax.xaxis.set_tick_params(color='white')
        # ax.yaxis.set_tick_params(color='white')
        # ax.zaxis.set_tick_params(color='white')
        # ax.xaxis.set_ticklabels([])
        # ax.yaxis.set_ticklabels([])
        # ax.zaxis.set_ticklabels([])
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.tick_params(axis='both', which='minor', labelsize=18)
        elev = 35
        azim = -125
        ax.view_init(elev, azim)

        plt.show()

    def store_oil_budget_csv(self):
        from conversion_functions import remove_non_numeric

        parcel_result = []
        release_duration_hrs = self.release_duration.total_seconds() / 3600

        for i in self.FF_results:
            if len(i) == 2:
                m_Cloud_list = np.array(i[1].m_Cloud_list)
                if len(i[1].m_Cloud_list) == len(i[1].time):
                    time_sub = i[1].time
                else:
                    time_sub = i[1].time[0:-1]
                time_sub = time_sub.reshape(1, time_sub.shape[0])
                m_Cloud_list = m_Cloud_list.reshape(1, m_Cloud_list.shape[0])
                evap = np.zeros_like(time_sub)
                disp = np.zeros_like(time_sub)
                strand = np.zeros_like(time_sub)
                result = np.concatenate((time_sub, m_Cloud_list, evap, disp, strand), axis=0)

            elif len(i) == 3:
                m_Cloud_list = np.array(i[1].m_Cloud_list)
                if len(i[1].m_Cloud_list) == len(i[1].time):
                    time_sub = i[1].time
                else:
                    time_sub = i[1].time[0:-1]
                time_sub = time_sub.reshape(1, time_sub.shape[0])
                m_Cloud_list = m_Cloud_list.reshape(1, m_Cloud_list.shape[0])
                evap = np.zeros_like(time_sub)
                disp = np.zeros_like(time_sub)
                strand = np.zeros_like(time_sub)
                result_sub = np.concatenate((time_sub, m_Cloud_list, evap, disp, strand), axis=0)

                time_sur = i[2].time
                time_sur = time_sur.reshape(1, time_sur.shape[0])
                m_Slick_list = np.array(i[2].m_Slick_list)
                m_Slick_list = m_Slick_list.reshape(1, m_Slick_list.shape[0])
                evap_sur = i[2].evap_list
                evap_sur = evap_sur.reshape(1, evap_sur.shape[0])
                disp_sur = i[2].disp_list
                disp_sur = disp_sur.reshape(1, disp_sur.shape[0])
                strand = np.zeros_like(time_sur)
                if i[2].strand:
                    strand[0, -1] = m_Slick_list[0, -1]
                    m_Slick_list[0, -1] = 0

                if time_sur[0, -1] < release_duration_hrs and m_Slick_list[0, -1] == 0:
                    t_tail = np.arange(time_sur[0, -1], release_duration_hrs, self.dt_sur)[1:]
                    t_tail = t_tail.reshape(1, t_tail.shape[0])

                    surface_tail = np.zeros_like(t_tail)

                    evap_tail = np.zeros_like(t_tail)
                    evap_tail[:] = evap_sur[0, -1]

                    disp_tail = np.zeros_like(t_tail)
                    disp_tail[:] = disp_sur[0, -1]

                    time_sur = np.hstack((time_sur, t_tail))
                    m_Slick_list_copy = copy.deepcopy(m_Slick_list)
                    m_Slick_list = np.hstack((m_Slick_list, surface_tail))
                    evap_sur = np.hstack((evap_sur, evap_tail))
                    disp_sur = np.hstack((disp_sur, disp_tail))

                    if i[2].strand:
                        strand_tail = np.zeros_like(t_tail)
                        strand_tail[:] = strand[0, -1]

                        strand = np.zeros_like(m_Slick_list_copy)
                        strand = np.hstack((strand, strand_tail))
                    else:
                        strand = np.zeros_like(time_sur)

                result_sur = np.concatenate((time_sur, m_Slick_list, evap_sur, disp_sur, strand), axis=0)

                result = np.concatenate((result_sub, result_sur), axis=1)

            else:
                raise Exception('FFM.store_csv is wrong')

            parcel_result.append(result)

        end_time = remove_non_numeric(str(self.end_time))
        file_name = 'disp_GNOME_0308_ssdi_mass_balance_{0}.csv'.format(end_time)
        with open(file_name, 'w', newline="") as f:
            csvwriter = csv.writer(f)
            for parcel in parcel_result:
                csvwriter.writerows(parcel)

        return file_name

    def store_oil_trajectory_csv(self, file_name=None):
        from conversion_functions import remove_non_numeric

        if file_name is None:
            end_time = remove_non_numeric(str(self.end_time))
            file_name = 'slick_trajectory_{0}.csv'.format(end_time)

        parcel_result = []
        for i in self.FF_results:
            if len(i) == 3:
                cloud = i[1]
                slick = i[2]

                time_sub = np.array(cloud.time)
                x_Cloud_list = np.array(cloud.x_Cloud_list)
                y_Cloud_list = np.array(cloud.y_Cloud_list)
                z_Cloud_list = np.array(cloud.z_Cloud_list)

                time_sur = np.array(slick.time)
                x_Slick_list = np.array(slick.x_Slick_list)
                y_Slick_list = np.array(slick.y_Slick_list)
                z_Slick_list = np.array(slick.z_Slick_list)

                time_sub = time_sub.reshape(1, time_sub.shape[0])
                x_Cloud_list = x_Cloud_list.reshape(1, x_Cloud_list.shape[0])
                y_Cloud_list = y_Cloud_list.reshape(1, y_Cloud_list.shape[0])
                z_Cloud_list = z_Cloud_list.reshape(1, z_Cloud_list.shape[0])

                time_sur = time_sur.reshape(1, time_sur.shape[0])
                x_Slick_list = x_Slick_list.reshape(1, x_Slick_list.shape[0])
                y_Slick_list = y_Slick_list.reshape(1, y_Slick_list.shape[0])
                z_Slick_list = z_Slick_list.reshape(1, z_Slick_list.shape[0])

                result_sur = np.concatenate((x_Slick_list, y_Slick_list, z_Slick_list,
                                             time_sur), axis=0)
                result_sub = np.concatenate((x_Cloud_list, y_Cloud_list, z_Cloud_list,
                                             time_sub), axis=0)
                result = np.concatenate((result_sub, result_sur), axis=1)

                parcel_result.append(result)

        with open(file_name, 'w', newline="") as f:
            csvwriter = csv.writer(f)
            for parcel in parcel_result:
                csvwriter.writerows(parcel)

        return file_name



def assemble_oil_inputs(near_data, current_data, wind_data, oil_droplets, initial_location, Dxy_sub, Dz_sub, Dxy_sur,
                    start_time, release_duration, end_time, release_interval,
                    para_current, para_wind, dt_sub, dt_sur, shore_data):
    """
    Assemble the inputs required by far-field model into a list.
    The DWOSM use multiprocessing technique to speed up the simulation.

    :param near_data: str
            A path of a csv file;
            Vertical profile of ambient water temperature, current velocity, and salinity
    :param oil_droplets: list
        A list contains multiple PlumeParticle
    :param initial_location: list
        Coordinates and depth of the release point, [latitude, longitude, depth (m)]
    :param Dxy_sub: float
        Horizontal diffusion coefficients for subsurface particles in the x- and y-direction
    :param Dz_sub: float
        Vertical diffusion coefficients for subsurface particles in the z-direction
    :param Dxy_sur: float
        Horizontal diffusion coefficients for oil slicks in the x- and y-direction
    :param start_time: datetime.datetime
            Start time of simulation (Y/M/D/H/M).
    :param release_duration: datetime.datetime
        Duration of an oil/gas blowout.
    :param end_time: datetime.datetime
            End time of simulation (Y/M/D/H/M).
    :param release_interval: float
        The time interval that transfers the output of the near-field model
        as the input to the far-field model (hours).
    :param para_current: float, default=1
        current drift coefficient
    :param para_wind: float, default=0.03
        wind drift coefficient
    :param dt_sub: float
        Time step of far-field model for underwater oil (hours)
    :param dt_sur: float
        Time step of far-field model for surfaced oil (hours)

    :return all_inputs: list
        A list contain the inputs required by SPM

    """
    if isinstance(current_data, str):
        current_data = Dataset(current_data)
        wind_data = Dataset(wind_data)
    elif isinstance(current_data, Dataset):
        pass

    # get the hour number between baseline time and start time
    dt02_current = get_delta_t02(current_data, start_time, timezone='cdt')
    dt02_wind = get_delta_t02(wind_data, start_time, timezone='cdt')

    # get the hour number between baseline time and start time
    dt_end_time = (end_time - start_time).total_seconds() / 3600

    # Instantiate a 3d profile
    profile = Profile3d(near_data, current_data, wind_data)

    # Get all the fixed information required by SPM
    all_profiles = profile.return_all()

    # Obtain the exiting time of each particle in hours
    time_exit = get_exit_time(oil_droplets)

    # Obtain the release time of all particle clouds in ndarray
    # and the number of particle cloud released from the near-field
    particle_release_time, release_num = get_release_time(time_exit, release_duration, release_interval)

    # Create a list of ParticleClouds based on the output of NFM
    clouds = assemble_clouds(release_num, oil_droplets, release_interval, initial_location)

    if shore_data is not None:
        shore_polygon = get_shore_polygon(shore_data)

        # create a container
        all_inputs = []

        # iterate all element in the list of ParticleClouds to get a list containing the inputs required by FFM
        count = 0
        for cloud in clouds:
            inputs = []
            inputs.extend((cloud, all_profiles, initial_location, Dxy_sub, Dz_sub, Dxy_sur,
                           particle_release_time[count], dt_end_time, para_current, para_wind,
                           dt_sub, dt_sur, dt02_current, dt02_wind, shore_polygon))
            all_inputs.append(inputs)
            count += 1
    else:
        # create a container
        all_inputs = []

        # iterate all element in the list of ParticleClouds to get a list containing the inputs required by FFM
        count = 0
        for cloud in clouds:
            inputs = []
            inputs.extend((cloud, all_profiles, initial_location, Dxy_sub, Dz_sub, Dxy_sur,
                           particle_release_time[count], dt_end_time, para_current, para_wind,
                           dt_sub, dt_sur, dt02_current, dt02_wind))
            all_inputs.append(inputs)
            count += 1

    # Each element return is a list containing the inputs required by SPM
    return all_inputs


def assemble_gas_inputs(near_data, current_data, wind_data, gas_bubbles, initial_location, Dxy_sub, Dz_sub,
                    start_time, end_time, release_interval, dt_sub):

    if isinstance(current_data, str):
        current_data = Dataset(current_data)
        wind_data = Dataset(wind_data)
    elif isinstance(current_data, Dataset):
        pass

    # get the hour number between baseline time and start time
    dt02_current = get_delta_t02(current_data, start_time, timezone='cdt')
    dt02_wind = get_delta_t02(wind_data, start_time, timezone='cdt')

    # get the hour number between baseline time and start time
    dt_end_time = (end_time - start_time).total_seconds() / 3600

    # Instantiate a 3d profile
    profile = Profile3d(near_data, current_data, wind_data)

    # Get all the fixed information required by SPM
    all_profiles = profile.return_all()

    # Obtain the exiting time of each particle in hours
    time_exit = get_exit_time(gas_bubbles)

    clouds_mass = initial_cloud_mass(gas_bubbles, release_interval)

    clouds = []
    for idx in range(len(gas_bubbles)):
        lat, lon = convert_loc(initial_location, gas_bubbles[idx])
        cloud = ParticleCloud(gas_bubbles[idx], clouds_mass[idx], lon, lat, gas_bubbles[idx].z)
        clouds.append(cloud)

    all_inputs = []

    # iterate all element in the list of ParticleClouds to get a list containing the inputs required by FFM
    count = 0
    for cloud in clouds:
        inputs = []
        inputs.extend((cloud, all_profiles, initial_location, Dxy_sub, Dz_sub, 0,
                       time_exit[count], dt_end_time, 0, 0, dt_sub, 0, dt02_current, dt02_wind))
        all_inputs.append(inputs)
        count += 1

    return all_inputs
