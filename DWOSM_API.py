"""
DWOSM Main API
==============

Deep Water Oil Spill Model - integrated near-field and far-field simulation framework.

Classes
-------
DWOSM : Main model interface integrating NFM and FFM

"""

from __future__ import annotations
from FFM import FFM_API
from NFM import Blowout
import numpy as np


class DWOSM(object):
    """
    Deep Water Oil Spill Model (DWOSM)
    An API integrates near-field and far-field models
    """

    def __init__(self, near_data, current_data, wind_data, shore_data=None, spill_image=None, SSDI=False):
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
            A path of a BNA file;
            shoreline data

        Attributes
        ----------
        simulate: bool, default=False
            A flag used to indicate whether the model run or not
        SSDI: bool, default=False
            A flag used to indicate whether to apply SSDI

        """
        self.near_data = near_data
        self.current_data = current_data
        self.wind_data = wind_data
        self.shore_data = shore_data
        self.SSDI = SSDI
        self.spill_image = spill_image
        self.simulate = False

    def run_all(self, start_time: datetime, end_time: datetime, location: list, oil: str, q_oil: float, d_pipe: float,
                release_interval: float = 15 / 60, release_duration: None or timedelta = None,
                dt_near: float = 30, dt_sub: float or int = 15 / 60, dt_sur: float or int = 15 / 60,
                phi_0=-np.pi / 2, theta_0: float or int = 0, gor: float or int = 0,
                num_gas_elements: int = 0, num_oil_elements: int = 15, Dxy_sub: float or int = 1e4,
                Dz_sub: float or int = 1e1, Dxy_sur: float or int = 1e6,
                para_current: float = 1, para_wind: float = 0.03,
                far_field: bool = True, exit_only: bool = True):
        """
        Run the simulation holistically

        :param start_time: datetime.datetime
            Start time of simulation (Y/M/D/H/M).
        :param end_time: datetime.datetime
            End time of simulation (Y/M/D/H/M).
        :param release_duration: datetime.datetime
            Duration of an oil/gas blowout.
        :param location: list
            Coordinates and depth of the release point, [latitude, longitude, depth (m)]
        :param oil: str
            An oil from the NOAA OilLibrary, this should be a string containing the Adios oil
            ID number (e.g., 'AD01554' for Louisiana Light Sweet).
        :param q_oil: float
            Release rate of the dead oil at the release point (bbl/day).
        :param d_pipe: float
            Equivalent circular diameter of the release (m)
        :param release_interval: float
            The time interval that transfers the output of the near-field model
            as the input to the far-field model (hours).
        :param dt_near: float
            Time step of near-field model (s)
        :param dt_sub: float
            Time step of far-field model for underwater oil (hours)
        :param dt_sur: float
            Time step of far-field model for surfaced oil (hours)
        :param phi_0: float, default=-np.pi / 2 (vertical release)
            Vertical angle of the release relative to the horizontal plane; z is
            positive down so that -pi/2 represents a vertically upward flowing release (rad)
        :param theta_0: float, default=0
            Horizontal angle of the release relative to the x-direction (rad)
        :param gor: float, default=0
            Gas to oil ratio at standard surface conditions (ft^3/bbl)
        :param num_gas_elements: int
            Number of gas bubble sizes to include in the gas bubble size distribution from DSD model
        :param num_oil_elements: int
            Number of oil droplet sizes to include in the oil droplet size distribution from DSD model
        :param Dxy_sub: float
            Horizontal diffusion coefficients for subsurface particles in the x- and y-direction
        :param Dz_sub: float
            Vertical diffusion coefficients for subsurface particles in the z-direction
        :param Dxy_sur: float
            Horizontal diffusion coefficients for oil slicks in the x- and y-direction
        :param para_current: float, default=1
            Current drift coefficient
        :param para_wind: float, default=0.03
            Wind drift coefficient
        :param far_field: bool, default=True
            Switch of FFM
        :param exit_only: bool, default=True
            A switch tracks oil exiting from the plume or all the oil.

        """
        self.start_time = start_time
        self.end_time = end_time
        self.release_duration = end_time - start_time if release_duration is None else release_duration
        self.location = location
        self.oil = oil
        self.q_oil = q_oil
        self.d_pipe = d_pipe
        self.release_interval = release_interval
        self.dt_near = dt_near
        self.dt_sub = dt_sub
        self.dt_sur = dt_sur
        self.phi_0 = phi_0
        self.theta_0 = theta_0
        self.gor = gor
        self.num_gas_elements = num_gas_elements
        self.num_oil_elements = num_oil_elements
        self.Dxy_sub = Dxy_sub
        self.Dz_sub = Dz_sub
        self.Dxy_sur = Dxy_sur
        self.para_current = para_current
        self.para_wind = para_wind
        self.far_field = far_field
        self.exit_only = exit_only

        t0 = datetime.now()

        # check the validity of inputs before running
        self.check_inputs()

        # initialize the near-field model
        self.near_field_model = Blowout(location, d_pipe, oil, q_oil, gor, phi_0, theta_0,
                                        num_gas_elements, num_oil_elements,
                                        self.near_data, dt_near, self.SSDI)
        # run the near-field model
        self.near_field_model.simulate()

        if exit_only:
            # get the particles escaping from the plume region
            particles = self.near_field_model.plume_model.particles_outside
        else:
            # get the particles escaping from the plume region
            particles = self.near_field_model.plume_model.particles

        # Set the flag to indicate the near-field simulation is ready completed
        self.simulate_NFM = True

        # Run FFM when switch is on
        if self.far_field:
            # initialize the far-field model
            self.far_field_model = FFM_API(self.near_data, self.current_data, self.wind_data, self.shore_data,
                                           particles, self.location, self.SSDI, self.spill_image)
            # run the far-field model
            self.far_field_model.simulate(self.start_time, self.end_time, self.release_duration, self.release_interval,
                                          self.Dxy_sub, self.Dz_sub, self.Dxy_sur,
                                          self.para_current, self.para_wind,
                                          self.dt_sub, self.dt_sur)
            # self.far_field_model.add_gas_result()

        t1 = datetime.now()
        print('time: ', (t1 - t0).seconds / 60, 'mins')

    def check_inputs(self):
        if (not isinstance(self.near_data, str) or not isinstance(self.current_data, str)
                or not isinstance(self.wind_data, str)):
            raise Exception('Wrong inputs related to near_data/current_data/wind_data')
        elif not isinstance(self.shore_data, str) and self.shore_data is not None:
            raise Exception('Wrong inputs related to shore_data')
        elif not isinstance(self.start_time, datetime) or not isinstance(self.end_time, datetime):
            raise Exception('Wrong inputs related to start_time and end_time')
        elif (not isinstance(self.release_duration, timedelta) and self.release_duration is not None
              and not isinstance(self.release_duration, float)):
            raise Exception('Wrong inputs related to release_duration')
        elif not isinstance(self.location, list):
            raise Exception('Wrong inputs related to location')
        elif not isinstance(self.oil, str):
            raise Exception('Wrong inputs related to oil')
        elif not isinstance(self.q_oil, float) and not isinstance(self.q_oil, int) or self.q_oil < 0:
            raise Exception('Wrong inputs related to q_oil')
        elif not isinstance(self.d_pipe, float):
            raise Exception('Wrong inputs related to d_pipe')
        elif not isinstance(self.release_interval, float):
            raise Exception('Wrong inputs related to release_interval')
        elif not isinstance(self.dt_near, float) and not isinstance(self.dt_near, int) \
                and not isinstance(self.dt_sub, float) and not isinstance(self.dt_sub, int) \
                and not isinstance(self.dt_sur, float) and not isinstance(self.dt_sur, int) \
                or self.dt_near < 0 or self.dt_sub < 0 or self.dt_sur < 0:
            raise Exception('Wrong inputs related to dt_near/dt_sub/dt_sur')
        elif not isinstance(self.phi_0, float) or not isinstance(self.theta_0, int):
            raise Exception('Wrong inputs related to phi_0/theta_0')
        elif not isinstance(self.gor, float) and not isinstance(self.gor, int) or gor < 0:
            raise Exception('Wrong inputs related to gor')
        elif not isinstance(self.num_gas_elements, int) or not isinstance(self.num_oil_elements, int) \
                or self.num_gas_elements < 0 or self.num_oil_elements < 0:
            raise Exception('Wrong inputs related to num_gas_elements/num_oil_elements')
        elif not isinstance(self.Dxy_sub, int) and not isinstance(self.Dxy_sub, float) \
                and not isinstance(self.Dz_sub, int) and not isinstance(self.Dz_sub, float) \
                and not isinstance(self.Dxy_sur, int) and not isinstance(self.Dxy_sur, float) \
                or self.Dxy_sub < 0 or self.Dz_sub < 0 or self.Dxy_sur < 0:
            raise Exception('Wrong inputs related to Dxy_sub/Dz_sub/Dxy_sur')
        elif not isinstance(self.para_current, int) and not isinstance(self.para_current, float) \
                or self.para_current > 1.5 or self.para_current < 0.5 \
                and not isinstance(self.para_wind, float) or self.para_wind > 0.1 or self.para_current < 0:
            raise Exception('Wrong inputs related to para_current/para_wind')
        elif not isinstance(self.far_field, bool) or not isinstance(self.exit_only, bool):
            raise Exception('Wrong inputs related to far_field/exit_only')
        else:
            print('The inputs are verified')
            print('simulation duration: {0} days ({1} hours)'.
                  format(round(self.release_duration.total_seconds() / 24 / 3600, 1),
                         round(self.release_duration.total_seconds() / 3600, 1)))


if __name__ == "__main__":
    from conversion_functions import *

    z0 = 1500
    d0 = 0.45
    substance = 'AD01978'
    q_oil, gor = 60000, 1600
    phi_0, theta_0 = -np.pi / 2., 0.
    num_gas_elements, num_oil_elements = 10, 10
    start_time_CDT = get_t2(2010, 4, 22, 10, 30)
    lon, lat = -88.366, 28.738
    loc = [lat, lon, z0]
    dt_sub, dt_sur = 15 / 60, 15 / 60  # unit: hour
    release_interval = 15 / 60  # unit: hour
    Dxy_sub, Dxy_sur = 1e3, 2e6  # 1.5e6
    SSDI = True

    near_data = 'data/nearfield2010.csv'
    current_data = 'data/current2010_422_512.nc4'
    wind_data = 'data/wind2010.nc'
    shore_data = 'data/shore_GOM/coast_int.bna'

    nearshore_CDT = get_t2(2010, 4, 28, 16)  # 28-APR-10 1600 CDT

    spill_model = DWOSM(near_data, current_data, wind_data, shore_data=shore_data, SSDI=SSDI)
    spill_model.run_all(start_time_CDT, nearshore_CDT, loc, substance, q_oil, d0,
                        dt_sub=dt_sub, dt_sur=dt_sur,
                        num_oil_elements=num_oil_elements, num_gas_elements=num_gas_elements,
                        Dxy_sub=Dxy_sub, Dxy_sur=Dxy_sur, gor=gor, far_field=True, exit_only=False,
                        para_wind=0.03, para_current=1)

    spill_model.near_field_model.plume_model.plot_2d()
    spill_model.near_field_model.plume_model.plot_3d()
    spill_model.far_field_model.plot_2d()
    spill_model.far_field_model.plot_3d()
