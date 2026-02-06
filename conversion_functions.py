"""
Unit Conversions and Coordinate Utilities
=========================================

Common unit conversions and coordinate transformations for oil spill modeling.

"""

from __future__ import annotations
from datetime import timedelta, datetime

import numpy as np
from netCDF4 import Dataset, num2date


def convert_loc(location, particle):
    # location = [Lat, lon]
    lat = location[0] + particle.y / 111700
    lon = location[1] + particle.x / (111700 * np.cos(np.deg2rad(location[0])))

    return lat, lon


def hours2seconds(time):
    return time * 3600


def seconds2hours(time):
    return time / 3600


def hours2mins(time):
    return time * 60


def m3tobbls(volume):
    return volume * 6.28981


def bblstom3(volume):
    return volume / 6.28981


def ms2knots(speed):
    return speed * 1.94384


def c2k(temp):
    return temp + 273.15


def k2c(temp):
    return temp - 273.15


def remove_non_numeric(input_string: str or datetime):
    """
    remove the non_numeric value in time description from NetCDF file

    :param input_string:
    :return:
    """
    if isinstance(input_string, datetime):
        input_string = str(input_string)
    numeric_string = ''.join(char for char in input_string if char.isdigit())
    return numeric_string


def get_time_string(numeric_string: str):
    """
    get time values from a string

    :param numeric_string:
    :return:
    """
    year, month, day, hour, minute, second = int(numeric_string[0:4]), int(numeric_string[4:6]), \
        int(numeric_string[6:8]), int(numeric_string[8:10]), \
        int(numeric_string[10:12]), int(numeric_string[12:14])

    return year, month, day, hour, minute, second


def get_t0(data: Dataset):
    """

    :param data:
    :return:
    """
    if isinstance(data, str):
        data = Dataset(data)
    time_string = data.variables['time'].units
    time_string_non_numeric = remove_non_numeric(time_string)
    year, month, day, hour, minute, second = get_time_string(time_string_non_numeric)
    t0 = datetime(year, month, day, hour, minute, second)

    return t0


def get_t1(data: Dataset):
    """

    :param data:
    :return:
    """
    time0 = data.variables['time'][0]
    time_string = data.variables['time'].units
    t1 = num2date(time0, time_string)

    return t1


def get_t2(year: int, month: int, day: int, hour: int = 0, minute: int = 0):
    """
    Input a local datetime to generate the input for the start time of the oil spill modeling

    :param year:
    :param month:
    :param day:
    :param hour:
    :param minute:
    :return: GMT
    """

    t2 = datetime(year, month, day, hour, minute)

    return t2


def CDTtoUTC(cdt: datetime):
    """
    Convert UTC time to CDT zone

    :param utc:
    :return:
    """
    utc = cdt + timedelta(hours=5)

    return utc


def UTCtoCDT(utc: datetime):
    """
    Convert GMT time to CDT zone

    :param cdt:
    :return:
    """
    cdt = utc - timedelta(hours=5)

    return cdt


def get_delta_t01(data: Dataset):
    t0 = get_t0(data)
    t1 = get_t1(data)
    t_delta01 = (t1 - t0).total_seconds() / 3600

    return t_delta01


def get_delta_t02(data: Dataset or str, start_time: datetime, timezone: str):
    t0 = get_t0(data)

    if str.upper(timezone) == 'GMT':
        t2 = start_time
    elif str.upper(timezone) == "CDT":
        t2 = CDTtoUTC(start_time)

    t_delta02 = (t2 - t0).total_seconds() / 3600

    return t_delta02


def get_delta_t03(data: Dataset, end_time: datetime, timezone: str):
    t0 = get_t0(data)

    if str.upper(timezone) == 'GMT':
        t3 = end_time
    elif str.upper(timezone) == "CDT":
        t3 = CDTtoUTC(end_time)

    t_delta03 = (t3 - t0).total_seconds() / 3600

    return t_delta03


def get_delta_t12(data: Dataset, start_time: datetime, timezone: str):
    t1 = get_t1(data)

    if str.upper(timezone) == 'GMT':
        t2 = start_time
    elif str.upper(timezone) == "CDT":
        t2 = CDTtoUTC(start_time)
    else:
        print("wrong input for timezone")

    t_delta12 = (t2 - t1).total_seconds() / 3600

    return t_delta12
