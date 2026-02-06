"""
FFM Utility Functions
=====================

Helper functions for far-field model data processing and
particle cloud management.

"""

from __future__ import annotations
import copy

import numpy as np

from conversion_functions import convert_loc
from SPM_utilities import ParticleCloud


def separate_oil_gas(particles):
    oil_droplets = []
    gas_bubbles = []

    for particle in particles:
        if particle.FluidParticle.fp_type == 1:
            oil_droplets.append(particle)
        else:
            gas_bubbles.append(particle)

    return gas_bubbles, oil_droplets


def assemble_clouds(release_num, particles, release_interval, initial_location):
    """
    The function to assemble the particle clouds for FFM

    :param release_num: list
        Each element refers to the number of each cloud released from the near field.
        Those clouds correspond to each particle size bin.
    :param particles: list
        A list contains multiple PlumeParticle
    :param release_interval: float
        The time interval that transfers the output of the near-field model
        as the input to the far-field model (hours).
    :param initial_location: list
        Coordinates and depth of the release point, [latitude, longitude, depth (m)]

    :return clouds: list
        A list contains all ParticleCloud that will be transferred into the FFM

    """

    # Create a container for ParticleCloud
    clouds = []

    # Get all the clouds' masses in a list
    cloud_mass = initial_cloud_mass(particles, release_interval)

    # Record the index of exiting particles
    count = 0

    # First, loop the element in release_num
    for i in release_num:
        # Convert x-y-z coordinate of exiting particles to geo-coordinate
        lat, lon = convert_loc(initial_location, particles[count])

        # Second, loop the clouds with the same particle size
        for j in range(i):
            # Extract the count-th exiting particle
            particle = particles[count]

            # Assemble a ParticleCloud
            cloud = ParticleCloud(particle, cloud_mass[count], lon, lat, particle.z)

            # Add a cloud to list
            clouds.append(cloud)

        # Loop the index of exiting particle type
        count += 1

    return clouds


def get_release_time(exit_time, release_duration, release_interval):
    """
    Get the release time for all the particle clouds in ndarray

    :param exit_time: list
        A list of time that particles exit from the near-field plume
    :param release_duration: datetime.datetime
        Duration of an oil/gas blowout.
    :param release_interval: float
        The time interval that transfers the output of the near-field model
        as the input to the far-field model (hours).

    :return release_time: list
        Each element represents the time that the near-field releases particle cloud into the far-field.
    :return release_num: list
        Each element refers to the number of clouds released from the near field.
        Those clouds correspond to each particle size bin.

    """
    # Create containers to collect
    #   1) all the start time of clouds release from the near field (release_time);
    #   2) total number of particle clouds (release_num).
    release_time = []
    release_num = []
    release_duration_hrs = release_duration.total_seconds() / 3600

    # For the exit time of cloud particles after the spill
    for te in exit_time:
        # Generate a ndarray that starts from the exit time and ends up with release duration,
        # using the step size with release_interval
        time_array = np.arange(te, release_duration_hrs, release_interval)

        # Add a ndarray contain release time of each cloud to a list
        release_time.append(time_array)

        # Add a ndarray contain the length of the release time,
        # i.e., the number of each particle bin size, to a list
        release_num.append(time_array.shape[0])

    # Concatenate a list of ndarray to a single list
    release_time = np.concatenate(release_time, axis=0)

    print('Total release number from the near-field:', release_time.shape[0])

    return release_time, release_num


def initial_cloud_mass(particles, release_interval):
    """
    Get the initial mass of all the clouds exiting from the plume

    :param particles: list
        A list contains multiple PlumeParticle
    :param release_interval: float
        The time interval that transfers the output of the near-field model
        as the input to the far-field model (hours).

    :return clouds_mass: list
        Each element is the total mass (kg) of a cloud corresponding to the particles with
        a specific size released from NF within a time interval

    """

    # Create a list to contain the mass of all the clouds
    clouds_mass = []

    for particle in particles:
        # Get the mass (kg) of certain type particle exiting for NNF within a release interval (hour)
        # mass (kg) = mass of particle when it exits from the plume (kg)
        # * number flux of this particle (number of particles/s)
        #             * time interval for release cloud from NF to FF (h) * 3600 (s/h)
        mass = particle.m * particle.nb0 * release_interval * 3600

        # Add the cloud mass to a list
        clouds_mass.append(mass)

    return clouds_mass


def get_exit_time(particles):
    """
    Get the time that particles exiting from the near-field plume in list

    :param particles: list
        A list contains multiple PlumeParticle

    :return exit_time: list
        A list of time that particles exit from the near-field plume

    """

    # Create a list as a container of exiting time
    exit_time = []

    # Loop all the particles to unify their unit as hour
    for particle in particles:
        exit_time.append(particle.t / 3600)

    return exit_time


def check_time(array, value_to_check):
    result = np.any(array[:, 0] <= value_to_check)

    return result


def clip_state_variables(array, threshold):
     first_elements = array[:, 0]
     clipped_array = array[first_elements <= threshold]

     return clipped_array


def duplicate_gas_results(gas_bubbles_results: list, start_time, end_time, release_interval):

    delta_t23 = (end_time - start_time).total_seconds() / 3600
    last_time_release = delta_t23 - release_interval
    new_gas_bubbles_results = []
    count_bubble = 0
    for result in gas_bubbles_results:
        count_bubble += 1
        state_variables, cloud = result[0], result[1]
        new_state_variables = []
        while check_time(state_variables, last_time_release):
            new_state_variables.append(copy.copy(state_variables))
            state_variables[:, 0] += release_interval
        for element in new_state_variables:
            clipped_array = clip_state_variables(element, last_time_release)
            # new_cloud = update_cloud(clipped_array, cloud)
            new_gas_bubbles_results.append([clipped_array, cloud])

    return new_gas_bubbles_results


def update_cloud(clipped_array, cloud):
    time, x, y, z, mass = (clipped_array[-1, 0], clipped_array[-1, 1], clipped_array[-1, 2],
                           clipped_array[-1, 3], clipped_array[-1, 4:-1])
    cloud.x, cloud.y, cloud.z = x, y, z
    cloud.time = clipped_array[:, 0]

"""
        self.particle = particle
        self.m0_Cloud = m0_Cloud
        self.m_Cloud = m0_Cloud
        self.m0_particle = self.particle.m0
        self.m_particle = self.particle.m
        self.mf = get_mf(self.m_Cloud)
        self.m_Cloud_list = []
        self.x = x
        self.y = y
        self.z = z
        self.age = 0
        self.time = None

"""