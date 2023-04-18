#!/usr/bin/env python

import astropy.units as u
import pyoorb
import numpy as np
import os
from astropy.time import Time
import matplotlib.pyplot as plt
from astroquery.jplhorizons import Horizons
from minimoon_check import minimoon_check
from minimoon_check import minimoon_check_jpl
from get_data import get_data_jpl
from get_data import get_data_openorb
import sys
import numpy
from astropy import constants as const
from space_fncs import eci_ecliptic_to_sunearth_synodic
numpy.set_printoptions(threshold=sys.maxsize)

if __name__ == "__main__":

    # Switches
    reading_data = 1  # 1: Analyze data
    making_data = 0  # 1: Create ephemeris files

    # Constants
    mu_e = const.GM_earth.value  # Nominal Earth mass parameter (m3/s2)
    M_moon = 7.34767309E22  # Mass of moon in kg
    mu_emb = const.G.value * (const.M_earth.value + M_moon)
    minimoon = '2006 RH120'  # For Generating data
    minimoons = ['2006 RH120', '2020 CD3']  # For reading/using data
    int_step_jpl = 1  # Integration step
    int_step_unit = 'h'
    int_step_oorb = 1/24  # OOrb integration step in days

    # Note: If you want to analyze the number of revolutions made, data should be in hours to avoid large changes in
    # ecliptic longitude

    if minimoon == '2006 RH120':
        # Integration range start and end dates - if you change dates here you have to change orbital elements
        start_time = '2006-01-01T00:00:00'
        end_time = '2008-02-01T00:00:00'
    else:
        # Integration range start and end dates
        start_time = '2018-01-01T00:00:00'
        end_time = '2020-09-01T00:00:00'

    # Perturbers (for OpenOrb) - Array of ints (0 = FALSE (i.e. not included) and 1 = TRUE) if a gravitational body
    # should be included in integrations
    mercury = 1
    venus = 1
    earth = 1
    mars = 1
    jupiter = 1
    saturn = 1
    uranus = 1
    neptune = 1
    pluto = 1
    moon = 1
    perturbers = [mercury, venus, earth, mars, jupiter, saturn, uranus, neptune, pluto, moon]

    # If making data
    if making_data == 1:
        print("Making text files to store state vector information for JPL Horizons...")
        get_data_jpl(minimoon, int_step_jpl, int_step_unit, start_time, end_time)
        print("...done")
        print("Making text files to store state vector information for OpenOrb...")
        get_data_openorb(minimoon, int_step_oorb, perturbers, start_time, end_time)
        print("...done")

    # if reading data from files
    if reading_data == 1:

        print("Reading ephemeris files...")

        for i in range(len(minimoons)):

            print("Reading ephemeris files for " + minimoons[i] + '...')

            if minimoons[i] == '2006 RH120':
                # Integration range start and end dates - if you change dates here you have to change orbital elements
                start_time = '2006-01-01T00:00:00'
            else:
                # Integration range start and end dates
                start_time = '2018-01-01T00:00:00'

            # JPL Files
            earth_wrt_sun_jpl = np.loadtxt('ephemeris/' + minimoons[i] + '_earth_wrt_sun_jpl_' + int_step_unit + '.txt')  # Earth with respect to (wrt) sun, from JPL
            emb_wrt_sun_jpl = np.loadtxt('ephemeris/' + minimoons[i] + '_emb_wrt_sun_jpl_' + int_step_unit + '.txt')  # Earth-moon barycenter (emb)
            moon_wrt_earth_jpl = np.loadtxt('ephemeris/' + minimoons[i] + '_moon_wrt_earth_jpl_' + int_step_unit + '.txt')
            minimoon_wrt_earth_jpl = np.loadtxt('ephemeris/' + minimoons[i] + '_wrt_earth_jpl_' + int_step_unit + '.txt')
            minimoon_wrt_emb_jpl = np.loadtxt('ephemeris/' + minimoons[i] + '_wrt_emb_jpl_' + int_step_unit + '.txt')
            sun_wrt_earth_jpl = np.loadtxt('ephemeris/' + minimoons[i] + '_sun_wrt_earth_jpl_' + int_step_unit + '.txt')

            # Open Orb
            minimoon_wrt_earth_openorb = np.loadtxt('ephemeris/' + minimoons[i] + '_0_wrt_earth_openorb_' + int_step_unit + '.txt')
            print("...done")

            # Transform points from ECI ecliptic to Synodic
            # To do: does velocity need to be transformed too?
            trans_minim_jpl = eci_ecliptic_to_sunearth_synodic(sun_wrt_earth_jpl[0:3, :],
                                                               minimoon_wrt_earth_jpl[0:3, :])
            trans_minim_oorb = eci_ecliptic_to_sunearth_synodic(sun_wrt_earth_jpl[0:3, :],
                                                                minimoon_wrt_earth_openorb[:, 24:27].T -
                                                                earth_wrt_sun_jpl[0:3, 0:len(minimoon_wrt_earth_openorb)])

            ephs_oorb_geo = minimoon_wrt_earth_openorb.copy()
            ephs_oorb_geo[:, 24:27] = minimoon_wrt_earth_openorb[:, 24:27] - earth_wrt_sun_jpl[0:3,
                                                                             0:len(minimoon_wrt_earth_openorb)].T
            # Grab the ecliptic longitudes over the trajectory
            # For JPL Horizons
            eclip_long = np.zeros((1, len(minimoon_wrt_earth_jpl[0])))
            for j in range(0, len(minimoon_wrt_earth_jpl[0])):
                angle = np.rad2deg(np.arctan2(trans_minim_jpl[1, j], trans_minim_jpl[0, j]))
                if angle < 0:
                    angle = 360 + angle
                eclip_long[0, j] = angle
            # For Open Orb
            eclip_long_oorb = np.zeros((1, len(minimoon_wrt_earth_openorb)))
            for j in range(0, len(minimoon_wrt_earth_openorb)):
                angle = np.rad2deg(np.arctan2(trans_minim_oorb[1, j], trans_minim_oorb[0, j]))
                if angle < 0:
                    angle = 360 + angle
                eclip_long_oorb[0, j] = angle

            # Generate some minimoon statistics
            # Minimoon check in synodic
            print('Generate minimoon statistics in synodic frame for ' + minimoons[i] + '...')
            minimoon_wrt_earth_jpl_cp = minimoon_wrt_earth_jpl.copy()
            minimoon_wrt_earth_jpl_cp[0:3, :] = trans_minim_jpl
            ephs_oorb_syn = minimoon_wrt_earth_openorb.copy()
            ephs_oorb_syn[:, 24:27] = trans_minim_oorb.T
            minimoon_check(ephs_oorb_syn, earth_wrt_sun_jpl, mu_e, start_time, int_step_oorb, eclip_long_oorb)
            minimoon_check_jpl(minimoon_wrt_earth_jpl_cp, mu_e, start_time, int_step_unit, eclip_long)

            x_moon = moon_wrt_earth_jpl[0, :]
            y_moon = moon_wrt_earth_jpl[1, :]
            z_moon = moon_wrt_earth_jpl[2, :]
            x_earth = earth_wrt_sun_jpl[0, :]
            y_earth = earth_wrt_sun_jpl[1, :]
            z_earth = earth_wrt_sun_jpl[2, :]
            x_jpl = minimoon_wrt_earth_jpl[0, :]
            y_jpl = minimoon_wrt_earth_jpl[1, :]
            z_jpl = minimoon_wrt_earth_jpl[2, :]
            x_oorb = minimoon_wrt_earth_openorb[:, 24] - x_earth[0:len(minimoon_wrt_earth_openorb)]
            y_oorb = minimoon_wrt_earth_openorb[:, 25] - y_earth[0:len(minimoon_wrt_earth_openorb)]
            z_oorb = minimoon_wrt_earth_openorb[:, 26] - z_earth[0:len(minimoon_wrt_earth_openorb)]

            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.plot3D(x_jpl, y_jpl, z_jpl, 'green', linewidth=5, label='JPL Horizons')
            ax.plot3D(x_oorb, y_oorb, z_oorb, 'gray', label='Open Orb')
            ax.plot3D(x_moon, y_moon, z_moon, 'red', label='Moon')

            # draw Earth
            R_E = (6378 * u.km).to(u.AU)/u.AU
            u_E, v_E = numpy.mgrid[0:2 * numpy.pi:50j, 0:numpy.pi:50j]
            x_E = R_E * numpy.cos(u_E) * numpy.sin(v_E)
            y_E = R_E * numpy.sin(u_E) * numpy.sin(v_E)
            z_E = R_E * numpy.cos(v_E)
            # alpha controls opacity
            ax.plot_surface(x_E, y_E, z_E, color="b", alpha=1)
            leg = ax.legend(loc='best')
            ax.set_xlabel('x (AU)')
            ax.set_ylabel('y (AU)')
            ax.set_zlabel('z (AU)')

            fig2 = plt.figure()
            plt.plot(x_jpl, y_jpl, 'green', linewidth=5, label='JPL Horizons')
            plt.plot(x_oorb, y_oorb, 'gray', label='Open Orb')

            fig3 = plt.figure()
            plt.plot(x_jpl, z_jpl, 'green', linewidth=5, label='JPL Horizons')
            plt.plot(x_oorb, z_oorb, 'gray', label='Open Orb')

            fig4 = plt.figure()
            plt.plot(y_jpl, z_jpl, 'green', linewidth=5, label='JPL Horizons')
            plt.plot(y_oorb, z_oorb, 'gray', label='Open Orb')


            fig5 = plt.figure()
            plt.plot(trans_minim_jpl[0, :], trans_minim_jpl[1, :])
            plt.plot(trans_minim_oorb[0, :], trans_minim_oorb[1, :])

            plt.show()




