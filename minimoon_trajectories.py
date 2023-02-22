#!/usr/bin/env python

import astropy.units as u
import pyoorb
import numpy
import os
from astropy.time import Time
import matplotlib.pyplot as plt
from astroquery.jplhorizons import Horizons
from minimoon_check import minimoon_check
from minimoon_check import minimoon_check_jpl
import sys
import numpy
from astropy import constants as const
numpy.set_printoptions(threshold=sys.maxsize)

if __name__ == "__main__":

    with_rh120 = 1  # Toggle between RH120 and CD3

    if with_rh120 == 1:
        # Integration range start and end dates - if you change dates here you have to change orbital elements
        start_time = '2006-01-01T00:00:00'
        end_time = '2008-02-01T00:00:00'
    else:
        # Integration range start and end dates
        start_time = '2018-01-01T00:00:00'
        end_time = '2020-09-01T00:00:00'

    # Integration step (in days)
    int_step = 1

    # Observation locations
    earth_obs = '500'  # Geocentric earth (works for both openorb and jpl)
    emb_obs = '500@3'  # Earth-moon barycenter (work only for jpl)
    sun_obs = '500@10'

    # Constants
    mu_e = const.GM_earth.value  # Nominal Earth mass parameter (m3/s2)
    M_moon = 7.34767309E22
    mu_emb = const.G.value * (const.M_earth.value + M_moon)

    ############
    # Using JPL Horizons
    ############

    if with_rh120 == 1:

        obj_minim_emb = Horizons(id='2006 RH120', location=emb_obs,
                       epochs={'start': start_time, 'stop': end_time,
                               'step': str(int_step) + 'd'})

        obj = Horizons(id='2006 RH120', location=earth_obs,
                       epochs={'start': start_time, 'stop': end_time,
                               'step': str(int_step) + 'd'})

    else:

        obj_minim_emb = Horizons(id='2020 CD3', location=emb_obs,
                       epochs={'start': start_time, 'stop': end_time,
                               'step': str(int_step) + 'd'})

        obj = Horizons(id='2020 CD3', location=earth_obs,
                       epochs={'start': start_time, 'stop': end_time,
                               'step': str(int_step) + 'd'})

    # Moon
    obj_moon = Horizons(id='301', location=earth_obs,
                    epochs={'start': start_time, 'stop': end_time,
                            'step': '1d'})

    # From Earth-Moon Barycenter
    obj_emb = Horizons(id='Earth-Moon Barycenter', location=sun_obs,
                    epochs={'start': start_time, 'stop': end_time,
                            'step': str(int_step) + 'd'})

    # Earth
    obj_earth = Horizons(id='399', location=sun_obs,
                    epochs={'start': start_time, 'stop': end_time,
                            'step': str(int_step) + 'd'})

    # Get the vectors table from JPL horizons
    eph_moon = obj_moon.vectors()
    eph_minim = obj.vectors()
    eph_minim_2 = obj.ephemerides()
    eph_emb = obj_emb.vectors()
    eph_earth = obj_earth.vectors()
    eph_minim_emb = obj_minim_emb.vectors()
    eph_minim_emb_2 = obj_minim_emb.ephemerides()

    # x,y,z of minimoon wrt earth
    x_minim = eph_minim['x'].quantity / u.AU
    y_minim = eph_minim['y'].quantity / u.AU
    z_minim = eph_minim['z'].quantity / u.AU
    vx_minim = eph_minim['vx'].quantity * u.d / u.AU
    vy_minim = eph_minim['vy'].quantity * u.d / u.AU
    vz_minim = eph_minim['vz'].quantity * u.d / u.AU
    ecl_long = eph_minim_2['ObsEclLon'].quantity / u.deg

    # x,y,z of moon wrt earth
    x_moon = eph_moon['x'].quantity / u.AU
    y_moon = eph_moon['y'].quantity / u.AU
    z_moon = eph_moon['z'].quantity / u.AU

    # position and velocity of earth-moon barycenter (emb) wrt to Sun
    x_emb = eph_emb['x'].quantity / u.AU
    y_emb = eph_emb['y'].quantity / u.AU
    z_emb = eph_emb['z'].quantity / u.AU
    vx_emb = eph_emb['vx'].quantity * u.d / u.AU
    vy_emb = eph_emb['vy'].quantity * u.d / u.AU
    vz_emb = eph_emb['vz'].quantity * u.d / u.AU

    # velocity of earth wrt to sun
    vx_earth = eph_earth['vx'].quantity * u.d / u.AU
    vy_earth = eph_earth['vy'].quantity * u.d / u.AU
    vz_earth = eph_earth['vz'].quantity * u.d / u.AU

    # position and velocity of object wrt emb
    x_minim_emb = eph_minim_emb['x'].quantity / u.AU
    y_minim_emb = eph_minim_emb['y'].quantity / u.AU
    z_minim_emb = eph_minim_emb['z'].quantity / u.AU
    vx_minim_emb = eph_minim_emb['vx'].quantity * u.d / u.AU
    vy_minim_emb = eph_minim_emb['vy'].quantity * u.d / u.AU
    vz_minim_emb = eph_minim_emb['vz'].quantity * u.d / u.AU
    ecl_long_emb = eph_minim_emb_2['ObsEclLon'].quantity / u.deg

    nm_jpl = minimoon_check_jpl(x_minim, y_minim, z_minim, vx_minim, vy_minim, vz_minim, mu_e,
                                start_time, ecl_long)  # Relative to Earth
    nm_jpl_emb = minimoon_check_jpl(x_minim_emb, y_minim_emb, z_minim_emb, vx_minim_emb, vy_minim_emb, vz_minim_emb,
                                    mu_emb, start_time, ecl_long_emb)

    ######################
    ### open orb
    #######################

    # Number of orbits
    n = 1

    print("starting...")
    print("calling oorb_init():")
    ephfile = ""
    if os.getenv('OORB_DATA'):
        ephfile = os.path.join(os.getenv('OORB_DATA'), 'de430.dat')
    pyoorb.pyoorb.oorb_init(ephfile)
    # orb is id, 6 elements, epoch_mjd, H, G, element type index
    # keplerian appears to be element type index 3
    # orbits = numpy.array([0.,1.,2.,3.,4.,5.,6.,5373.,1.,1.,3.])
    # using the first item in PS-SSM, 1/100th density, s1.01 file.

    orbit = numpy.zeros([n, 12], dtype=numpy.double, order='F')

    if with_rh120 == 1:
        # RH120 orbit - id = 1
        # For Keplerian orbit initialization - you have to specify the heliocentric ecliptic osculating elements
        # from the JPL horizons file !!According to starting date!!
        orbit[0][0] = 1.0
        orbit[0][1] = 0.9977199096279007   # semimajor axis
        orbit[0][2] = 0.03744067182298642  # eccentricity
        orbit[0][3] = numpy.deg2rad(0.3093522132878884)  # Inclination (rad)
        orbit[0][4] = numpy.deg2rad(323.0881705631016)  # Longitude of Ascending node (rad)
        orbit[0][5] = numpy.deg2rad(206.3169504727512)  # Argument of Periapsis (rad)
        orbit[0][6] = numpy.deg2rad(294.6211657400694)  # Mean anomaly (rad)
        orbit[0][7] = 3.0  # Type of orbit ID 1:Cartesian 2:Cometary 3:Keplerian
        orbit[0][8] = Time(2454101.5, format='jd').to_value('mjd', 'long')  # Epoch of perihelion (MJD)
        orbit[0][9] = 3.0  # timescale type of the epochs provided; integer value: UTC: 1, UT1: 2, TT: 3, TAI: 4
        orbit[0][10] = 29.5  # absolute magnitude of object (H)
        orbit[0][11] = 0.15  # photometric slope parameter of the target - G from HG model
    else:
        # CD3 orbit - id = 1
        # For Keplerian orbit initialization - you have to specify the heliocentric ecliptic osculating elements from
        # the JPL horizons file
        orbit[0][0] = 1.0
        orbit[0][1] = 1.022115305313184  # semimajor axis
        orbit[0][2] = 0.03494529757096312  # eccentricity
        orbit[0][3] = numpy.deg2rad(0.7388576324362984)  # Inclination (rad)
        orbit[0][4] = numpy.deg2rad(134.7432090092998)  # Longitude of Ascending node (rad)
        orbit[0][5] = numpy.deg2rad(342.2597100247675)  # Argument of Periapsis (rad)
        orbit[0][6] = numpy.deg2rad(45.87779192987005)  # Mean anomaly (rad)
        orbit[0][7] = 3.0  # Type of orbit ID 1:Cartesian 2:Cometary 3:Keplerian
        orbit[0][8] = Time(2458914.5, format='jd').to_value('mjd', 'long')  # Epoch of perihelion (MJD)
        orbit[0][9] = 3.0  # timescale type of the epochs provided; integer value: UTC: 1, UT1: 2, TT: 3, TAI: 4
        orbit[0][10] = 31.74  # absolute magnitude of object (H)
        orbit[0][11] = 0.15  # photometric slope parameter of the target - G from HG model

    # Perturbers - Array of ints (0 = FALSE (i.e. not included) and 1 = TRUE) if a gravitational body should
    # be included in integrations
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

    # Time and observer information
    obscode = earth_obs  # Where are you observing from: https://minorplanetcenter.net/iau/lists/ObsCodesF.html
    mjds = numpy.arange(Time(start_time, format='isot', scale='utc').to_value('mjd', 'long'),
                        Time(end_time, format='isot', scale='utc').to_value('mjd', 'long'), int_step)
    epochs = numpy.array(list(zip(mjds, [1] * len(mjds))), dtype=numpy.double, order='F')

    # Check output format from: https://github.com/oorb/oorb/tree/master/python
    print("calling oorb_ephemeris_full (n-body)")
    eph, err = pyoorb.pyoorb.oorb_ephemeris_full(in_orbits=orbit[0][:].reshape(1, 12),
                                                 in_obscode=obscode,
                                                 in_date_ephems=epochs,
                                                 in_dynmodel='N',
                                                 in_perturbers=perturbers)
    if err != 0: raise Exception("OpenOrb Exception: error code = %d" % err)

    # To get observer-centered cartesian
    x_obs = eph[0][:, 30]
    y_obs = eph[0][:, 31]
    z_obs = eph[0][:, 32]

    #nm = minimoon_check(eph, x_obs, y_obs, z_obs, vx_earth, vy_earth, vz_earth, mu_e, start_time)  # Relative to earth
    # Relative earth-moon barycenter
    nm_emb = minimoon_check(eph, x_emb, y_emb, z_emb, vx_emb, vy_emb, vz_emb, mu_emb, start_time)

    # To get observer centered cartesian
    x = eph[0][:, 24] - eph[0][:, 30]
    y = eph[0][:, 25] - eph[0][:, 31]
    z = eph[0][:, 26] - eph[0][:, 32]
    
    fig3 = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(x_minim, y_minim, z_minim, 'green', linewidth=5, label='JPL Horizons')
    ax.plot3D(x, y, z, 'gray', label='Open Orb')
    ax.plot3D(x_moon, y_moon, z_moon, 'red', label='Moon')

    # draw Earth
    R_E = (6378 * u.km).to(u.AU)/u.AU
    u, v = numpy.mgrid[0:2 * numpy.pi:50j, 0:numpy.pi:50j]
    x_E = R_E * numpy.cos(u) * numpy.sin(v)
    y_E = R_E * numpy.sin(u) * numpy.sin(v)
    z_E = R_E * numpy.cos(v)
    # alpha controls opacity
    ax.plot_surface(x_E, y_E, z_E, color="b", alpha=1)
    leg = ax.legend(loc='best')
    ax.set_xlabel('x (AU)')
    ax.set_ylabel('y (AU)')
    ax.set_zlabel('z (AU)')

    plt.show()




