#!/usr/bin/env python

import astropy.units as u
import pyoorb
import numpy
import os
from astropy.time import Time
import matplotlib.pyplot as plt
from astroquery.jplhorizons import Horizons
from minimoon_check import minimoon_check
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

if __name__ == "__main__":

    ############
    # Using JPL Horizons
    ############
    obj = Horizons(id='2006 RH120', location='500',
                   epochs={'start': '2006-04-01', 'stop': '2007-09-01',
                           'step': '1d'})

    obj2 = Horizons(id='301', location='500',
                    epochs={'start': '2006-04-01', 'stop': '2007-09-01',
                            'step': '1d'})

    obj3 = Horizons(id='2020 CD3', location='500',
                   epochs={'start': '2018-01-01', 'stop': '2020-05-01',
                           'step': '1d'})

    cd3_eph = obj3.vectors()
    moon_eph = obj2.vectors()
    eph_jpl = obj.vectors()
    eph_full = obj3.ephemerides()

    x_jpl = eph_jpl['x'].quantity
    y_jpl = eph_jpl['y'].quantity
    z_jpl = eph_jpl['z'].quantity
    x_moon = moon_eph['x'].quantity
    y_moon = moon_eph['y'].quantity
    z_moon = moon_eph['z'].quantity
    x_cd3 = cd3_eph['x'].quantity
    y_cd3 = cd3_eph['y'].quantity
    z_cd3 = cd3_eph['z'].quantity

    ######################
    ### open orb
    #######################

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

    # Number of orbits
    n = 2
    orbit = numpy.zeros([n, 12], dtype=numpy.double, order='F')

    # RH120 orbit - id = 1
    # For Keplerian orbit initialization - you have to specify the heliocentric ecliptic osculating elements from the JPL horizons file
    orbit[0][0] = 1.0
    orbit[0][1] = 0.9977199096279007   # semimajor axis
    orbit[0][2] = 0.03744067182298642  # eccentricity
    orbit[0][3] = numpy.deg2rad(0.3093522132878884)  # Inclination (rad)
    orbit[0][4] = numpy.deg2rad(323.0881705631016)  # Longitude of Ascending node (rad)
    orbit[0][5] = numpy.deg2rad(206.3169504727512)  # Argument of Periapsis (rad)
    orbit[0][6] = numpy.deg2rad(294.6211657400694) # Mean anomaly (rad)
    orbit[0][7] = 3.0  # Type of orbit ID 1:Cartesian 2:Cometary 3:Keplerian
    orbit[0][8] = Time(2454101.5, format='jd').to_value('mjd', 'long')  # Epoch of perihelion (MJD)
    orbit[0][9] = 3.0  # timescale type of the epochs provided; integer value: UTC: 1, UT1: 2, TT: 3, TAI: 4
    orbit[0][10] = 29.5  # absolute magnitude of object (H)
    orbit[0][11] = 0.15  # photometric slope parameter of the target - G from HG model

    # CD3 orbit - id = 2
    # For Keplerian orbit initialization - you have to specify the heliocentric ecliptic osculating elements from the JPL horizons file
    orbit[1][0] = 2.0
    orbit[1][1] = 1.022115305313184  # semimajor axis
    orbit[1][2] = 0.03494529757096312  # eccentricity
    orbit[1][3] = numpy.deg2rad(0.7388576324362984)  # Inclination (rad)
    orbit[1][4] = numpy.deg2rad(134.7432090092998)  # Longitude of Ascending node (rad)
    orbit[1][5] = numpy.deg2rad(342.2597100247675)  # Argument of Periapsis (rad)
    orbit[1][6] = numpy.deg2rad(45.87779192987005)  # Mean anomaly (rad)
    orbit[1][7] = 3.0  # Type of orbit ID 1:Cartesian 2:Cometary 3:Keplerian
    orbit[1][8] = Time(2458914.5, format='jd').to_value('mjd', 'long')  # Epoch of perihelion (MJD)
    orbit[1][9] = 3.0  # timescale type of the epochs provided; integer value: UTC: 1, UT1: 2, TT: 3, TAI: 4
    orbit[1][10] = 31.74 # absolute magnitude of object (H)
    orbit[1][11] = 0.15  # photometric slope parameter of the target - G from HG model

    # Perturbers - Array of ints (0 = FALSE (i.e. not included) and 1 = TRUE) if a gravitational body should be included in integrations
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


    obscode = "500"  # Where are you observing from: https://minorplanetcenter.net/iau/lists/ObsCodesF.html
    mjds = numpy.arange(Time('2005-04-01T00:00:00', format='isot', scale='utc').to_value('mjd', 'long'),
                        Time('2009-09-01T00:00:00', format='isot', scale='utc').to_value('mjd', 'long'), 1)
    mjds_cd3 = numpy.arange(Time('2018-01-01T00:00:00', format='isot', scale='utc').to_value('mjd', 'long'),
                            Time('2020-05-01T00:00:00', format='isot', scale='utc').to_value('mjd', 'long'), 1)
    epochs = numpy.array(list(zip(mjds, [1] * len(mjds))), dtype=numpy.double, order='F')
    epochs_cd3 = numpy.array(list(zip(mjds_cd3, [1] * len(mjds_cd3))), dtype=numpy.double, order='F')

    # Check output format from: https://github.com/oorb/oorb/tree/master/python
    print("calling oorb_ephemeris_full (n-body)")
    eph, err = pyoorb.pyoorb.oorb_ephemeris_full(in_orbits=orbit[0][:].reshape(1, 12),
                                                 in_obscode=obscode,
                                                 in_date_ephems=epochs,
                                                 in_dynmodel='N',
                                                 in_perturbers=perturbers)
    if err != 0: raise Exception("OpenOrb Exception: error code = %d" % err)

    n = minimoon_check(eph)

    """
    # Check output format from: https://github.com/oorb/oorb/tree/master/python
    print("calling oorb_ephemeris_full (n-body)")
    cd3_eph, err = pyoorb.pyoorb.oorb_ephemeris_full(in_orbits=orbit[1][:].reshape(1, 12),
                                                 in_obscode=obscode,
                                                 in_date_ephems=epochs_cd3,
                                                 in_dynmodel='N',
                                                 in_perturbers=perturbers)
    if err != 0: raise Exception("OpenOrb Exception: error code = %d" % err)

    # To get observer centered cartesian
    x = eph[0][:, 24] - eph[0][:, 30]
    y = eph[0][:, 25] - eph[0][:, 31]
    z = eph[0][:, 26] - eph[0][:, 32]
    x_cd3_o = cd3_eph[0][:, 24] - cd3_eph[0][:, 30]
    y_cd3_o = cd3_eph[0][:, 25] - cd3_eph[0][:, 31]
    z_cd3_o = cd3_eph[0][:, 26] - cd3_eph[0][:, 32]

    fig3 = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(x_jpl, y_jpl, z_jpl, 'green', linewidth=5, label='JPL Horizons')
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

    fig4 = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(x_cd3, y_cd3, z_cd3, 'green', linewidth=5, label='JPL Horizons')
    ax.plot3D(x_cd3_o, y_cd3_o, z_cd3_o, 'gray', label='Open Orb')
    ax.plot3D(x_moon, y_moon, z_moon, 'red', label='Moon')
    ax.plot_surface(x_E, y_E, z_E, color="b", alpha=1)  # alpha controls opacity
    leg = ax.legend(loc='best')
    ax.set_xlabel('x (AU)')
    ax.set_ylabel('y (AU)')
    ax.set_zlabel('z (AU)')

    plt.show()
    """



