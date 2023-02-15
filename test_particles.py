#!/usr/bin/env python

import astropy.units as u
import pyoorb
import numpy as np
import os
from astropy.time import Time
import matplotlib.pyplot as plt
from astroquery.jplhorizons import Horizons
from minimoon_check import  minimoon_check

np.set_printoptions(formatter={'float': lambda x: "{0:0.16f}".format(x)})


if __name__ == "__main__":

    ############
    # Using JPL Horizons
    ############

    # Moon
    obj = Horizons(id='301', location='500',
                    epochs={'start': '2006-04-01', 'stop': '2007-09-01', 'step': '1d'})

    moon_eph = obj.vectors()
    x_moon = moon_eph['x'].quantity
    y_moon = moon_eph['y'].quantity
    z_moon = moon_eph['z'].quantity


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
    N = 10
    orbit_ids = np.linspace(0., N, num=N, dtype=int).astype(float)
    a_dist = 0.9977199096279007 * np.ones(N)  # Distribution of semimajor axes (AU)
    e_dist = np.linspace(0.03, 0.04, num=N)  # Eccentricity
    in_dist = np.deg2rad(0.3093522132878884) * np.ones(N)  # Inclinations (rad)
    om_dist = np.deg2rad(323.0881705631016) * np.ones(N)  # Longitude of Ascending node (rad)
    w_dist = np.deg2rad(206.3169504727512) * np.ones(N)  # Argument of Periapsis (rad)
    ma_dist = np.deg2rad(294.6211657400694) * np.ones(N)  # Mean anomaly (rad)
    elmmt_tp_dist = 3.0 * np.ones(N)   # Type of orbit ID 1:Cartesian 2:Cometary 3:Keplerian
    epoch_dist = Time(2454101.5, format='jd').to_value('mjd', 'long') * np.ones(N) # Epoch of perihelion (MJD)
    tscale_dist = 3.0 * np.ones(N)  # timescale type of the epochs provided; integer value: UTC: 1, UT1: 2, TT: 3, TAI: 4
    H_dist = 29.5 * np.ones(N)  # absolute magnitude of object (H)
    G_dist = 0.15 * np.ones(N)  # photometric slope parameter of the target - G from HG model

    orbit = np.array([orbit_ids, a_dist, e_dist, in_dist, om_dist, w_dist, ma_dist, elmmt_tp_dist,
                       epoch_dist, tscale_dist, H_dist, G_dist], dtype=np.double, order='F').transpose()



    # RH 120 orbit - id = 1
    # For Keplerian orbit initialization - you have to specify the heliocentric ecliptic osculating elements from the JPL horizons file
    orbitrh = np.zeros([1, 12], dtype=np.double, order='F')
    orbitrh[0][0] = 1.0
    orbitrh[0][1] = 0.9977199096279007   # semimajor axis
    orbitrh[0][2] = 0.03744067182298642  # eccentricity
    orbitrh[0][3] = np.deg2rad(0.3093522132878884)  # Inclination (rad)
    orbitrh[0][4] = np.deg2rad(323.0881705631016)  # Longitude of Ascending node (rad)
    orbitrh[0][5] = np.deg2rad(206.3169504727512)  # Argument of Periapsis (rad)
    orbitrh[0][6] = np.deg2rad(294.6211657400694) # Mean anomaly (rad)
    orbitrh[0][7] = 3.0  # Type of orbit ID 1:Cartesian 2:Cometary 3:Keplerian
    orbitrh[0][8] = Time(2454101.5, format='jd').to_value('mjd', 'long')  # Epoch of perihelion (MJD)
    orbitrh[0][9] = 3.0  # timescale type of the epochs provided; integer value: UTC: 1, UT1: 2, TT: 3, TAI: 4
    orbitrh[0][10] = 29.5  # absolute magnitude of object (H)
    orbitrh[0][11] = 0.15  # photometric slope parameter of the target - G from HG model

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
    mjds = np.arange(Time('2006-04-01T00:00:00', format='isot', scale='utc').to_value('mjd', 'long'),
                        Time('2007-04-01T00:00:00', format='isot', scale='utc').to_value('mjd', 'long'), 1)
    epochs = np.array(list(zip(mjds, [1] * len(mjds))), dtype=np.double, order='F')

    # Check output format from: https://github.com/oorb/oorb/tree/master/python
    print("calling oorb_ephemeris_full (n-body)")
    eph, err = pyoorb.pyoorb.oorb_ephemeris_full(in_orbits=orbit,
                                                 in_obscode=obscode,
                                                 in_date_ephems=epochs,
                                                 in_dynmodel='N',
                                                 in_perturbers=perturbers)
    if err != 0: raise Exception("OpenOrb Exception: error code = %d" % err)

    # Check output format from: https://github.com/oorb/oorb/tree/master/python
    print("calling oorb_ephemeris_full (n-body)")
    ephrh, err = pyoorb.pyoorb.oorb_ephemeris_full(in_orbits=orbitrh,
                                                 in_obscode=obscode,
                                                 in_date_ephems=epochs,
                                                 in_dynmodel='N',
                                                 in_perturbers=perturbers)
    if err != 0: raise Exception("OpenOrb Exception: error code = %d" % err)

    # draw Earth
    R_E = (6378 * u.km).to(u.AU) / u.AU
    up, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:50j]
    x_E = R_E * np.cos(up) * np.sin(v)
    y_E = R_E * np.sin(up) * np.sin(v)
    z_E = R_E * np.cos(v)

    xrh = ephrh[0][:, 24] - ephrh[0][:, 30]
    yrh = ephrh[0][:, 25] - ephrh[0][:, 31]
    zrh = ephrh[0][:, 26] - ephrh[0][:, 32]

    fig3 = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(xrh, yrh, zrh, 'gray', label='True')
    ax.plot3D(x_moon, y_moon, z_moon, 'red', label='Moon')
    ax.plot_surface(x_E, y_E, z_E, color="b", alpha=1)  # alpha controls opacity

    # To get observer centered cartesian
    for i in range(0,N):

        x = eph[i][:, 24] - eph[i][:, 30]
        y = eph[i][:, 25] - eph[i][:, 31]
        z = eph[i][:, 26] - eph[i][:, 32]

        orbit_lbl = 'Orbit {}'.format(i)
        ax.plot3D(x, y, z, 'blue', label=orbit_lbl)

    leg = ax.legend(loc='best')
    ax.set_xlabel('x (AU)')
    ax.set_ylabel('y (AU)')
    ax.set_zlabel('z (AU)')
    plt.show()








