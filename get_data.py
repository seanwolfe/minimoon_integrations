import pyoorb
import numpy as np
import os
from astropy.time import Time
from astroquery.jplhorizons import Horizons
import sys

np.set_printoptions(threshold=sys.maxsize)

def get_data_jpl(minimoon, int_step, int_step_unit, start_time, end_time):
    """
    Get ephemeris data from jpl horizons and store them in text files
    The text files are organized [[x], [y], [z], [vx], [vy], [vz]] and ecliptic longitude for
    :return:
    """

    # Integration step (in days)
    int_step_jpl = int_step
    int_step_jpl_unit = int_step_unit

    # Observation locations
    earth_obs = '500'  # Geocentric earth (works for both openorb and jpl)
    emb_obs = '500@3'  # Earth-moon barycenter (work only for jpl)
    sun_obs = '500@10'

    ###############################################
    # Minimoon data
    ###############################################

    if minimoon == '2006 RH120':

        obj_minim_emb = Horizons(id='2006 RH120', location=emb_obs,
                                 epochs={'start': start_time, 'stop': end_time,
                                         'step': str(int_step_jpl) + int_step_jpl_unit})

        obj = Horizons(id='2006 RH120', location=earth_obs,
                       epochs={'start': start_time, 'stop': end_time,
                               'step': str(int_step_jpl) + int_step_jpl_unit})

    else:

        obj_minim_emb = Horizons(id='2020 CD3', location=emb_obs,
                                 epochs={'start': start_time, 'stop': end_time,
                                         'step': str(int_step_jpl) + int_step_jpl_unit})

        obj = Horizons(id='2020 CD3', location=earth_obs,
                       epochs={'start': start_time, 'stop': end_time,
                               'step': str(int_step_jpl) + int_step_jpl_unit})

    eph_minim = obj.vectors()
    eph_minim_2 = obj.ephemerides()

    eph_minim_emb = obj_minim_emb.vectors()
    eph_minim_emb_2 = obj_minim_emb.ephemerides()

    # x,y,z of minimoon wrt earth
    x_minim = eph_minim['x'].quantity.value
    y_minim = eph_minim['y'].quantity.value
    z_minim = eph_minim['z'].quantity.value
    vx_minim = eph_minim['vx'].quantity.value
    vy_minim = eph_minim['vy'].quantity.value
    vz_minim = eph_minim['vz'].quantity.value
    ecl_long = eph_minim_2['ObsEclLon'].value

    # position and velocity of object wrt emb
    x_minim_emb = eph_minim_emb['x'].quantity.value
    y_minim_emb = eph_minim_emb['y'].quantity.value
    z_minim_emb = eph_minim_emb['z'].quantity.value
    vx_minim_emb = eph_minim_emb['vx'].quantity.value
    vy_minim_emb = eph_minim_emb['vy'].quantity.value
    vz_minim_emb = eph_minim_emb['vz'].quantity.value
    ecl_long_emb = eph_minim_emb_2['ObsEclLon'].quantity.value

    # Data
    minim_wrt_earth_jpl = [x_minim.tolist(), y_minim.tolist(), z_minim.tolist(), vx_minim.tolist(), vy_minim.tolist(),
                           vz_minim.tolist(), ecl_long.tolist()]

    # Write data to text file
    np.savetxt('ephemeris/' + minimoon + '_wrt_earth_jpl_' + int_step_jpl_unit + '.txt', minim_wrt_earth_jpl)

    # Data
    minim_wrt_emb_jpl = [x_minim_emb.tolist(), y_minim_emb.tolist(), z_minim_emb.tolist(), vx_minim_emb.tolist(),
                         vy_minim_emb.tolist(), vz_minim_emb.tolist(), ecl_long_emb.tolist()]

    # Write data to text file
    np.savetxt('ephemeris/' + minimoon + '_wrt_emb_jpl_' + int_step_jpl_unit + '.txt', minim_wrt_emb_jpl)
    ##################################################

    #############################################
    # Moon wrt Earth
    ##############################################
    # Connect with JPL Horizons
    obj_moon = Horizons(id='301', location=earth_obs,
                        epochs={'start': start_time, 'stop': end_time,
                                'step': str(int_step_jpl) + int_step_jpl_unit})

    # Get the vectors table from JPL horizons
    eph_moon = obj_moon.vectors()

    # x,y,z of moon wrt earth
    x_moon = eph_moon['x'].quantity.value
    y_moon = eph_moon['y'].quantity.value
    z_moon = eph_moon['z'].quantity.value
    vx_moon = eph_moon['vx'].quantity.value
    vy_moon = eph_moon['vy'].quantity.value
    vz_moon = eph_moon['vz'].quantity.value

    # Data
    moon_jpl = [x_moon.tolist(), y_moon.tolist(), z_moon.tolist(), vx_moon.tolist(), vy_moon.tolist(), vz_moon.tolist()]

    # Write data to text file
    np.savetxt('ephemeris/' + minimoon + '_moon_wrt_earth_jpl_' + int_step_jpl_unit + '.txt', moon_jpl)
    ###############################################

    #############################################
    # Earth-Moon Barycenter wrt to sun
    ##############################################
    obj_emb = Horizons(id='Earth-Moon Barycenter', location=sun_obs,
                       epochs={'start': start_time, 'stop': end_time,
                               'step': str(int_step_jpl) + int_step_jpl_unit})

    # Get the vectors table from JPL horizons
    eph_emb = obj_emb.vectors()

    # position and velocity of earth-moon barycenter (emb) wrt to Sun
    x_emb = eph_emb['x'].quantity.value
    y_emb = eph_emb['y'].quantity.value
    z_emb = eph_emb['z'].quantity.value
    vx_emb = eph_emb['vx'].quantity.value
    vy_emb = eph_emb['vy'].quantity.value
    vz_emb = eph_emb['vz'].quantity.value

    # Data
    emb_jpl = [x_emb.tolist(), y_emb.tolist(), z_emb.tolist(), vx_emb.tolist(), vy_emb.tolist(), vz_emb.tolist()]

    # Write data to text file
    np.savetxt('ephemeris/' + minimoon + '_emb_wrt_sun_jpl_' + int_step_jpl_unit + '.txt', emb_jpl)
    #################################################

    ################################################
    # Earth
    ###############################################
    obj_earth = Horizons(id='399', location=sun_obs,
                         epochs={'start': start_time, 'stop': end_time,
                                 'step': str(int_step_jpl) + int_step_jpl_unit})

    # Get the vectors table from JPL horizons
    eph_earth = obj_earth.vectors()

    # x,y,z,vx,vy,vz of earth wrt to sun
    x_earth = eph_earth['x'].quantity.value
    y_earth = eph_earth['y'].quantity.value
    z_earth = eph_earth['z'].quantity.value
    vx_earth = eph_earth['vx'].quantity.value
    vy_earth = eph_earth['vy'].quantity.value
    vz_earth = eph_earth['vz'].quantity.value

    # Data
    earth_wrt_sun_jpl = [x_earth.tolist(), y_earth.tolist(), z_earth.tolist(), vx_earth.tolist(), vy_earth.tolist(), vz_earth.tolist()]

    # Write data to text file
    np.savetxt('ephemeris/' + minimoon + '_earth_wrt_sun_jpl_' + int_step_jpl_unit + '.txt', earth_wrt_sun_jpl)
    ###########################################################

    ################################################
    # Sun wrt Earth
    ###############################################
    obj_sun = Horizons(id='Sun', location=earth_obs,
                         epochs={'start': start_time, 'stop': end_time,
                                 'step': str(int_step_jpl) + int_step_jpl_unit})

    # Get the vectors table from JPL horizons
    eph_sun = obj_sun.vectors()

    # x,y,z,vx,vy,vz of earth wrt to sun
    x_sun = eph_sun['x'].quantity.value
    y_sun = eph_sun['y'].quantity.value
    z_sun = eph_sun['z'].quantity.value
    vx_sun = eph_sun['vx'].quantity.value
    vy_sun = eph_sun['vy'].quantity.value
    vz_sun = eph_sun['vz'].quantity.value

    # Data
    sun_wrt_earth_jpl = [x_sun.tolist(), y_sun.tolist(), z_sun.tolist(), vx_sun.tolist(), vy_sun.tolist(),
                         vz_earth.tolist()]

    # Write data to text file
    np.savetxt('ephemeris/' + minimoon + '_sun_wrt_earth_jpl_' + int_step_jpl_unit + '.txt', sun_wrt_earth_jpl)
    ###########################################################

    return

def get_data_openorb(minimoon, int_step, perturbers, start_time, end_time):
    """
    Retrieve and store data in text file for open orb ephemeris
    :return:
    """
    ######################
    ### open orb
    #######################

    # Observation locations
    earth_obs = '500'  # Geocentric earth (works for both openorb and jpl)

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
    # orbits = np.array([0.,1.,2.,3.,4.,5.,6.,5373.,1.,1.,3.])
    # using the first item in PS-SSM, 1/100th density, s1.01 file.

    orbit = np.zeros([n, 12], dtype=np.double, order='F')

    if minimoon == '2006 RH120':
        # RH120 orbit - id = 1
        # For Keplerian orbit initialization - you have to specify the heliocentric ecliptic osculating elements
        # from the JPL horizons file !!According to starting date!!
        orbit[0][0] = 1.0
        orbit[0][1] = 0.9977199096279007  # semimajor axis
        orbit[0][2] = 0.03744067182298642  # eccentricity
        orbit[0][3] = np.deg2rad(0.3093522132878884)  # Inclination (rad)
        orbit[0][4] = np.deg2rad(323.0881705631016)  # Longitude of Ascending node (rad)
        orbit[0][5] = np.deg2rad(206.3169504727512)  # Argument of Periapsis (rad)
        orbit[0][6] = np.deg2rad(294.6211657400694)  # Mean anomaly (rad)
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
        orbit[0][3] = np.deg2rad(0.7388576324362984)  # Inclination (rad)
        orbit[0][4] = np.deg2rad(134.7432090092998)  # Longitude of Ascending node (rad)
        orbit[0][5] = np.deg2rad(342.2597100247675)  # Argument of Periapsis (rad)
        orbit[0][6] = np.deg2rad(45.87779192987005)  # Mean anomaly (rad)
        orbit[0][7] = 3.0  # Type of orbit ID 1:Cartesian 2:Cometary 3:Keplerian
        orbit[0][8] = Time(2458914.5, format='jd').to_value('mjd', 'long')  # Epoch of perihelion (MJD)
        orbit[0][9] = 3.0  # timescale type of the epochs provided; integer value: UTC: 1, UT1: 2, TT: 3, TAI: 4
        orbit[0][10] = 31.74  # absolute magnitude of object (H)
        orbit[0][11] = 0.15  # photometric slope parameter of the target - G from HG model


    # Time and observer information
    obscode = earth_obs  # Where are you observing from: https://minorplanetcenter.net/iau/lists/ObsCodesF.html
    mjds = np.arange(Time(start_time, format='isot', scale='utc').to_value('mjd', 'long'),
                        Time(end_time, format='isot', scale='utc').to_value('mjd', 'long'), int_step)
    epochs = np.array(list(zip(mjds, [1] * len(mjds))), dtype=np.double, order='F')

    # Check output format from: https://github.com/oorb/oorb/tree/master/python
    print("calling oorb_ephemeris_full (n-body)")
    eph, err = pyoorb.pyoorb.oorb_ephemeris_full(in_orbits=orbit[0][:].reshape(1, 12),
                                                 in_obscode=obscode,
                                                 in_date_ephems=epochs,
                                                 in_dynmodel='N',
                                                 in_perturbers=perturbers)
    if err != 0: raise Exception("OpenOrb Exception: error code = %d" % err)

    if int_step == 1:
        int_step_unit = 'd'
    elif int_step == 1/24:
        int_step_unit = 'h'
    else:
        int_step_unit = 'm'

    for i in range(len(eph)):
        np.savetxt('ephemeris/' + minimoon + '_' + str(i) + '_wrt_earth_openorb_' + int_step_unit + '.txt', eph[i][:, :])

    return


