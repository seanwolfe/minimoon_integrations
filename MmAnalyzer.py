from astropy.time import Time
import pandas as pd
from space_fncs import get_eclip_long
import astropy.units as u
import numpy
from space_fncs import eci_ecliptic_to_sunearth_synodic
from space_fncs import get_M
from space_fncs import get_theta_from_M
from space_fncs import getH2D
from space_fncs import get_emb_synodic
import pyoorb
import numpy as np
import os
from astropy.time import Time
from astroquery.jplhorizons import Horizons
import sys
import matplotlib.pyplot as plt
from astropy.units import cds
from astropy import constants as const
from MM_Parser import MmParser
from scipy.signal import argrelextrema
from space_fncs import get_geo_v
from poliastro.twobody import Orbit
from poliastro.bodies import Sun, Earth, Moon
from space_fncs import get_r_and_v_cr3bp_from_nbody_sun_emb
from space_fncs import jacobi_dim_and_non_dim
from space_fncs import model
from space_fncs import jacobi
from scipy.integrate import odeint
import matplotlib.ticker as ticker

cds.enable()
numpy.set_printoptions(threshold=sys.maxsize)


class MmAnalyzer:

    # Properties
    capture_start = ""
    capture_end = ""
    capture_duration = ""
    three_eh_duration = ""
    minimoon_flag = ""
    epsilon_duration = ""
    revolutions = ""
    one_eh_flag = ""
    one_eh_duration = ""
    min_dist = ""
    max_dist = ""
    rev_flag = ""
    cap_idx = ""
    rel_idx = ""
    H = ""
    retrograde = ""
    x_eh = ""
    y_eh = ""
    z_eh = ""
    stc = ""
    t_ems = ""
    peri_ems = ""
    peri_3hill = ""
    peri_2hill = ""
    peri_1hill = ""
    stc_start = ""
    stc_start_idx = ""
    stc_end = ""
    stc_end_idx = ""
    ems_start = ""
    ems_start_idx = ""
    ems_end = ""
    ems_end_idx = ""

    def __init__(self):
        """
        Constructor for the minimoon analyzer class. Basically, the point of this class is to take data from fedorets
        data, jpl horizons, or even open oorb and analize them.
        """
        return

    def minimoon_check(self, data, grav_param):
        """
        this function is meant to check if and for what period of time the generated ephemerides from data given by the
        integration results of Fedorets in the 2017 paper. More specifically, checks to see if the four conditions
        to become a minimoon are satisfied:
        1) It remains within 3 Earth Hill Radii
            - Calculated as the distance from Earth, given the geocentric cartesian coordinates
        2) It has a negative planetocentric energy, that is, its specific energy wrt to the Earth-Moon barycenter
           (and also just Earth) is negative epsilon = v^2/2 - mu/r
        3) It completes an entire revolution around Earth - tracked through the geocentric ecliptic longitude in the
           geocentric rotating frame
        4) It has at least one approach to within 1 Earth Hill Radii

        :param
                x, y, z, vx, vy, vz: Takes the x y z and vx vy vz ephemeris of the body wrt to the observer
                grav_param: Is the gravitational constant of the body in question (m3/s2)
                start_time: Start time of the integrations (As a calendar date)
                elciptic longitude wrt to observer (degrees)
        :return: several statistics...
        """

        print("Using data to analyze synthetic minimoon: " + str(data.loc[0, "Object id"]))

        # Important constants
        mu = grav_param # Nominal Earth mass parameter (m3/s2)
        aupd = u.AU / u.d  # AU per day
        mps = u.m / u.s  # Meters per second

        conv_day = data["Julian Date"].iloc[1] - data["Julian Date"].iloc[0]

        # Grab the geocentric cartesian coordinates of the test particle
        steps = data.shape[0]  # Number of steps in the integration
        N = 1  # Number of test particles
        strt_tm = Time(data.loc[0, "Julian Date"], format='jd')

        # Variables to provide information of condition 1 (see function definition)
        satisfied_1 = np.zeros((N, steps))  # ==1 when condition one is satisfied ==0 otherwise
        three_eh = 0.03  # Three Earth-Hill radius (in AU)
        distances = np.zeros((N, steps))  # The distance to earth (in AU) for all particles at all times
        time_satisfied_1 = np.zeros((N, 1))  # The number of steps for which the condition 1 was satisfied

        # Variables to provide information of condition 2 (see function definition)
        satisfied_2 = np.zeros((N, steps))  # ==1 when condition two is satisfied ==0 otherwise
        time_satisfied_2 = np.zeros((N, 1))   # The number of steps for which condition two is satisfied
        epsilons = np.zeros((N, steps))  # Specific energy relative to Earth
        vs_rel = np.zeros((N, steps))  # Relative velocity of asteroid wrt to Earth
        # This is the same as 'distances': rs_rel_e = np.zeros((N, steps))  # Relative distance of asteroid wrt to Earth

        # Variables to provide information of condition 3 (see function definition)
        satisfied_3 = np.zeros((N, 1))   # If the object satisfied condition 3 from the function definition during its
        # temp capture
        revolutions = np.zeros((N, 1))   # The number of revolutions completed during the temp capture
        cum_angle_ecl_jpl = 0.  # The cumulative angle over the temporary capture
        thresh = 200
        used_3 = np.zeros((N, steps))

        # Variables to provide information of condition 4 (see function definition)
        satisfied_4 = np.zeros((N, steps))  # ==1 when condition one is satisfied ==0 otherwise
        satisfied_4_overall = np.zeros((N, 1))
        min_distances = np.zeros((N, 1))  # Minimum geocentric distance reached by minimoon
        one_eh = 0.01  # One Earth-Hill radius (in AU)
        time_satisfied_4 = np.zeros((N, 1))  # The number of steps for which the condition 1 was satisfied

        # Variables describing overall captured
        captured = np.zeros((N, steps))  # When the asteroid is captured (i.e. condition 1 and 2 met)
        time_captured = np.zeros((N, 1))
        first_capture = 0
        capture_date = 0
        capture_idx = 0
        prev_capture = 0
        release_date = 0
        release_idx = 0
        ident = 0
        release_idxs = []
        release_dates = []
        one_hill = 0.01

        # State vector components of the minimoon with respect to earth
        vx = data["Geo vx"]
        vy = data["Geo vy"]
        vz = data["Geo vz"]

        # get the ecliptic longitude of the minimoon (to count rotations)
        eclip_long = data["Eclip Long"]

        distance = np.zeros((1, steps))
        epsilon_j = np.zeros((1, steps))
        v_rel_j = np.zeros((1, steps))

        for j in range(0, steps):

            d = data.loc[j, "Distance"]  # Distance of minimoon to observer
            distance[0, j] = d

            # First and fourth conditions
            if d < three_eh:
                satisfied_1[0, j] = 1
                time_satisfied_1[0, 0] += 1 * conv_day

                if d < one_eh:
                    satisfied_4[0, j] = 1
                    satisfied_4_overall[0, 0] = 1
                    time_satisfied_4[0, 0] += 1 * conv_day

            vx_j = (vx[j] * aupd).to(mps) / mps
            vy_j = (vy[j] * aupd).to(mps) / mps
            vz_j = (vz[j] * aupd).to(mps) / mps
            v = np.sqrt(vx_j ** 2 + vy_j ** 2 + vz_j ** 2)  # Velocity of minimoon relative observer - meters per second
            v_rel_j[0, j] = v
            r = (d * u.AU).to(u.m) / u.m  # Distance of minimoon reative to observer - in meters
            epsilon = v**2/2 - mu / r  # Specific energy relative to observer in question
            epsilon_j[0, j] = epsilon

            # Check if condition 2 from the function definition is satisfied
            if epsilon < 0:
                satisfied_2[0, j] = 1
                time_satisfied_2[0, 0] += 1 * conv_day

            # Identify beginning of capture
            if satisfied_1[0, j] == 1 and satisfied_2[0, j] == 1:
                prev_capture = 1
                # Store date of capture
                if first_capture == 0:
                    capture_date = strt_tm + (j * conv_day) * u.day
                    capture_idx = j
                    first_capture = 1
                captured[0, j] = 1
                time_captured[0, 0] += 1 * conv_day

            # Identify end of capture
            if (satisfied_1[0, j] == 0 or satisfied_2[0, j] == 0):
                if prev_capture == 1:
                    if j < steps - 1:
                        # Store date of end of capture
                        prev_capture = 0
                        release_dates.append(strt_tm + (j * conv_day) * u.day)
                        release_idxs.append(j)
                        ident += 1

            if ident == 0 and (j == steps - 1):
                release_dates.append(strt_tm + (steps * conv_day) * u.day)
                release_idxs.append(steps - 1)

            # Check to see how many revolutions were made during capture phase
            if captured[0, j] == 1 and j > 0:
                if captured[0, j - 1] == 1:
                    i0 = eclip_long[j]
                    im1 = eclip_long[j - 1]
                    if abs(i0 - im1) > thresh:
                        if i0 > im1:
                            cum_angle_ecl_jpl += eclip_long[j] - eclip_long[j - 1] - 360
                        elif im1 > i0:
                            cum_angle_ecl_jpl += eclip_long[j] - eclip_long[j - 1] + 360
                    else:
                        cum_angle_ecl_jpl += eclip_long[j] - eclip_long[j - 1]
                    used_3[0, j] = 1

        pd.set_option('display.max_rows', None)
        pd.set_option('display.float_format', lambda x: '%.5f' %x)
        distances[0, :] = distance[0, :]
        vs_rel[0, :] = v_rel_j[0, :]
        epsilons[0, :] = epsilon_j[0, :]
        release_idx = release_idxs[-1]
        release_date = release_dates[-1]
        min_distances[0, 0] = min(distance[0, capture_idx:release_idx]) \
            if distance[0, capture_idx:release_idx].size > 0 else np.nan
        max_distance = max(distance[0, capture_idx:release_idx]) \
            if distance[0, capture_idx:release_idx].size > 0 else np.nan
        revolutions[0, 0] = cum_angle_ecl_jpl / 360.0
        if abs(revolutions[0, 0]) >= 1:
            satisfied_3[0, 0] = 1

        distance = data['Distance']  # don't grab other crossing for the exit of SOI
        hill_idxs = [index for index, value in enumerate(distance) if value <= one_hill]
        if hill_idxs:
            data_eh_crossing = data.iloc[hill_idxs[0]]
            self.x_eh = data_eh_crossing['Synodic x']
            self.y_eh = data_eh_crossing['Synodic y']
            self.z_eh = data_eh_crossing['Synodic z']
        else:
            self.x_eh = np.nan
            self.y_eh = np.nan
            self.z_eh = np.nan

        if time_captured > 0 and satisfied_3[0, 0] == 1 and satisfied_4_overall[0, 0] == 1:
            print("Object became minimoon: YES")
            print("Start of temporary capture: " + str(capture_date.isot))
            print("End of temporary capture: " + str(release_date.isot))
            print("Duration (days) of Temporary capture: " + str(time_captured[0, 0]))
        else:
            print("Object became minimoon: NO")
        print("Time spent (days) closer than 3 Earth Hill Radii: " + str(time_satisfied_1[0, 0]))
        print("Time spent (days) with specific energy less than zero (wrt to observer): " + str(time_satisfied_2[0, 0]))
        if satisfied_3[0, 0] == 1:
            print("Completed at least a revolution: YES")
        else:
            print("Completed at least a revolution: NO")
        print("Number of revolutions completed: " + str(revolutions[0, 0]))
        if satisfied_4_overall[0, 0] == 1:
            print("Approached to within 1 Earth Hill radius: YES")
            print("Time spent (days) close than 1 Earth Hill radius: " + str(time_satisfied_4[0, 0]))
        else:
            print("Approached to within 1 Earth Hill radius: NO")

        if revolutions[0, 0] < 0:
            print("Orbit is retrograde")
            self.retrograde = 1
        else:
            print("Orbit is prograde")
            self.retrograde = 0
        print("Minimum distance reached to observer (AU): " + str(min_distances[0, 0]))
        print("...Done")
        print("\n")

        # Properties
        self.minimoon_flag = 1 if time_captured > 0 and satisfied_3[0, 0] == 1 and satisfied_4_overall[0, 0] == 1 \
            else 0
        self.capture_start = capture_date
        self.capture_end = release_date
        self.capture_duration = time_captured[0, 0]
        self.three_eh_duration = time_satisfied_1[0, 0]
        self.epsilon_duration = time_satisfied_2[0, 0]
        self.revolutions = revolutions[0, 0]
        self.one_eh_flag = True if satisfied_3[0, 0] == 1 else False
        self.one_eh_duration = time_satisfied_4[0, 0]
        self.min_dist = min_distances[0, 0]
        self.rev_flag = True if satisfied_3[0, 0] == 1 else False
        self.cap_idx = int(capture_idx)
        self.rel_idx = int(release_idx)
        self.max_dist = max_distance

        return

    @staticmethod
    def taxonomy(data, master):
        """
        Determine the taxonomic designation of the TCO population according to Urrutxua 2017
        :param data: the file containing all relevant ephemeris data of the object in question
        :return: a taxonomic desination: 1-A,B,C 2-A,B, U:unknown
        """

        one_eh = 0.01  # AU

        # grab distance, capture end and start indices
        distance = data["Distance"]
        name = str(data["Object id"].iloc[0])
        index = master.index[master['Object id'] == name].tolist()
        cap_idx = int(master.loc[index[0], "Capture Index"])
        rel_idx = int(master.loc[index[0], "Release Index"])

        # during the temporary capture
        cap_dist = distance[cap_idx:rel_idx]

        # was its distance smaller than 1 Hill radius for entire capture?
        if all(ele < one_eh for ele in cap_dist):
            # yes? type 2-A
            designation = "2A"

        # no? was its distance greater than 1 Hill radius for entire capture?
        elif all(ele >= one_eh for ele in cap_dist):
            # yes? type 2-B
            designation = "2B"

        # no? --> it crossed the Hill radius during capture --> type 1
        else:
            # did the temporary capture start outside the Hill sphere?
            if cap_dist.iloc[0] >= one_eh:
                # yes? did the temporary capture end outside the Hill sphere?
                if cap_dist.iloc[-1] >= one_eh:
                    # yes? type 1-A
                    designation = "1A"
                else:
                    # no? type 1-B
                    designation = "1B"

            else:
                # no? did the temporary capture end outside the Hill sphere?
                if cap_dist.iloc[-1] >= one_eh:
                    # yes? type 1-C
                    designation = "1C"

                else:
                    # no? Unknown: U
                    designation = "U"

        return designation

    def get_data_mm_oorb_w_horizons(self, mm_parser, data, int_step, perturbers, start_time, end_time, grav_param, minimoon):
        """
        Retrieve data, organize as pandas dataframe, store to file, starting with fedorets data

        :param minimoon:
        :param data:
        :param int_step:
        :param perturbers:
        :param start_time:
        :param end_time:
        :return:
        """
        ######################
        # open orb
        #######################

        # Integration step (in days)
        if int_step == 1:
            int_step_unit = 'd'
            int_step_jpl = 1
        elif int_step == 1 / 24:
            int_step_unit = 'h'
            int_step_jpl = 1
        elif int_step == 1/24/60:
            int_step_unit = 'm'
            int_step_jpl = 1
        else:
            int_step_unit = 'd'
            int_step_jpl = int_step

        int_step_jpl_unit = int_step_unit

        # Observation locations
        earth_obs = '500'  # Geocentric earth (works for both openorb and jpl)
        sun_obs = '500@10'

        # Number of orbits
        n = 1

        print("Generating data using Oorb for minimoon:" + minimoon + "...")
        ephfile = ""
        if os.getenv('OORB_DATA'):
            ephfile = os.path.join(os.getenv('OORB_DATA'), 'de430.dat')
        pyoorb.pyoorb.oorb_init(ephfile)
        # orb is id, 6 elements, epoch_mjd, H, G, element type index
        # keplerian appears to be element type index 3
        # orbits = np.array([0.,1.,2.,3.,4.,5.,6.,5373.,1.,1.,3.])
        # using the first item in PS-SSM, 1/100th density, s1.01 file.
        orbit = np.zeros([n, 12], dtype=np.double, order='F')

        # Initialize
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
            orbit[0][9] = 1.0  # timescale type of the epochs provided; integer value: UTC: 1, UT1: 2, TT: 3, TAI: 4
            orbit[0][10] = 29.5  # absolute magnitude of object (H)
            orbit[0][11] = 0.15  # photometric slope parameter of the target - G from HG model
        elif minimoon == '2020 CD3':
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
            orbit[0][9] = 1.0  # timescale type of the epochs provided; integer value: UTC: 1, UT1: 2, TT: 3, TAI: 4
            orbit[0][10] = 31.74  # absolute magnitude of object (H)
            orbit[0][11] = 0.15  # photometric slope parameter of the target - G from HG model
        else:
            # A synthetic minimoon orbit
            # For Keplerian orbit initialization - you have to specify the heliocentric ecliptic osculating elements from
            # the JPL horizons file
            orbit[0][0] = 1.0
            orbit[0][1] = data["Helio x"].iloc[0]  # x
            orbit[0][2] = data["Helio y"].iloc[0]  # y
            orbit[0][3] = data["Helio z"].iloc[0]  # z
            orbit[0][4] = data["Helio vx"].iloc[0]  # vx
            orbit[0][5] = data["Helio vy"].iloc[0]  # vy
            orbit[0][6] = data["Helio vz"].iloc[0]  # vz
            orbit[0][7] = 1.  # Type of orbit ID 1:Cartesian 2:Cometary 3:Keplerian
            orbit[0][8] = Time(data["Julian Date"].iloc[0], format='jd').to_value('mjd', 'long')  # Epoch of osculating
            orbit[0][9] = 1.0  # timescale type of the epochs provided; integer value: UTC: 1, UT1: 2, TT: 3, TAI: 4
            orbit[0][10] = mm_parser.mm_data["x7"].iloc[2]  # absolute magnitude of object (H)
            orbit[0][11] = 0.15  # photometric slope parameter of the target - G from HG model

        #############################################
        # Integrations
        ############################################

        # Open orb generates eph
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
        print("...done")

        ########
        # JPL horizons
        #######
        # by default horizons eph have one more value than oorb eph
        print("Generating ephimerides data using Horizons JPL...")



        ################################################
        # Earth wrt to Sun
        ###############################################
        print("Earth with respect to Sun...")
        obj_sun = Horizons(id='399', location=sun_obs,
                           epochs={'start': start_time, 'stop': end_time,
                                   'step': str(int_step_jpl) + int_step_jpl_unit})

        # Get the vectors table from JPL horizons
        print(obj_sun)
        eph_sun = obj_sun.vectors()
        print("...done")

        ################################################
        # Moon wrt to Sun
        ###############################################
        print("Moon with respect to Sun...")
        obj_moon = Horizons(id='301', location=sun_obs,
                           epochs={'start': start_time, 'stop': end_time,
                                   'step': str(int_step_jpl) + int_step_jpl_unit})

        # Get the vectors table from JPL horizons
        eph_moon = obj_moon.vectors()
        print("...done")

        #############################################
        # Generate 39 element data frame containing results of the integrations: elements from 0-38
        #############################################
        """
        Units are au, au/day, degrees
        "Object id", "Julian Date", "Distance", "Helio q", "Helio e", "Helio i", "Helio Omega ", "Helio omega",
        "Helio M", "Helio x", "Helio y", "Helio z", "Helio vx", "Helio vy", "Helio vz", "Geo x", "Geo y", "Geo z",
        "Geo vx", "Geo vy", "Geo vz", "Geo q", "Geo e", "Geo i", "Geo Omega", "Geo omega", "Geo M", "Earth x (Helio)",
        "Earth y (Helio)", "Earth z (Helio)", "Earth vx (Helio)", "Earth vy (Helio)", "Earth vz (Helio)",
        "Moon x (Helio)", "Moon y (Helio)", "Moon z (Helio)", "Moon vx (Helio)", "Moon vy (Helio)",
        "Moon vz (Helio)", "Synodic x", "Synodic y", "Synodic z", "Eclip Long"
        """

        nsteps = 0
        if len(eph[0]) == len(eph_sun) - 1:
            nsteps = len(eph[0])
        elif len(eph[0]) == len(eph_sun) + 1:
            nsteps = len(eph_sun)
        elif len(eph[0]) == len(eph_sun):
            nsteps = len(eph[0])
        else:
            print("Horizons and OOrb integration step error")

        data_temp = {}
        new_data = pd.DataFrame(data_temp)

        # element 0 - object id
        new_data["Object id"] = [minimoon] * nsteps

        # element 1 - julian date
        new_data["Julian Date"] = [Time(eph[0, i, 0], format='mjd').to_value('jd', 'long') for i in range(nsteps)]

        # element 2 - distance
        new_data["Distance"] = [np.sqrt((eph[0, i, 24] - eph_sun[i]['x']) ** 2 + (eph[0, i, 25] - eph_sun[i]['y']) ** 2
                                        + (eph[0, i, 26] - eph_sun[i]['z'])**2) for i in range(nsteps)]

        # for elements 3 to 8, state vector should be converted to cometary and keplarian orbital elements
        orbits = np.zeros([nsteps, 1, 12], dtype=np.double, order='F')
        sun_c = 11  # 11 for Sun, 3 for Earth
        print("Getting helio keplarian osculating elements...")
        for i in range(nsteps):
            # the original orbit is in cartesian:[id x y z vx vy vz type epoch timescale H G]
            orbits[i, :] = [i, eph[0, i, 24], eph[0, i, 25], eph[0, i, 26], eph[0, i, 27], eph[0, i, 28],
                           eph[0, i, 29], 1., eph[0, i, 0], 1., self.H, 0.15]

        # new orbit is in cometary: [id q e i Om om tp type epoch timescale H G]
        new_orbits_com, err = pyoorb.pyoorb.oorb_element_transformation(in_orbits=orbits,
                                                                                  in_element_type=2, in_center=sun_c)

        # new orbit is in keplarian: [id a e i Om om M type epoch timescale H G]
        new_orbits_kep, err = pyoorb.pyoorb.oorb_element_transformation(in_orbits=orbits,
                                                                                  in_element_type=3, in_center=sun_c)
        print("...done")

        # element 3 - Helio q
        new_data["Helio q"] = new_orbits_com[:, 1]

        # element 4 - Helio e
        new_data["Helio e"] = new_orbits_com[:, 2]

        # element 5 - Helio i
        new_data["Helio i"] = np.rad2deg(new_orbits_com[:, 3])

        # element 6 - Helio Omega
        new_data["Helio Omega"] = np.rad2deg(new_orbits_com[:, 4])

        # element 7 - Helio omega
        new_data["Helio omega"] = np.rad2deg(new_orbits_com[:, 5])

        # element 8 - Helio M
        new_data["Helio M"] = np.rad2deg(new_orbits_kep[:, 6])

        # element 9-14 - State vector
        new_data["Helio x"] = eph[0, :nsteps, 24]
        new_data["Helio y"] = eph[0, :nsteps, 25]
        new_data["Helio z"] = eph[0, :nsteps, 26]
        new_data["Helio vx"] = eph[0, :nsteps, 27]
        new_data["Helio vy"] = eph[0, :nsteps, 28]
        new_data["Helio vz"] = eph[0, :nsteps, 29]

        # element 15-20 - Geocentric State vector
        new_data["Geo x"] = eph[0, :nsteps, 24] - eph_sun[:nsteps]['x']  # sun-mm pos vec - sun-earth pos vec
        new_data["Geo y"] = eph[0, :nsteps, 25] - eph_sun[:nsteps]['y']
        new_data["Geo z"] = eph[0, :nsteps, 26] - eph_sun[:nsteps]['z']
        new_data["Geo vx"] = eph[0, :nsteps, 27] - eph_sun[:nsteps]['vx']  # geo is non-rotating frame vrel = vmm - ve
        new_data["Geo vy"] = eph[0, :nsteps, 28] - eph_sun[:nsteps]['vy']
        new_data["Geo vz"] = eph[0, :nsteps, 29] - eph_sun[:nsteps]['vz']

        # for elements 21 to 26, geo state vector should be converted to cometary and keplarian orbital elements
        orbits_geo = np.zeros([nsteps, 1, 12], dtype=np.double, order='F')
        earth_c = 3
        print("Getting geocentric osculating keplarian elements...")
        for i in range(nsteps):
            # the original orbit is in cartesian:[id x y z vx vy vz type epoch timescale H G]
            orbits_geo[i, :, :] = [i, new_data["Geo x"].iloc[i], new_data["Geo y"].iloc[i], new_data["Geo z"].iloc[i],
                                new_data["Geo vx"].iloc[i], new_data["Geo vy"].iloc[i], new_data["Geo vz"].iloc[i],
                                1., eph[0, i, 0], 1, self.H, 0.15]

        # new orbit is in cometary: [id q e i Om om tp type epoch timescale H G]
        new_orbits_com_geo, err = pyoorb.pyoorb.oorb_element_transformation(in_orbits=orbits_geo,
                                                                                  in_element_type=2, in_center=earth_c)
        print("...Done")

        # element 21 - Geo q
        new_data["Geo q"] = new_orbits_com_geo[:, 1]

        # element 22 - Geo e
        new_data["Geo e"] = new_orbits_com_geo[:, 2]

        # element 23 - Geo i
        new_data["Geo i"] = np.rad2deg(new_orbits_com_geo[:, 3])

        # element 24 - Geo Omega
        new_data["Geo Omega"] = np.rad2deg(new_orbits_com_geo[:, 4])

        # element 25 - Geo omega
        new_data["Geo omega"] = np.rad2deg(new_orbits_com_geo[:, 5])

        # element 26 - Geo M
        print("Calculating mean anomaly...")
        temp = np.rad2deg(get_M(new_data, new_orbits_com_geo[:, 6], grav_param))
        new_data["Geo M"] = temp[0, :]
        print("...done")

        # element 27-32 - Heliocentric state vector of Earth
        new_data["Earth x (Helio)"] = eph_sun[:nsteps]['x']
        new_data["Earth y (Helio)"] = eph_sun[:nsteps]['y']
        new_data["Earth z (Helio)"] = eph_sun[:nsteps]['z']
        new_data["Earth vx (Helio)"] = eph_sun[:nsteps]['vx']
        new_data["Earth vy (Helio)"] = eph_sun[:nsteps]['vy']
        new_data["Earth vz (Helio)"] = eph_sun[:nsteps]['vz']

        # element 33-38 - Heliocentric state vector of moon
        new_data["Moon x (Helio)"] = eph_moon[:nsteps]['x']
        new_data["Moon y (Helio)"] = eph_moon[:nsteps]['y']
        new_data["Moon z (Helio)"] = eph_moon[:nsteps]['z']
        new_data["Moon vx (Helio)"] = eph_moon[:nsteps]['vx']
        new_data["Moon vy (Helio)"] = eph_moon[:nsteps]['vy']
        new_data["Moon vz (Helio)"] = eph_moon[:nsteps]['vz']

        print("Getting synodic x,y,z, ecliptic longitude...")
        earth_xyz = np.array([new_data["Earth x (Helio)"], new_data["Earth y (Helio)"], new_data["Earth z (Helio)"]])
        mm_xyz = np.array([new_data["Geo x"], new_data["Geo y"], new_data["Geo z"]])
        trans_xyz = eci_ecliptic_to_sunearth_synodic(-earth_xyz, mm_xyz)  # minus is to have sun relative to earth

        new_data["Synodic x"] = trans_xyz[0, :]
        new_data["Synodic y"] = trans_xyz[1, :]
        new_data["Synodic z"] = trans_xyz[2, :]

        eclip_long = get_eclip_long(trans_xyz)
        new_data["Eclip Long"] = eclip_long[0, :]

        print("...done")

        # Encapsulate comparison graphs into function to compare with fedorets data
        #if (minimoon != '2006 RH120') and (minimoon != '2020 CD3'):
        #    self.compare(eph, data, new_data)

        return new_data

    def get_data_mm_oorb(self, master_i, old_data, int_step, perturbers, start_time, end_time, grav_param):
        """
        Retrieve data, organize as pandas dataframe, store to file, starting with one the newly integreated files

        :param minimoon:
        :param data:
        :param int_step:
        :param perturbers:
        :param start_time:
        :param end_time:
        :return:
        """
        ######################
        # open orb
        #######################

        # Observation locations
        earth_obs = '500'  # Geocentric earth (works for both openorb and jpl)
        sun_obs = '500@10'

        # Number of orbits
        n = 1

        print("Generating data using Oorb for minimoon:" + str(master_i['Object id'].iloc[0]) + "...")
        ephfile = ""
        if os.getenv('OORB_DATA'):
            ephfile = os.path.join(os.getenv('OORB_DATA'), 'de430.dat')
        pyoorb.pyoorb.oorb_init(ephfile)
        # orb is id, 6 elements, epoch_mjd, H, G, element type index
        # keplerian appears to be element type index 3
        # orbits = np.array([0.,1.,2.,3.,4.,5.,6.,5373.,1.,1.,3.])
        # using the first item in PS-SSM, 1/100th density, s1.01 file.
        orbit = np.zeros([n, 12], dtype=np.double, order='F')

        # Initialize
        # A synthetic minimoon orbit
        # For Keplerian orbit initialization - you have to specify the heliocentric ecliptic osculating elements from
        # the JPL horizons file
        orbit[0][0] = 1.0
        orbit[0][1] = master_i["Helio x at Capture"].iloc[0]  # x
        orbit[0][2] = master_i["Helio y at Capture"].iloc[0]  # y
        orbit[0][3] = master_i["Helio z at Capture"].iloc[0]  # z
        orbit[0][4] = master_i["Helio vx at Capture"].iloc[0]  # vx
        orbit[0][5] = master_i["Helio vy at Capture"].iloc[0]  # vy
        orbit[0][6] = master_i["Helio vz at Capture"].iloc[0]  # vz
        orbit[0][7] = 1.  # Type of orbit ID 1:Cartesian 2:Cometary 3:Keplerian
        orbit[0][8] = Time(master_i['Capture Date'].iloc[0], format='jd', scale='utc').to_value('mjd', 'long')  # Epoch of osculating
        orbit[0][9] = 1.0  # timescale type of the epochs provided; integer value: UTC: 1, UT1: 2, TT: 3, TAI: 4
        orbit[0][10] = master_i['H'].iloc[0]  # absolute magnitude of object (H)
        orbit[0][11] = 0.15  # photometric slope parameter of the target - G from HG model

        #############################################
        # Integrations
        ############################################

        # Open orb generates eph
        # Time and observer information
        obscode = earth_obs  # Where are you observing from: https://minorplanetcenter.net/iau/lists/ObsCodesF.html
        mjds = np.arange(start_time.to_value('mjd', 'long'),  end_time.to_value('mjd', 'long'), int_step)
        epochs = np.array(list(zip(mjds, [1] * len(mjds))), dtype=np.double, order='F')

        # Check output format from: https://github.com/oorb/oorb/tree/master/python
        print("calling oorb_ephemeris_full (n-body)")
        eph, err = pyoorb.pyoorb.oorb_ephemeris_full(in_orbits=orbit[0][:].reshape(1, 12),
                                                     in_obscode=obscode,
                                                     in_date_ephems=epochs,
                                                     in_dynmodel='N',
                                                     in_perturbers=perturbers)
        if err != 0: raise Exception("OpenOrb Exception: error code = %d" % err)
        print("...done")


        #############################################
        # Generate 39 element data frame containing results of the integrations: elements from 0-38
        #############################################
        """
        Units are au, au/day, degrees
        "Object id", "Julian Date", "Distance", "Helio q", "Helio e", "Helio i", "Helio Omega ", "Helio omega",
        "Helio M", "Helio x", "Helio y", "Helio z", "Helio vx", "Helio vy", "Helio vz", "Geo x", "Geo y", "Geo z",
        "Geo vx", "Geo vy", "Geo vz", "Geo q", "Geo e", "Geo i", "Geo Omega", "Geo omega", "Geo M", "Earth x (Helio)",
        "Earth y (Helio)", "Earth z (Helio)", "Earth vx (Helio)", "Earth vy (Helio)", "Earth vz (Helio)",
        "Moon x (Helio)", "Moon y (Helio)", "Moon z (Helio)", "Moon vx (Helio)", "Moon vy (Helio)",
        "Moon vz (Helio)", "Synodic x", "Synodic y", "Synodic z", "Eclip Long"
        """


        data_temp = {}
        new_data = pd.DataFrame(data_temp)

        nsteps = len(eph[0])

        # element 0 - object id
        new_data["Object id"] = [master_i['Object id'].iloc[0]] * nsteps

        # element 1 - julian date
        new_data["Julian Date"] = [Time(eph[0, i, 0], format='mjd').to_value('jd', 'long') for i in range(nsteps)]

        # element 2 - distance
        new_data["Distance"] = eph[0, :, 8]

        # for elements 3 to 8, state vector should be converted to cometary and keplarian orbital elements
        orbits = np.zeros([nsteps, 1, 12], dtype=np.double, order='F')
        sun_c = 11  # 11 for Sun, 3 for Earth
        print("Getting helio keplarian osculating elements...")
        for i in range(nsteps):
            # the original orbit is in cartesian:[id x y z vx vy vz type epoch timescale H G]
            orbits[i, :] = [i, eph[0, i, 24], eph[0, i, 25], eph[0, i, 26], eph[0, i, 27], eph[0, i, 28],
                           eph[0, i, 29], 1., eph[0, i, 0], 1., master_i['H'].iloc[0], 0.15]

        # new orbit is in cometary: [id q e i Om om tp type epoch timescale H G]
        new_orbits_com, err = pyoorb.pyoorb.oorb_element_transformation(in_orbits=orbits,
                                                                                  in_element_type=2, in_center=sun_c)

        # new orbit is in keplarian: [id a e i Om om M type epoch timescale H G]
        new_orbits_kep, err = pyoorb.pyoorb.oorb_element_transformation(in_orbits=orbits,
                                                                                  in_element_type=3, in_center=sun_c)
        print("...done")

        # element 3 - Helio q
        new_data["Helio q"] = new_orbits_com[:, 1]

        # element 4 - Helio e
        new_data["Helio e"] = new_orbits_com[:, 2]

        # element 5 - Helio i
        new_data["Helio i"] = np.rad2deg(new_orbits_com[:, 3])

        # element 6 - Helio Omega
        new_data["Helio Omega"] = np.rad2deg(new_orbits_com[:, 4])

        # element 7 - Helio omega
        new_data["Helio omega"] = np.rad2deg(new_orbits_com[:, 5])

        # element 8 - Helio M
        new_data["Helio M"] = np.rad2deg(new_orbits_kep[:, 6])

        # element 9-14 - State vector
        new_data["Helio x"] = eph[0, :, 24]
        new_data["Helio y"] = eph[0, :, 25]
        new_data["Helio z"] = eph[0, :, 26]
        new_data["Helio vx"] = eph[0, :, 27]
        new_data["Helio vy"] = eph[0, :, 28]
        new_data["Helio vz"] = eph[0, :, 29]

        # element 15-20 - Geocentric State vector
        new_data["Geo x"] = eph[0, :, 24] - eph[0, :, 30]  # sun-mm pos vec - sun-earth pos vec
        new_data["Geo y"] = eph[0, :, 25] - eph[0, :, 31]
        new_data["Geo z"] = eph[0, :, 26] - eph[0, :, 32]
        new_data["Geo vx"] = eph[0, :, 27] - eph[0, :, 33]
        new_data["Geo vy"] = eph[0, :, 28] - eph[0, :, 34]
        new_data["Geo vz"] = eph[0, :, 29] - eph[0, :, 35]

        # for elements 21 to 26, geo state vector should be converted to cometary and keplarian orbital elements
        orbits_geo = np.zeros([nsteps, 1, 12], dtype=np.double, order='F')
        earth_c = 3
        print("Getting geocentric osculating keplarian elements...")
        for i in range(nsteps):
            # the original orbit is in cartesian:[id x y z vx vy vz type epoch timescale H G]
            orbits_geo[i, :, :] = [i, new_data["Geo x"].iloc[i], new_data["Geo y"].iloc[i], new_data["Geo z"].iloc[i],
                                new_data["Geo vx"].iloc[i], new_data["Geo vy"].iloc[i], new_data["Geo vz"].iloc[i],
                                1., eph[0, i, 0], 1, master_i['H'].iloc[0], 0.15]

        # new orbit is in cometary: [id q e i Om om tp type epoch timescale H G]
        new_orbits_com_geo, err = pyoorb.pyoorb.oorb_element_transformation(in_orbits=orbits_geo,
                                                                                  in_element_type=2, in_center=earth_c)
        print("...Done")

        # element 21 - Geo q
        new_data["Geo q"] = new_orbits_com_geo[:, 1]

        # element 22 - Geo e
        new_data["Geo e"] = new_orbits_com_geo[:, 2]

        # element 23 - Geo i
        new_data["Geo i"] = np.rad2deg(new_orbits_com_geo[:, 3])

        # element 24 - Geo Omega
        new_data["Geo Omega"] = np.rad2deg(new_orbits_com_geo[:, 4])

        # element 25 - Geo omega
        new_data["Geo omega"] = np.rad2deg(new_orbits_com_geo[:, 5])

        # element 26 - Geo M
        print("Calculating mean anomaly...")
        temp = np.rad2deg(get_M(new_data, new_orbits_com_geo[:, 6], grav_param))
        new_data["Geo M"] = temp[0, :]
        print("...done")

        # element 27-32 - Heliocentric state vector of Earth
        new_data["Earth x (Helio)"] = eph[0, :, 30]
        new_data["Earth y (Helio)"] = eph[0, :, 31]
        new_data["Earth z (Helio)"] = eph[0, :, 32]
        new_data["Earth vx (Helio)"] = eph[0, :, 33]
        new_data["Earth vy (Helio)"] = eph[0, :, 34]
        new_data["Earth vz (Helio)"] = eph[0, :, 35]

        # element 33-38 - Heliocentric state vector of moon
        new_data["Moon x (Helio)"] = old_data["Moon x (Helio)"]
        new_data["Moon y (Helio)"] = old_data["Moon y (Helio)"]
        new_data["Moon z (Helio)"] = old_data["Moon z (Helio)"]
        new_data["Moon vx (Helio)"] = old_data["Moon vx (Helio)"]
        new_data["Moon vy (Helio)"] = old_data["Moon vy (Helio)"]
        new_data["Moon vz (Helio)"] = old_data["Moon vz (Helio)"]

        print("Getting synodic x,y,z, ecliptic longitude...")
        earth_xyz = np.array([new_data["Earth x (Helio)"], new_data["Earth y (Helio)"], new_data["Earth z (Helio)"]])
        mm_xyz = np.array([new_data["Geo x"], new_data["Geo y"], new_data["Geo z"]])
        trans_xyz = eci_ecliptic_to_sunearth_synodic(-earth_xyz, mm_xyz)  # minus is to have sun relative to earth

        new_data["Synodic x"] = trans_xyz[0, :]
        new_data["Synodic y"] = trans_xyz[1, :]
        new_data["Synodic z"] = trans_xyz[2, :]

        eclip_long = get_eclip_long(trans_xyz)
        new_data["Eclip Long"] = eclip_long[0, :]

        print("...done")

        # Encapsulate comparison graphs into function to compare with fedorets data
        #if (minimoon != '2006 RH120') and (minimoon != '2020 CD3'):
        # self.compare(eph, old_data, new_data)

        return new_data

    @staticmethod
    def compare(eph, data, new_data):
        """
        This function provides a graphical comparison of all the parameters in a dataframe that is the result of orbit
        propogation for a synthetic minimoon with fedorets given data
        :param eph: orbital integration results
        :param data: original fedorets data
        :param new_data: new dataframe of orbit integration results
        :return:
        """


        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(data["Helio x"], data["Helio y"], data["Helio z"], 'green', linewidth=1, label='Fedorets mm')
        ax.plot3D(new_data["Helio x"], new_data["Helio y"], new_data["Helio z"], 'gray', label='Open Orb mm')
        ax.plot3D(data["Moon x (Helio)"], data["Moon y (Helio)"], data["Moon z (Helio)"], 'blue', linewidth=1,
                  label='Fedorets moon')
        ax.plot3D(new_data["Earth x (Helio)"], new_data["Earth y (Helio)"], new_data["Earth z (Helio)"], 'black',
                  label='Open Orb earth')
        leg = ax.legend(loc='best')
        ax.set_xlabel('Hx')
        ax.set_ylabel('Hy')
        ax.set_zlabel('Hz')

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(data["Geo x"], data["Geo y"], data["Geo z"], 'green', linewidth=5, label='Fedorets')
        ax.plot3D(new_data["Geo x"], new_data["Geo y"], new_data["Geo z"], 'gray', label='Open Orb')
        leg = ax.legend(loc='best')
        ax.set_xlabel('Gx')
        ax.set_ylabel('Gy')
        ax.set_zlabel('Gz')

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(data["Earth x (Helio)"], data["Earth y (Helio)"], data["Earth z (Helio)"], 'green', linewidth=10,
                  label='Fedorets')
        ax.plot3D(new_data["Earth x (Helio)"], new_data["Earth y (Helio)"], new_data["Earth z (Helio)"], 'gray', linewidth=7,
                  label='Open Orb')
        ax.plot3D(eph[0, :, 30], eph[0, :, 31], eph[0, :, 32], 'blue', label='Open Orb 2')
        leg = ax.legend(loc='best')
        ax.set_xlabel('Ex')
        ax.set_ylabel('Ey')
        ax.set_zlabel('Ez')

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(data["Moon x (Helio)"], data["Moon y (Helio)"], data["Moon z (Helio)"], 'green', linewidth=5,
                  label='Fedorets')
        ax.plot3D(new_data["Moon x (Helio)"], new_data["Moon y (Helio)"], new_data["Moon z (Helio)"], 'gray',
                  label='Open Orb')
        leg = ax.legend(loc='best')
        ax.set_xlabel('HMx')
        ax.set_ylabel('HMy')
        ax.set_zlabel('HMz')

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(data["Julian Date"], data["Distance"], data["Helio q"], 'green', linewidth=5,
                  label='Fedorets')
        ax.plot3D(new_data["Julian Date"], new_data["Distance"], new_data["Helio q"], 'gray',
                  label='Open Orb')
        leg = ax.legend(loc='best')
        ax.set_xlabel('jd')
        ax.set_ylabel('d')
        ax.set_zlabel('q')

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(data["Helio e"], data["Helio i"], data["Helio Omega"], 'green', linewidth=5,
                  label='Fedorets')
        ax.plot3D(new_data["Helio e"], new_data["Helio i"], new_data["Helio Omega"], 'gray',
                  label='Open Orb')
        leg = ax.legend(loc='best')
        ax.set_xlabel('e')
        ax.set_ylabel('i')
        ax.set_zlabel('Om')

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(data["Helio omega"], data["Helio M"], data["Helio vx"], 'green', linewidth=5,
                  label='Fedorets')
        ax.plot3D(new_data["Helio omega"], new_data["Helio M"], new_data["Helio vx"], 'gray',
                  label='Open Orb')
        leg = ax.legend(loc='best')
        ax.set_xlabel('om')
        ax.set_ylabel('M')
        ax.set_zlabel('vx')

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(data["Helio vy"], data["Helio vz"], data["Geo vx"], 'green', linewidth=5,
                  label='Fedorets')
        ax.plot3D(new_data["Helio vy"], new_data["Helio vz"], new_data["Geo vx"], 'gray',
                  label='Open Orb')
        leg = ax.legend(loc='best')
        ax.set_xlabel('vy')
        ax.set_ylabel('vz')
        ax.set_zlabel('Gvx')

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(data["Geo vx"], data["Geo vy"], data["Geo vz"], 'green', linewidth=5,
                  label='Fedorets')
        ax.plot3D(new_data["Geo vx"], new_data["Geo vy"], new_data["Geo vz"], 'gray',
                  label='Open Orb')
        leg = ax.legend(loc='best')
        ax.set_xlabel('Gvx')
        ax.set_ylabel('Gvy')
        ax.set_zlabel('Gvz')

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(data["Geo vy"], data["Geo vz"], data["Geo q"], 'green', linewidth=5,
                  label='Fedorets')
        ax.plot3D(new_data["Geo vy"], new_data["Geo vz"], new_data["Geo q"], 'gray',
                  label='Open Orb')
        leg = ax.legend(loc='best')
        ax.set_xlabel('Gvy')
        ax.set_ylabel('Gvz')
        ax.set_zlabel('Gq')

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(data["Geo e"], data["Geo i"], data["Geo Omega"], 'green', linewidth=5,
                  label='Fedorets')
        ax.plot3D(new_data["Geo e"], new_data["Geo i"], new_data["Geo Omega"], 'gray',
                  label='Open Orb')
        leg = ax.legend(loc='best')
        ax.set_xlabel('Ge')
        ax.set_ylabel('Gi')
        ax.set_zlabel('GOm')

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(data["Geo omega"], data["Geo M"], data["Earth vx (Helio)"], 'green', linewidth=5,
                  label='Fedorets')
        ax.plot3D(new_data["Geo omega"], new_data["Geo M"], new_data["Earth vx (Helio)"], 'gray',
                  label='Open Orb')
        leg = ax.legend(loc='best')
        ax.set_xlabel('Gom')
        ax.set_ylabel('GM')
        ax.set_zlabel('Evx')

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(data["Earth vx (Helio)"], data["Earth vy (Helio)"], data["Earth vz (Helio)"], 'green', linewidth=5,
                  label='Fedorets')
        ax.plot3D(new_data["Earth vx (Helio)"], new_data["Earth vy (Helio)"], new_data["Earth vz (Helio)"], 'gray',
                  label='Open Orb')
        leg = ax.legend(loc='best')
        ax.set_xlabel('Evx')
        ax.set_ylabel('Evy')
        ax.set_zlabel('Evz')

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(data["Moon vx (Helio)"], data["Moon vy (Helio)"], data["Moon vz (Helio)"], 'green', linewidth=5,
                  label='Fedorets')
        ax.plot3D(new_data["Moon vx (Helio)"], new_data["Moon vy (Helio)"], new_data["Moon vz (Helio)"], 'gray',
                  label='Open Orb')
        leg = ax.legend(loc='best')
        ax.set_xlabel('Mvx')
        ax.set_ylabel('Mvy')
        ax.set_zlabel('Mvz')

        plt.show()

        return

    def short_term_capture(self, object_id):

        # constants
        three_eh = 0.03
        r_ems = 0.0038752837677  # sphere of influence of earth-moon system
        two_hill = 0.02
        one_hill = 0.01

        stc = np.NAN

        # go through all the files of test particles
        # population_dir = os.path.join(os.getcwd(), 'minimoon_files_oorb')
        # population_dir = os.path.join('/media', 'aeromec', 'data', 'Test_Set')
        population_dir = os.path.join('/media', 'aeromec', 'data', 'minimoon_files_oorb_nomoon')
        # population_dir = os.path.join(os.getcwd(), 'Test_Set')

        mm_parser = MmParser("", population_dir, "")

        # Initial: will be result if no file found with that name
        results = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                   np.full(6, np.nan), np.full(6, np.nan), np.full(6, np.nan)]

        for root, dirs, files in os.walk(population_dir):
            # find files that are minimoons
            name = str(object_id) + ".csv"

            if name in files:
                file_path = os.path.join(root, name)

                # read the file
                data = mm_parser.mm_file_parse_new(file_path)

                print("Analyzing the Short-Term Capture Statistics of minimoon: " + str(object_id))

                # get data with respect to the earth-moon barycentre in a co-rotating frame
                emb_xyz_synodic = get_emb_synodic(data)

                stc_moon_dist_nomoon = [np.linalg.norm(np.array([rows2['Helio x'] - rows2['Moon x (Helio)'],
                                                                 rows2['Helio y'] - rows2['Moon y (Helio)'],
                                                                 rows2['Helio z'] - rows2['Moon z (Helio)']])) for
                                        i, rows2 in data.iterrows()]

                distance_emb_synodic = np.sqrt(emb_xyz_synodic[:, 0] ** 2 + emb_xyz_synodic[:, 1] ** 2 + emb_xyz_synodic[:, 2] ** 2)

                # identify when inside the 3 earth hill sphere
                three_hill_under = np.NAN * np.zeros((len(distance_emb_synodic),))
                three_hill_idxs = [index for index, value in enumerate(distance_emb_synodic) if value <= three_eh]
                for index in three_hill_idxs:
                    three_hill_under[index] = 1

                captured_x = emb_xyz_synodic[:, 0] * three_hill_under
                captured_y = emb_xyz_synodic[:, 1] * three_hill_under
                captured_z = emb_xyz_synodic[:, 2] * three_hill_under
                captured_distance = distance_emb_synodic * three_hill_under

                # identify periapses that exist in the 3 earth hill sphere
                local_minima_indices = argrelextrema(captured_distance, np.less)[0]
                local_dist = captured_distance[local_minima_indices]
                time = data["Julian Date"] #- data["Julian Date"].iloc[0]
                local_time = time.iloc[local_minima_indices]

                # identify when inside the sphere of influence of the EMS
                in_ems = np.NAN * np.zeros((len(distance_emb_synodic),))
                in_ems_idxs = [index for index, value in enumerate(distance_emb_synodic) if value <= r_ems]
                for index in in_ems_idxs:
                    in_ems[index] = 1

                ems_x = emb_xyz_synodic[:, 0] * in_ems
                ems_y = emb_xyz_synodic[:, 1] * in_ems
                ems_z = emb_xyz_synodic[:, 2] * in_ems
                captured_distance_ems = distance_emb_synodic * in_ems

                # identify periapses that exist in the EMS SOI
                local_minima_indices_ems = argrelextrema(captured_distance_ems, np.less)[0]
                local_dist_ems = captured_distance_ems[local_minima_indices_ems]
                local_time_ems = time.iloc[local_minima_indices_ems]

                # identify when inside the sphere of influence 2 hill
                in_2hill = np.NAN * np.zeros((len(distance_emb_synodic),))
                in_2hill_idxs = [index for index, value in enumerate(distance_emb_synodic) if value <= two_hill]
                for index in in_2hill_idxs:
                    in_2hill[index] = 1

                twohill_x = emb_xyz_synodic[:, 0] * in_2hill
                twohill_y = emb_xyz_synodic[:, 1] * in_2hill
                twohill_z = emb_xyz_synodic[:, 2] * in_2hill
                captured_distance_2hill = distance_emb_synodic * in_2hill

                # identify periapses that exist in the EMS SOI
                local_minima_indices_2hill = argrelextrema(captured_distance_2hill, np.less)[0]
                local_dist_2hill = captured_distance_2hill[local_minima_indices_2hill]
                local_time_2hill = time.iloc[local_minima_indices_2hill]

                # identify when inside the sphere of influence 1 hill
                in_1hill = np.NAN * np.zeros((len(distance_emb_synodic),))
                in_1hill_idxs = [index for index, value in enumerate(distance_emb_synodic) if value <= one_hill]
                for index in in_1hill_idxs:
                    in_1hill[index] = 1

                onehill_x = emb_xyz_synodic[:, 0] * in_1hill
                onehill_y = emb_xyz_synodic[:, 1] * in_1hill
                onehill_z = emb_xyz_synodic[:, 2] * in_1hill
                captured_distance_1hill = distance_emb_synodic * in_1hill

                # identify periapses that exist in the EMS SOI
                local_minima_indices_1hill = argrelextrema(captured_distance_1hill, np.less)[0]
                local_dist_1hill = captured_distance_1hill[local_minima_indices_1hill]
                local_time_1hill = time.iloc[local_minima_indices_1hill]

                ems_line = r_ems * np.ones(len(time),)
                three_eh_line = three_eh * np.ones(len(time),)
                two_eh_line = two_hill * np.ones(len(time), )
                one_eh_line = one_hill * np.ones(len(time), )

                stc = False
                # decide if short-term capture or not
                if len(three_hill_idxs) >= 2:
                    if len(local_minima_indices_ems) >= 2:
                        stc = True

                print(str(object_id) + ": " + str(stc))

                # Data of interest (for the master):
                # Whether a STC took place or not
                # Time spent in SOI EMS
                time_step = data["Julian Date"].iloc[1] - data["Julian Date"].iloc[0]
                time_SOI_EMS = time_step * len(in_ems_idxs)

                # Number of periapsides inside SOI EMS
                peri_in_SOI_EMS = len(local_minima_indices_ems)
                # Number of periapsides inside three Earth Hill
                peri_in_3_hill = len(local_minima_indices)
                # Number of periapsides inside two earth Hill
                peri_in_2_hill = len(local_minima_indices_2hill)
                # Number of periapsides inside one Earth Hill
                peri_in_1_hill = len(local_minima_indices_1hill)

                # STC start - not really...if it is an STC...yes the start...if not...the time of entry/exit of three hill
                stc_start = data['Julian Date'].iloc[three_hill_idxs[0]]
                stc_start_idx = three_hill_idxs[0]
                # STC end
                stc_end_idx = three_hill_idxs[-1]
                stc_end = data['Julian Date'].iloc[stc_end_idx]


                # State of TCO at entrance to SOI EMS
                if in_ems_idxs:

                    Earth_state = data[['Earth x (Helio)', 'Earth y (Helio)', 'Earth z (Helio)', 'Earth vx (Helio)',
                                  'Earth vy (Helio)', 'Earth vz (Helio)']].iloc[in_ems_idxs[0]]
                    # Helio TCO at entrance to SOI EMS
                    TCO_state = data[['Helio x', 'Helio y', 'Helio z', 'Helio vx', 'Helio vy', 'Helio vz']].iloc[in_ems_idxs[0]]
                    # Helio Moon at entrance to SOI EMS
                    moon_state = data[['Moon x (Helio)', 'Moon y (Helio)', 'Moon z (Helio)', 'Moon vx (Helio)',
                                    'Moon vy (Helio)', 'Moon vz (Helio)']].iloc[in_ems_idxs[0]]

                    # fig3 = plt.figure()
                    # ax = fig3.add_subplot(111, projection='3d')
                    # ut, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
                    # xw = 0.0038752837677 / r_sE * np.cos(ut) * np.sin(v) + 1
                    # yw = 0.0038752837677 / r_sE * np.sin(ut) * np.sin(v)
                    # zw = 0.0038752837677 / r_sE * np.cos(v)
                    # ax.plot_wireframe(xw, yw, zw, color="b", alpha=0.1)
                    # ax.scatter(1, 0, 0, color='blue', s=10)
                    # ax.plot3D(xs, ys, zs, color='grey', zorder=10)
                    # ax.scatter(xs[in_ems_idxs[0]], ys[in_ems_idxs[0]], zs[in_ems_idxs[0]], color='red', s=10)
                    # ax.scatter(xs[in_ems_idxs[0] + 10], ys[in_ems_idxs[0] + 10], zs[in_ems_idxs[0] + 10], color='orange', s=10)
                    # ax.plot3D([xs[in_ems_idxs[0]], xs[in_ems_idxs[0]] + vxs[in_ems_idxs[0]]], [ys[in_ems_idxs[0]], ys[in_ems_idxs[0]] + vys[in_ems_idxs[0]]],
                    #           [zs[in_ems_idxs[0]], zs[in_ems_idxs[0]] + vzs[in_ems_idxs[0]]], color='red', zorder=15)
                    #
                    # ax.set_xlim([0.99, 1.01])
                    # ax.set_ylim([-0.01, 0.01])
                    # ax.set_zlim([-0.01, 0.01])
                    # plt.show()
                    # fig = plt.figure()
                    # fac = 0.01
                    # plt.scatter(Earth_state[0], Earth_state[1], color='blue')
                    # plt.scatter(TCO_state[0], TCO_state[1], color='red')
                    # plt.plot([TCO_state[0], TCO_state[0] + fac * TCO_state[3]], [TCO_state[1], TCO_state[1] + fac * TCO_state[4]], color='blue')
                    # c1 = plt.Circle((Earth_state[0], Earth_state[1]), radius=0.0038752837677, alpha=0.1)
                    # plt.plot(data['Helio x'], data['Helio y'], color='grey')
                    # plt.gca().add_artist(c1)
                    # plt.xlim([-0.01 + Earth_state[0], 0.01 + Earth_state[0]])
                    # plt.gca().set_aspect('equal')
                    # plt.scatter(Earth_statem1[0], Earth_statem1[1], color='blue')
                    # plt.scatter(TCO_statem1[0], TCO_statem1[1], color='red')
                    # plt.plot([TCO_statem1[0], TCO_statem1[0] + fac * TCO_statem1[3]], [TCO_statem1[1], TCO_statem1[1] + fac * TCO_statem1[4]],
                    #          color='red')
                    #
                    # fig2 = plt.figure()
                    # ax = fig2.add_subplot(111, projection='3d')
                    # ax.scatter(Earth_state[0], Earth_state[1], Earth_state[2], color='blue')
                    # ax.scatter(TCO_state[0], TCO_state[1], TCO_state[2], color='red')
                    # ax.plot3D([TCO_state[0], TCO_state[0] + fac * TCO_state[3]],
                    #          [TCO_state[1], TCO_state[1] + fac * TCO_state[4]], [TCO_state[2], TCO_state[2] + fac * TCO_state[5]], color='blue')
                    #
                    # plt.show()
                    # EMS start
                    ems_start = data['Julian Date'].iloc[in_ems_idxs[0]]
                    ems_start_idx = in_ems_idxs[0]
                    # EMS end
                    ems_end = data['Julian Date'].iloc[in_ems_idxs[-1]]
                    ems_end_idx = in_ems_idxs[-1]

                else:
                    Earth_state = np.full(6, np.nan)
                    TCO_state = np.full(6, np.nan)
                    moon_state = np.full(6, np.nan)
                    # EMS start
                    ems_start = np.nan
                    ems_start_idx = np.nan
                    # EMS end
                    ems_end = np.nan
                    ems_end_idx = np.nan

                results = [stc, time_SOI_EMS, peri_in_SOI_EMS, peri_in_3_hill, peri_in_2_hill, peri_in_1_hill,
                           stc_start, stc_start_idx, stc_end, stc_end_idx, ems_start, ems_start_idx, ems_end,
                           ems_end_idx, TCO_state, Earth_state, moon_state]

                self.stc = stc
                self.t_ems = time_SOI_EMS
                self.peri_ems = peri_in_SOI_EMS
                self.peri_3hill = peri_in_3_hill
                self.peri_2hill = peri_in_2_hill
                self.peri_1hill = peri_in_1_hill
                self.stc_start = stc_start
                self.stc_start_idx = stc_start_idx
                self.stc_end = stc_end
                self.stc_end_idx = stc_end_idx
                self.ems_start = ems_start
                self.ems_start_idx = ems_start_idx
                self.ems_end = ems_end
                self.ems_end_idx = ems_end_idx

                print("STC: " + str(stc))
                print("Time spent in EMS: " + str(time_SOI_EMS))
                print("Number of periapsides in the EMS: " + str(peri_in_SOI_EMS))
                print("Number of periapsides in 3 hill: " + str(peri_in_3_hill))
                print("Number of periapsides in 2 hill: " + str(peri_in_2_hill))
                print("Number of periapsides in 1 hill: " + str(peri_in_1_hill))
                print("STC start: " + str(stc_start))
                print("STC end: " + str(stc_end))
                print("EMS start: " + str(ems_start))
                print("EMS end: " + str(ems_end))
                print("...Done")
                print("\n")

                # print(results)
                #
                # fig = plt.figure()
                # plt.plot(time, captured_distance, color='#5599ff', linewidth=3, zorder=5, label='Inside 3 Earth Hill')
                # plt.plot(time, captured_distance_ems, color='red', linewidth=5, zorder=6, label='Inside SOI of EMS')
                # plt.plot(time, distance_emb_synodic, color='grey',
                #          linewidth=1, zorder=7, label='TCO Trajectory')
                # plt.plot(time, stc_moon_dist_nomoon, linestyle='--', color='blue',
                #          label='STC Moon Distance without Moon')
                # plt.scatter(local_time, local_dist, color='#ff80ff', zorder=8, label='Periapsides Inside 3 Earth Hill')
                # plt.scatter(local_time_ems, local_dist_ems, color='blue', zorder=9, label='Periapsides Inside SOI of EMS')
                # if in_ems_idxs:
                #     plt.scatter(time.iloc[in_ems_idxs[0]], distance_emb_synodic[in_ems_idxs[0]], zorder=15)
                # plt.plot(time, three_eh_line, linestyle='--', color='red', zorder=4, label='3 Earth Hill')
                # plt.plot(time, two_eh_line, linestyle='--', zorder=4, label='2 Earth Hill')
                # plt.plot(time, one_eh_line, linestyle='--', zorder=4, label='1 Earth Hill')
                # plt.plot(time, ems_line, linestyle='--', color='green', zorder=3, label='SOI of EMS')
                # plt.legend()
                # plt.xlabel('Time (days)')
                # plt.ylabel('Distance from Earth/Moon Barycentre (AU)')
                # plt.title(str(object_id))
                # plt.ylim([0, 0.06])
                # plt.xlim([0, time.iloc[-1]])
                # plt.savefig("figures/" + str(object_id) + ".svg", format="svg")
                # plt.show()

        return results

    @staticmethod
    def calc_coherence(C_r_TCO, C_v_TCO, C_r_TCO_p1, C_v_TCO_p1):

        return (C_r_TCO_p1 - C_r_TCO) / np.linalg.norm(C_r_TCO_p1 - C_r_TCO) @ (C_v_TCO_p1 + C_v_TCO) / np.linalg.norm(
            C_v_TCO_p1 + C_v_TCO)

    def find_good_cr3bp_state(self, data, in_ems_idxs, mu):

        # look backwards until you find a good coherence
        seconds_in_day = 86400
        i = in_ems_idxs - 1
        coherence = 0
        found = 0

        rs = []
        coherences = []

        # while coherence < coherence_threshold:

        # state before entry
        h_r_TCO = np.array(
            [data['Helio x'].iloc[i], data['Helio y'].iloc[i], data['Helio z'].iloc[i]]).ravel()  # AU
        h_r_M = np.array([data['Moon x (Helio)'].iloc[i], data['Moon y (Helio)'].iloc[i],
                          data['Moon z (Helio)'].iloc[i]]).ravel()
        h_r_E = np.array([data['Earth x (Helio)'].iloc[i], data['Earth y (Helio)'].iloc[i],
                          data['Earth z (Helio)'].iloc[i]]).ravel()
        h_v_TCO = np.array(
            [data['Helio vx'].iloc[i], data['Helio vy'].iloc[i], data['Helio vz'].iloc[i]]).ravel()  # AU/day
        h_v_M = np.array([data['Moon vx (Helio)'].iloc[i], data['Moon vy (Helio)'].iloc[i],
                          data['Moon vz (Helio)'].iloc[i]]).ravel()
        h_v_E = np.array([data['Earth vx (Helio)'].iloc[i], data['Earth vy (Helio)'].iloc[i],
                          data['Earth vz (Helio)'].iloc[i]]).ravel()
        date_mjd = Time(data['Julian Date'].iloc[i], format='jd').to_value('mjd')

        C_r_TCO, C_v_TCO, C_v_TCO_2, C_ems, C_moon, ems_barycentre, vems_barycentre, omega, omega_2, r_sE, mu_s, mu_EMS = (
            get_r_and_v_cr3bp_from_nbody_sun_emb(h_r_TCO, h_v_TCO, h_r_E, h_v_E, h_r_M, h_v_M, date_mjd))

        # state at entry
        i = i + 1
        h_r_TCO_p1 = np.array(
            [data['Helio x'].iloc[i], data['Helio y'].iloc[i], data['Helio z'].iloc[i]]).ravel()  # AU
        h_r_M_p1 = np.array([data['Moon x (Helio)'].iloc[i], data['Moon y (Helio)'].iloc[i],
                             data['Moon z (Helio)'].iloc[i]]).ravel()
        h_r_E_p1 = np.array([data['Earth x (Helio)'].iloc[i], data['Earth y (Helio)'].iloc[i],
                             data['Earth z (Helio)'].iloc[i]]).ravel()
        h_v_TCO_p1 = np.array([data['Helio vx'].iloc[i], data['Helio vy'].iloc[i],
                               data['Helio vz'].iloc[i]]).ravel()  # AU/day
        h_v_M_p1 = np.array([data['Moon vx (Helio)'].iloc[i], data['Moon vy (Helio)'].iloc[i],
                             data['Moon vz (Helio)'].iloc[i]]).ravel()
        h_v_E_p1 = np.array([data['Earth vx (Helio)'].iloc[i], data['Earth vy (Helio)'].iloc[i],
                             data['Earth vz (Helio)'].iloc[i]]).ravel()
        date_mjd_p1 = Time(data['Julian Date'].iloc[i], format='jd').to_value('mjd')

        C_r_TCO_p1, C_v_TCO_p1, C_v_TCO_2_p1, C_ems_p1, C_moon_p1, ems_barycentre_p1, vems_barycentre_p1, omega_p1, omega_2_p1, r_sE_p1, mu_s_p1, mu_EMS_p1 = (
            get_r_and_v_cr3bp_from_nbody_sun_emb(h_r_TCO_p1, h_v_TCO_p1, h_r_E_p1, h_v_E_p1, h_r_M_p1, h_v_M_p1,
                                                 date_mjd_p1))

        # state after
        i = i + 1
        h_r_TCO_p2 = np.array(
            [data['Helio x'].iloc[i], data['Helio y'].iloc[i], data['Helio z'].iloc[i]]).ravel()  # AU
        h_r_M_p2 = np.array([data['Moon x (Helio)'].iloc[i], data['Moon y (Helio)'].iloc[i],
                             data['Moon z (Helio)'].iloc[i]]).ravel()
        h_r_E_p2 = np.array([data['Earth x (Helio)'].iloc[i], data['Earth y (Helio)'].iloc[i],
                             data['Earth z (Helio)'].iloc[i]]).ravel()
        h_v_TCO_p2 = np.array([data['Helio vx'].iloc[i], data['Helio vy'].iloc[i],
                               data['Helio vz'].iloc[i]]).ravel()  # AU/day
        h_v_M_p2 = np.array([data['Moon vx (Helio)'].iloc[i], data['Moon vy (Helio)'].iloc[i],
                             data['Moon vz (Helio)'].iloc[i]]).ravel()
        h_v_E_p2 = np.array([data['Earth vx (Helio)'].iloc[i], data['Earth vy (Helio)'].iloc[i],
                             data['Earth vz (Helio)'].iloc[i]]).ravel()
        date_mjd_p2 = Time(data['Julian Date'].iloc[i], format='jd').to_value('mjd')

        (C_r_TCO_p2, C_v_TCO_p2, C_v_TCO_2_p2, C_ems_p2, C_moon_p2, ems_barycentre_p2, vems_barycentre_p2, omega_p2,
         omega_2_p2, r_sE_p2, mu_s_p2, mu_EMS_p2) = (
            get_r_and_v_cr3bp_from_nbody_sun_emb(h_r_TCO_p2, h_v_TCO_p2,
                                                 h_r_E_p2, h_v_E_p2, h_r_M_p2, h_v_M_p2, date_mjd_p2))

        C_r_TCO_nondim = C_r_TCO / r_sE
        C_v_TCO_nondim = C_v_TCO/ (np.linalg.norm(omega) * r_sE)
        C_r_TCO_nondim_p1 = C_r_TCO_p1 / r_sE_p1
        C_v_TCO_nondim_p1 = C_v_TCO_p1 / (np.linalg.norm(omega_p1) * r_sE_p1)
        C_r_TCO_nondim_p2 = C_r_TCO_p2 / r_sE_p2
        C_v_TCO_nondim_p2 = C_v_TCO_p2 / (np.linalg.norm(omega_p1) * r_sE_p2)

        rs.append(C_r_TCO_nondim)

        # coherence before entry
        coherence = self.calc_coherence(C_r_TCO_nondim, C_v_TCO_nondim, C_r_TCO_nondim_p1, C_v_TCO_nondim_p1)
        coherence_p1 = self.calc_coherence(C_r_TCO_nondim_p1, C_v_TCO_nondim_p1, C_r_TCO_nondim_p2, C_v_TCO_nondim_p2)
        coherences.append(coherence)

        # if coherence > coherence_threshold:
        #     found = 1
        #     i = i + 1
        #     grab state
            # good_r_coh = C_r_TCO_nondim
            # good_v_coh = C_v_TCO_nondim
        #
        # i = i - 1
        #
        # if found:

        # integrate state until soi of ems + 1
        r_SOIEMS = 0.0038752837677 / r_sE
        index_span = in_ems_idxs - i  # also number of hours for hour long integration step
        additional = 1000
        num_days = (index_span + additional) * (
                    data["Julian Date"].iloc[1] - data["Julian Date"].iloc[0])
        space = 8000  # number of points to plot
        start = 0  # start time
        end_time = num_days * (np.linalg.norm(omega))
        time_span = np.linspace(start, end_time, space)  # over T instead of T/2
        state = np.hstack((C_r_TCO_nondim, C_v_TCO_nondim))
        # res = odeint(model, state, time_span, args=(mu,))

        # fig3 = plt.figure()
        # ax = fig3.add_subplot(111, projection='3d')
        # ut, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        # xw = 0.0038752837677 / r_sE * np.cos(ut) * np.sin(v) + 1
        # yw = 0.0038752837677 / r_sE * np.sin(ut) * np.sin(v)
        # zw = 0.0038752837677 / r_sE * np.cos(v)
        # ax.plot_wireframe(xw, yw, zw, color="b", alpha=0.1, label='SOI of EMS')
        # ax.scatter(1, 0, 0, color='blue', s=50, label='Earth-Moon barycentre')
        # ax.plot3D(res[:, 0], res[:, 1], res[:, 2], linewidth=1, zorder=10)
        # ax.scatter([C_r_TCO_nondim[0], C_r_TCO_nondim_p1[0], C_r_TCO_nondim_p2[0]],
        #            [C_r_TCO_nondim[1], C_r_TCO_nondim_p1[1], C_r_TCO_nondim_p2[1]],
        #            [C_r_TCO_nondim[2], C_r_TCO_nondim_p1[2], C_r_TCO_nondim_p2[2]], linewidth=3, zorder=5)
        #
        # print(coherence)
        # print(coherence_p1)
        # plt.show()

        C_r_TCO_nondim_final = np.array([np.nan, np.nan, np.nan])
        C_v_TCO_nondim_final = np.array([np.nan, np.nan, np.nan])

        for t in time_span:

            # solve ODE
            new_time_span = np.linspace(start, t, space)

            res = odeint(model, state, new_time_span, args=(mu,))[-1]

            dist = np.linalg.norm(np.array([res[0] - 1, res[1], res[2]]))

            # Check the stop condition (for example, if the solution crosses a certain threshold)
            if dist < r_SOIEMS:
                C_r_TCO_nondim_final = np.array(res[0:3])
                C_v_TCO_nondim_final = np.array(res[3:])
                break

        return C_r_TCO_nondim_final, C_v_TCO_nondim_final

    def alpha_beta_jacobi(self, object_id):

        seconds_in_day = 86400
        km_in_au = 149597870700 / 1000

        # go through all the files of test particles
        population_dir = os.path.join(os.getcwd(), 'minimoon_files_oorb')
        # population_dir = os.path.join(os.getcwd(), 'Test_Set')
        population_file = 'minimoon_master_final (copy).csv'
        population_file_path = population_dir + '/' + population_file

        mm_parser = MmParser("", population_dir, "")

        # Initial: will be result if no file found with that name
        # results = np.array((len(object_id), 3))

        full_master = mm_parser.parse_master(population_file_path)
        master = full_master[full_master['Object id'] == object_id]
        print(object_id)

        x = master['Helio x at EMS']  # AU
        y = master['Helio y at EMS']
        z = master['Helio z at EMS']
        vx = master['Helio vx at EMS']  # AU/day
        vy = master['Helio vy at EMS']
        vz = master['Helio vz at EMS']
        vx_M = master['Moon vx at EMS (Helio)']  # AU/day
        vy_M = master['Moon vy at EMS (Helio)']
        vz_M = master['Moon vz at EMS (Helio)']
        x_M = master['Moon x at EMS (Helio)']  # AU
        y_M = master['Moon y at EMS (Helio)']
        z_M = master['Moon z at EMS (Helio)']
        x_E = master['Earth x at EMS (Helio)']
        y_E = master['Earth y at EMS (Helio)']
        z_E = master['Earth z at EMS (Helio)']
        vx_E = master['Earth vx at EMS (Helio)']  # AU/day
        vy_E = master['Earth vy at EMS (Helio)']
        vz_E = master['Earth vz at EMS (Helio)']
        date_ems = master['Entry Date to EMS'].iloc[0]  # Julian date

        if not np.isnan(date_ems):

            in_ems_idxs = int(master['Entry to EMS Index'].iloc[0])
            date_mjd = Time(date_ems, format='jd').to_value('mjd')

            name = str(object_id) + ".csv"

            data = mm_parser.mm_file_parse_new(population_dir + '/' + name)

            h_r_TCO = np.array([x, y, z]).ravel()  # AU
            h_r_M = np.array([x_M, y_M, z_M]).ravel()
            h_r_E = np.array([x_E, y_E, z_E]).ravel()
            h_v_TCO = np.array([vx, vy, vz]).ravel()  # AU/day
            h_v_M = np.array([vx_M, vy_M, vz_M]).ravel()
            h_v_E = np.array([vx_E, vy_E, vz_E]).ravel()

            C_r_TCO, C_v_TCO, C_v_TCO_2, C_ems, C_moon, ems_barycentre, vems_barycentre, omega, omega_2, r_sE, mu_s, mu_EMS = (
                get_r_and_v_cr3bp_from_nbody_sun_emb(h_r_TCO, h_v_TCO, h_r_E, h_v_E, h_r_M, h_v_M, date_mjd))

            # h_r_TCO_p1 = np.array([data['Helio x'].iloc[in_ems_idxs + 1], data['Helio y'].iloc[in_ems_idxs + 1], data['Helio z'].iloc[in_ems_idxs + 1]]).ravel()  # AU
            # h_r_M_p1 = np.array([data['Moon x (Helio)'].iloc[in_ems_idxs + 1], data['Moon y (Helio)'].iloc[in_ems_idxs + 1], data['Moon z (Helio)'].iloc[in_ems_idxs + 1]]).ravel()
            # h_r_E_p1 = np.array([data['Earth x (Helio)'].iloc[in_ems_idxs + 1], data['Earth y (Helio)'].iloc[in_ems_idxs + 1], data['Earth z (Helio)'].iloc[in_ems_idxs + 1]]).ravel()
            # h_v_TCO_p1 = np.array([data['Helio vx'].iloc[in_ems_idxs + 1], data['Helio vy'].iloc[in_ems_idxs + 1], data['Helio vz'].iloc[in_ems_idxs + 1]]).ravel()  # AU/day
            # h_v_M_p1 = np.array([data['Moon vx (Helio)'].iloc[in_ems_idxs + 1], data['Moon vy (Helio)'].iloc[in_ems_idxs + 1], data['Moon vz (Helio)'].iloc[in_ems_idxs + 1]]).ravel()
            # h_v_E_p1 = np.array([data['Earth vx (Helio)'].iloc[in_ems_idxs + 1], data['Earth vy (Helio)'].iloc[in_ems_idxs + 1], data['Earth vz (Helio)'].iloc[in_ems_idxs + 1]]).ravel()
            # date_mjd_p1 = Time(data['Julian Date'].iloc[in_ems_idxs + 1], format='jd').to_value('mjd')

            # C_r_TCO_p1, C_v_TCO_p1, C_v_TCO_2_p1, C_ems_p1, C_moon_p1, ems_barycentre_p1, vems_barycentre_p1, omega_p1, omega_2_p1, r_sE_p1, mu_s_p1, mu_EMS_p1 = (
            #     get_r_and_v_cr3bp_from_nbody_sun_emb(h_r_TCO_p1, h_v_TCO_p1, h_r_E_p1, h_v_E_p1, h_r_M_p1, h_v_M_p1, date_mjd_p1))
            mu = mu_EMS / (mu_EMS + mu_s)

            # C_r_TCO_nondim = C_r_TCO / r_sE
            # C_v_TCO_nondim = C_v_TCO / (np.linalg.norm(omega) * r_sE)
            # C_v_TCO_2_nondim = C_v_TCO_2 / (np.linalg.norm(omega_2) * r_sE)
            # C_r_TCO_nondim_p1 = C_r_TCO_p1 / r_sE_p1
            # C_v_TCO_nondim_p1 = C_v_TCO_p1 / (np.linalg.norm(omega_p1) * r_sE_p1)
            # C_v_TCO_2_nondim_p1 = C_v_TCO_2_p1 / (np.linalg.norm(omega_2_p1) * r_sE_p1)

            # coherence = self.calc_coherence(C_r_TCO_nondim, C_v_TCO_nondim, C_r_TCO_nondim_p1, C_v_TCO_nondim_p1)
            # coherence_2 = self.calc_coherence(C_r_TCO_nondim, C_v_TCO_2_nondim, C_r_TCO_nondim_p1, C_v_TCO_2_nondim_p1)
            # coherence_threshold = 0.99
            # if coherence > coherence_2 and coherence > coherence_threshold:
            #     good_r = C_r_TCO
            #     good_v = C_v_TCO
            #     good_omega = omega
            #     C_J_dimensional, C_J_nondimensional = jacobi_dim_and_non_dim(good_r, good_v, h_r_TCO, ems_barycentre,
            #                                                                  mu,
            #                                                                  mu_s, mu_EMS, good_omega, r_sE)
            #     good_r = good_r / r_sE
            #     good_v = good_v / (np.linalg.norm(good_omega) * r_sE)
            # elif coherence_2 >= coherence and coherence_2 > coherence_threshold:
            #     good_r = C_r_TCO
            #     good_v = C_v_TCO_2
            #     good_omega = omega_2
            #     C_J_dimensional, C_J_nondimensional = jacobi_dim_and_non_dim(good_r, good_v, h_r_TCO, ems_barycentre,
            #                                                                  mu,
            #                                                                  mu_s, mu_EMS, good_omega, r_sE)
            #     good_r = good_r / r_sE
            #     good_v = good_v / (np.linalg.norm(good_omega) * r_sE)
            #
            # else:
            #     good_r, good_v = self.find_good_cr3bp_state(data, in_ems_idxs, mu)
            #     if np.isnan(good_r[0]):
            #         good_r = C_r_TCO
            #         good_v = C_v_TCO
            #         good_omega = omega
            #         C_J_dimensional, C_J_nondimensional = jacobi_dim_and_non_dim(good_r, good_v, h_r_TCO,
            #                                                                      ems_barycentre,
            #                                                                      mu,
            #                                                                      mu_s, mu_EMS, good_omega, r_sE)
            #         good_r = good_r / r_sE
            #         good_v = good_v / (np.linalg.norm(good_omega) * r_sE)
            #     else:
            #         good_omega = omega
            #         C_J_nondimensional = jacobi(np.hstack((good_r, good_v)), mu)
            #         C_J_dimensional = C_J_nondimensional * (r_sE * km_in_au) ** 2 * (np.linalg.norm(good_omega) / seconds_in_day) ** 2


            good_r = C_r_TCO
            good_v = C_v_TCO_2
            good_omega = omega_2
            C_J_dimensional, C_J_nondimensional = jacobi_dim_and_non_dim(good_r, good_v, h_r_TCO, ems_barycentre,
                                                                         mu, mu_s, mu_EMS, good_omega, r_sE)
            # angle between x-axis of C and TCO at SOI of EMS centered at EMS barycentre
            alpha = np.rad2deg(np.arctan2((good_r[1] - C_ems[1]), (good_r[0] - C_ems[0])))
            if alpha < 0:
                alpha += 360

            # angle between x-axis of C and Moon when TCO is at SOI of EMS
            theta_M = np.rad2deg(np.arctan2((C_moon[1] - C_ems[1]), (C_moon[0] - C_ems[0])))
            if theta_M < 0:
                theta_M += 360

            # to calculate beta, first calculate psi, the angle from the x-axis of C centered at EMS-barycentre to the
            # vector described by the velocity of the TCO in the C frame
            psi = np.rad2deg(np.arctan2(good_v[1], good_v[0]))
            if psi < 0:
                psi += 360
            beta = (psi - 90 - alpha)  # negative to follow Qi convention
            if beta < 0:
                beta += 360
            beta = -beta  # negative to follow Qi convention

            results = [C_J_dimensional, C_J_nondimensional, alpha, beta, theta_M]

            print(results)
            print(str(object_id))
            print(psi)

            # fig3 = plt.figure()
            # ax = fig3.add_subplot(111, projection='3d')
            # ut, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
            # xw = 0.0038752837677 / r_sE * np.cos(ut) * np.sin(v) + 1
            # yw = 0.0038752837677 / r_sE * np.sin(ut) * np.sin(v)
            # zw = 0.0038752837677 / r_sE * np.cos(v)
            # ax.plot_wireframe(xw, yw, zw, color="b", alpha=0.1, label='SOI of EMS')
            # ax.scatter(1, 0, 0, color='blue', s=50, label='Earth-Moon barycentre')
            # ax.scatter(good_r[0], good_r[1], good_r[2], color='red', s=10, zorder=5,
            #            label='r')
            # ax.plot3D([good_r[0], good_r[0] + good_v[0]], [good_r[1], good_r[1] + good_v[1]],
            #           [good_r[2], good_r[2] + good_v[2]], color='red', zorder=15)
            # ax.scatter(C_r_TCO_nondim[0], C_r_TCO_nondim[1], C_r_TCO_nondim[2], color='blue', s=5, zorder=10,
            #            label='r')
            # ax.plot3D([C_r_TCO_nondim[0], C_r_TCO_nondim[0] + C_v_TCO_nondim[0]], [C_r_TCO_nondim[1], C_r_TCO_nondim[1] + C_v_TCO_nondim[1]],
            #           [C_r_TCO_nondim[2], C_r_TCO_nondim[2] + C_v_TCO_nondim[2]], color='blue', zorder=15)
            #
            # ax.set_xlim([0.99, 1.01])
            # ax.set_ylim([-0.01, 0.01])
            # ax.set_zlim([-0.01, 0.01])
            # num_ticks = 3
            # ax.xaxis.set_major_locator(ticker.MaxNLocator(num_ticks))
            # ax.yaxis.set_major_locator(ticker.MaxNLocator(num_ticks))
            # ax.zaxis.set_major_locator(ticker.MaxNLocator(num_ticks))
            # ax.set_xlabel('Synodic x ($\emptyset$)')
            # ax.set_ylabel('Synodic y ($\emptyset$)')
            # ax.set_zlabel('Synodic z ($\emptyset$)')
            # ax.legend()
            # plt.show()

            # fig2 = plt.figure()
            # plt.scatter(C_ems[0], C_ems[1], color='blue')
            # c1 = plt.Circle((C_ems[0], C_ems[1]), radius=0.0038752837677, alpha=0.1)
            # plt.scatter(C_r_TCO[0], C_r_TCO[1], color='red')
            # plt.plot([C_r_TCO[0], C_r_TCO[0] + C_v_TCO_2[0]], [C_r_TCO[1], C_r_TCO[1] + C_v_TCO_2[1]], color='red')
            # plt.gca().add_artist(c1)
            # plt.xlim([C_ems[0] - 0.01, C_ems[0] + 0.01])
            # plt.ylim([C_ems[1] - 0.01, C_ems[1] + 0.01])
            # plt.gca().set_aspect('equal')
            # plt.title(str(object_id))

            # fig3 = plt.figure()
            # plt.scatter(1, 0, color='blue')
            # c1 = plt.Circle((1, 0), radius=0.0038752837677 / r_sE, alpha=0.1)
            # plt.scatter(x_prime, y_prime, color='red')
            # plt.plot([x_prime, x_prime + x_dot_prime], [y_prime, y_prime + y_dot_prime], color='red')
            # plt.gca().add_artist(c1)
            # plt.xlim([0.99, 1.01])
            # plt.ylim([-0.01, 0.01])
            # plt.gca().set_aspect('equal')

            # fig3 = plt.figure()
            # ax = fig3.add_subplot(111, projection='3d')
            # ut, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
            # x = 0.0038752837677 / r_sE * np.cos(ut) * np.sin(v) + C_ems[0] / r_sE
            # y = 0.0038752837677 / r_sE * np.sin(ut) * np.sin(v) + C_ems[1] / r_sE
            # z = 0.0038752837677 / r_sE * np.cos(v) + C_ems[2] / r_sE
            # ax.plot_wireframe(x, y, z, color="b", alpha=0.1)
            # ax.scatter(C_ems[0] / r_sE, C_ems[1] / r_sE, C_ems[2] / r_sE, color='blue', s=10)
            # ax.scatter(x_prime, y_prime, z_prime, color='red', s=10)
            # ax.plot3D([x_prime, x_prime + x_dot_prime], [y_prime, y_prime + y_dot_prime], [z_prime, z_prime + z_dot_prime], color='red')
            # ax.plot3D([C_])

            # fig = plt.figure()
            # plt.scatter(ems_barycentre[0], ems_barycentre[1], color='red')
            # plt.scatter(h_r_TCO[0], h_r_TCO[1], color='blue')
            # plt.plot([ems_barycentre[0], h_r_M[0]], [ems_barycentre[1], h_r_M[1]], color='orange')
            # plt.plot([ems_barycentre[0], h_r_TCO[0]], [ems_barycentre[1], h_r_TCO[1]], color='red')
            # plt.plot([h_r_TCO[0], h_r_TCO[0] + C_v_TCO[0]], [h_r_TCO[1], h_r_TCO[1] + C_v_TCO[1]], color='green')
            # plt.plot([h_r_TCO[0], h_r_TCO[0] + h_v_TCO[0]], [h_r_TCO[1], h_r_TCO[1] + h_v_TCO[1]], color='grey')
            # plt.plot([r_C[0], r_C[0] + C_R_h[0, 0]], [r_C[1], r_C[1] + C_R_h[0, 1]], color='red')
            # plt.plot([r_C[0], r_C[0] + C_R_h[1, 0]], [r_C[1], r_C[1] + C_R_h[1, 1]], color='green')
            # plt.xlim([-1.5, 1.5])
            # plt.ylim([-1.5, 1.5])
            # plt.gca().set_aspect('equal')
            # plt.plot([r_C[0], ems_barycentre[0]], [r_C[1], ems_barycentre[1]])
            # plt.show()
        else:
            results = [np.nan, np.nan, np.nan, np.nan, np.nan]

        return results

    def get_cluster_data(self, object_id):

        # go through all the files of test particles
        population_dir = os.path.join(os.getcwd(), 'minimoon_files_oorb')
        # population_dir = os.path.join(os.getcwd(), 'Test_Set')
        population_file = 'minimoon_master_final.csv'
        population_file_path = population_dir + '/' + population_file

        mm_parser = MmParser("", population_dir, "")

        # Initial: will be result if no file found with that name
        # results = np.array((len(object_id), 3))

        full_master = mm_parser.parse_master(population_file_path)
        master_i = full_master[full_master['Object id'] == object_id]
        name = str(object_id) + ".csv"
        data_i = mm_parser.mm_file_parse_new(population_dir + '/' + name)

        # variables
        one_hill = 0.01
        seconds_in_day = 86400
        km_in_au = 149597870700 / 1000
        mu_e = 3.986e5 * seconds_in_day ** 2 / np.power(km_in_au, 3)  # km^3/s^2

        # indices of trajectory when insides one hill
        in_1hill_idxs = [index for index, value in enumerate(data_i['Distance']) if value <= one_hill]
        # Get 1 Hill entrance and last exit indices

        # Get index of minimum distance and min dist.
        min_dist = min(data_i['Distance'])
        min_dist_index = data_i.index[data_i['Distance'] == min_dist]

        # Get time to minimum distance
        time_to_min_dist = data_i['Julian Date'].iloc[min_dist_index] - master_i['Capture Date'].iloc[0]



        # Get min specific energy
        spec_energy_in_one_hill_temp = [(np.linalg.norm([data_i['Geo vx'].iloc[i], data_i['Geo vy'].iloc[i],
                                        data_i['Geo vz'].iloc[i]]) ** 2 / 2 - mu_e / data_i['Distance'].iloc[i]) * km_in_au ** 2 / seconds_in_day ** 2
                                        for i, value in data_i.iterrows()]

        if in_1hill_idxs:
            one_hill_start_idx = in_1hill_idxs[0]
            one_hill_end_idx = in_1hill_idxs[-1]

            # Get Eccentricity during 1 Hill
            geo_e_in_one_hill = data_i['Geo e'].iloc[one_hill_start_idx:one_hill_end_idx]

            # Get perihelion during 1 Hill
            geo_q_in_one_hill = data_i['Geo q'].iloc[one_hill_start_idx:one_hill_end_idx]

            # Get inclination during 1 Hill
            geo_i_in_one_hill = data_i['Geo i'].iloc[one_hill_start_idx:one_hill_end_idx]

            # Calculate means
            geo_e_mean = np.mean(geo_e_in_one_hill)
            geo_i_mean = np.mean(geo_i_in_one_hill)
            geo_q_mean = np.mean(geo_q_in_one_hill)

            # Calculate the std deviation
            geo_e_std_u = np.std(geo_e_in_one_hill)
            geo_q_std_u = np.std(geo_q_in_one_hill)
            geo_i_std_u = np.std(geo_i_in_one_hill)
            #in percent
            geo_e_std = geo_e_std_u / geo_e_mean * 100
            geo_i_std = geo_i_std_u / geo_i_mean * 100
            geo_q_std = geo_q_std_u / geo_q_mean * 100

            spec_energy_in_one_hill = spec_energy_in_one_hill_temp[one_hill_start_idx:one_hill_end_idx]
            min_spec_energy = min(spec_energy_in_one_hill)
            min_spec_energy_ind = pd.Series(spec_energy_in_one_hill).idxmin()


        fig = plt.figure()
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.plot(data_i['Julian Date'] - data_i['Julian Date'].iloc[0], data_i['Distance'], label='Geocentric Distance', color='grey', linewidth=1)
        ax1.scatter(data_i['Julian Date'].iloc[min_dist_index] - data_i['Julian Date'].iloc[0], data_i['Distance'].iloc[min_dist_index], color='black', label='Days to Min. Dist.: ' + str(round(time_to_min_dist.iloc[0], 2)))
        ax1.scatter(master_i['Capture Date'] - data_i['Julian Date'].iloc[0], data_i['Distance'].iloc[master_i['Capture Index']], color='yellow', label='Capture Start')
        ax1.set_ylabel('Distance to Earth (AU)')
        ax1.set_xlabel('Time (days)')
        ax1.set_title(str(object_id))

        ax2 = ax1.twinx()
        ax2.plot(data_i['Julian Date'] - data_i['Julian Date'].iloc[0], spec_energy_in_one_hill_temp, color='tab:purple')

        if in_1hill_idxs:
            ax1.scatter(data_i['Julian Date'].iloc[one_hill_start_idx] - data_i['Julian Date'].iloc[0], data_i['Distance'].iloc[one_hill_start_idx], color='red', label='One Hill Start')
            ax1.scatter(data_i['Julian Date'].iloc[one_hill_end_idx] - data_i['Julian Date'].iloc[0] , data_i['Distance'].iloc[one_hill_end_idx],
                        color='blue', label='One Hill End')
            ax1.plot(data_i['Julian Date'].iloc[one_hill_start_idx:one_hill_end_idx] - data_i['Julian Date'].iloc[0],
                     data_i['Distance'].iloc[one_hill_start_idx:one_hill_end_idx],
                     label='Days in 1 Hill: ' + str(round(master_i['1 Hill Duration'].iloc[0], 2)), color='green', linewidth=3)

            ax2.scatter(data_i['Julian Date'].iloc[min_spec_energy_ind + one_hill_start_idx] - data_i['Julian Date'].iloc[0], min_spec_energy, color='pink', label='Min. Spec Energy')
        ax2.set_ylabel('Spec. Energy ($km^2/s^2$)', color='tab:purple')
        ax2.tick_params(axis='y', labelcolor='tab:purple')
        ax1.legend(loc='upper right')
        ax1.set_ylim([0, 0.03])
        ax2.legend(loc='upper left')
        ax2.set_ylim([-1, 1])

        if in_1hill_idxs:
            ax3 = fig.add_subplot(2, 3, 2, projection='3d')
            ax3.plot(data_i['Geo x'].iloc[one_hill_start_idx:one_hill_end_idx], data_i['Geo y'].iloc[one_hill_start_idx:one_hill_end_idx], data_i['Geo z'].iloc[one_hill_start_idx:one_hill_end_idx], color='grey', label='i var: ' + str(round(geo_i_std, 2)) + '%\n' + 'q var: ' + str(round(geo_q_std, 2)) + '%\n' + 'e var: ' + str(round(geo_e_std, 2)) + '%')
            ax3.scatter(data_i['Geo x'].iloc[one_hill_start_idx],  data_i['Geo y'].iloc[one_hill_start_idx],
                     data_i['Geo z'].iloc[one_hill_start_idx], color='green',  label='Start')
            ax3.scatter(data_i['Geo x'].iloc[one_hill_end_idx], data_i['Geo y'].iloc[one_hill_end_idx],
                        data_i['Geo z'].iloc[one_hill_end_idx], color='red', label='End')

            num_ticks = 3
            ax3.xaxis.set_major_locator(ticker.MaxNLocator(num_ticks))
            ax3.yaxis.set_major_locator(ticker.MaxNLocator(num_ticks))
            ax3.zaxis.set_major_locator(ticker.MaxNLocator(num_ticks))
            ax3.set_xlabel('Geo x (AU)')
            ax3.set_ylabel('Geo y (AU)')
            ax3.set_zlabel('Geo z (AU)')
            ax3.legend()

            ax4 = fig.add_subplot(2, 3, 4)
            ax4.plot(data_i['Julian Date'].iloc[one_hill_start_idx:one_hill_end_idx] - data_i['Julian Date'].iloc[0], geo_q_in_one_hill)
            ax4.set_xlabel('Time (days)')
            ax4.set_ylabel('Geo. Osc. q (AU)')

            ax5 = fig.add_subplot(2, 3, 3)
            ax5.plot(data_i['Julian Date'].iloc[one_hill_start_idx:one_hill_end_idx] - data_i['Julian Date'].iloc[0], geo_i_in_one_hill)
            ax5.set_xlabel('Time (days)')
            ax5.set_ylabel('Geo. Osc. i ($\circ$)')

            ax6 = fig.add_subplot(2, 3, 5)
            ax6.plot(data_i['Julian Date'].iloc[one_hill_start_idx:one_hill_end_idx] - data_i['Julian Date'].iloc[0],
                     geo_e_in_one_hill)
            ax6.set_xlabel('Time (days)')
            ax6.set_ylabel('Geo. Osc. e')



        plt.show()



        # Get min distance

        # Get periapsidies in SOI of EMS

        # Get 1 Hill duration

        # Get TCO state vector at capture

        # Get moon state vector at capture

        # Get earth state vector at capture

        # Get epoch at capture


        # Add relevant to master

        # Store new dataframe as csv

        return