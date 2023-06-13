from astropy.time import Time
import pandas as pd
from space_fncs import get_eclip_long
import astropy.units as u
import numpy
from space_fncs import eci_ecliptic_to_sunearth_synodic
from space_fncs import get_M
from space_fncs import getH2D
# import pyoorb
import numpy as np
import os
from astropy.time import Time
# from astroquery.jplhorizons import Horizons
import sys
import matplotlib.pyplot as plt
from astropy.units import cds
from astropy import constants as const
from MM_Parser import MmParser

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
        mu = grav_param
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
                    if ((j > (steps-capture_idx - 2100)) and (j < (steps - capture_idx + 2100))):
                        if j < steps - 1:
                            print("Steps: " + str(steps))
                            print("cap id: " + str(capture_idx))
                            # Store date of end of capture
                            prev_capture = 0
                            release_date = strt_tm + (j * conv_day) * u.day
                            release_idx = j
                            ident = 1
                            print(release_idx)
            elif ident == 0 and (j == steps - 1):
                release_date = strt_tm + (steps * conv_day) * u.day
                release_idx = steps - 1
                print(release_idx)

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
        min_distances[0, 0] = min(distance[0, capture_idx:release_idx])
        max_distance = max(distance[0, capture_idx:release_idx])
        revolutions[0, 0] = cum_angle_ecl_jpl / 360.0
        if abs(revolutions[0, 0]) >= 1:
            satisfied_3[0, 0] = 1

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
            self.retrograde = True
        else:
            print("Orbit is prograde")
            self.retrograde = False
        print("Minimum distance reached to observer (AU): " + str(min_distances[0, 0]))
        print("...Done")
        print("\n")

        # Properties
        self.capture_start = capture_date
        self.capture_end = release_date
        self.capture_duration = time_captured[0, 0]
        self.three_eh_duration = time_satisfied_1[0, 0]
        self.minimoon_flag = True if time_captured > 0 and satisfied_3[0, 0] == 1 and satisfied_4_overall[0, 0] == 1 \
            else False
        self.epsilon_duration = time_satisfied_2[0, 0]
        self.revolutions = revolutions[0, 0]
        self.one_eh_flag = True if satisfied_3[0, 0] == 1 else False
        self.one_eh_duration = time_satisfied_4[0, 0]
        self.min_dist = min_distances[0, 0]
        self.rev_flag = True if satisfied_3[0, 0] == 1 else False
        self.cap_idx = int(capture_idx)
        self.rel_idx = int(release_idx)
        self.max_dist = max_distance

        # fig = plt.figure()
        # plt.plot(data["Julian Date"], data["Geo e"], color='blue', label='Geo e')
        # plt.plot(data["Julian Date"], satisfied_2[0, :], color='red', label='Spec. Energy')
        # plt.legend()
        # plt.show()

        return

    def taxonomy(self, data, master):
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
            # yes? type 2-B
            designation = "2B"

        # no? was its distance greater than 1 Hill radius for entire capture?
        elif all(ele >= one_eh for ele in cap_dist):
            # yes? type 2-A
            designation = "2A"

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

    # def get_data_mm_oorb_w_horizons(self, mm_parser, data, int_step, perturbers, start_time, end_time, grav_param, minimoon):
    #     """
    #     Retrieve data, organize as pandas dataframe, store to file
    #
    #     :param minimoon:
    #     :param data:
    #     :param int_step:
    #     :param perturbers:
    #     :param start_time:
    #     :param end_time:
    #     :return:
    #     """
    #     ######################
    #     ### open orb
    #     #######################
    #
    #     # Integration step (in days)
    #     if int_step == 1:
    #         int_step_unit = 'd'
    #         int_step_jpl = 1
    #     elif int_step == 1 / 24:
    #         int_step_unit = 'h'
    #         int_step_jpl = 1
    #     elif int_step == 1/24/60:
    #         int_step_unit = 'm'
    #         int_step_jpl = 1
    #     else:
    #         int_step_unit = 'd'
    #         int_step_jpl = int_step
    #
    #     int_step_jpl_unit = int_step_unit
    #
    #     # Observation locations
    #     earth_obs = '500'  # Geocentric earth (works for both openorb and jpl)
    #     sun_obs = '500@10'
    #
    #     # Number of orbits
    #     n = 1
    #
    #     print("Generating data using Oorb for minimoon:" + minimoon + "...")
    #     ephfile = ""
    #     if os.getenv('OORB_DATA'):
    #         ephfile = os.path.join(os.getenv('OORB_DATA'), 'de430.dat')
    #     pyoorb.pyoorb.oorb_init(ephfile)
    #     # orb is id, 6 elements, epoch_mjd, H, G, element type index
    #     # keplerian appears to be element type index 3
    #     # orbits = np.array([0.,1.,2.,3.,4.,5.,6.,5373.,1.,1.,3.])
    #     # using the first item in PS-SSM, 1/100th density, s1.01 file.
    #     orbit = np.zeros([n, 12], dtype=np.double, order='F')
    #
    #     # Initialize
    #     if minimoon == '2006 RH120':
    #         # RH120 orbit - id = 1
    #         # For Keplerian orbit initialization - you have to specify the heliocentric ecliptic osculating elements
    #         # from the JPL horizons file !!According to starting date!!
    #         orbit[0][0] = 1.0
    #         orbit[0][1] = 0.9977199096279007  # semimajor axis
    #         orbit[0][2] = 0.03744067182298642  # eccentricity
    #         orbit[0][3] = np.deg2rad(0.3093522132878884)  # Inclination (rad)
    #         orbit[0][4] = np.deg2rad(323.0881705631016)  # Longitude of Ascending node (rad)
    #         orbit[0][5] = np.deg2rad(206.3169504727512)  # Argument of Periapsis (rad)
    #         orbit[0][6] = np.deg2rad(294.6211657400694)  # Mean anomaly (rad)
    #         orbit[0][7] = 3.0  # Type of orbit ID 1:Cartesian 2:Cometary 3:Keplerian
    #         orbit[0][8] = Time(2454101.5, format='jd').to_value('mjd', 'long')  # Epoch of perihelion (MJD)
    #         orbit[0][9] = 1.0  # timescale type of the epochs provided; integer value: UTC: 1, UT1: 2, TT: 3, TAI: 4
    #         orbit[0][10] = 29.5  # absolute magnitude of object (H)
    #         orbit[0][11] = 0.15  # photometric slope parameter of the target - G from HG model
    #     elif minimoon == '2020 CD3':
    #         # CD3 orbit - id = 1
    #         # For Keplerian orbit initialization - you have to specify the heliocentric ecliptic osculating elements from
    #         # the JPL horizons file
    #         orbit[0][0] = 1.0
    #         orbit[0][1] = 1.022115305313184  # semimajor axis
    #         orbit[0][2] = 0.03494529757096312  # eccentricity
    #         orbit[0][3] = np.deg2rad(0.7388576324362984)  # Inclination (rad)
    #         orbit[0][4] = np.deg2rad(134.7432090092998)  # Longitude of Ascending node (rad)
    #         orbit[0][5] = np.deg2rad(342.2597100247675)  # Argument of Periapsis (rad)
    #         orbit[0][6] = np.deg2rad(45.87779192987005)  # Mean anomaly (rad)
    #         orbit[0][7] = 3.0  # Type of orbit ID 1:Cartesian 2:Cometary 3:Keplerian
    #         orbit[0][8] = Time(2458914.5, format='jd').to_value('mjd', 'long')  # Epoch of perihelion (MJD)
    #         orbit[0][9] = 1.0  # timescale type of the epochs provided; integer value: UTC: 1, UT1: 2, TT: 3, TAI: 4
    #         orbit[0][10] = 31.74  # absolute magnitude of object (H)
    #         orbit[0][11] = 0.15  # photometric slope parameter of the target - G from HG model
    #     else:
    #         # A synthetic minimoon orbit
    #         # For Keplerian orbit initialization - you have to specify the heliocentric ecliptic osculating elements from
    #         # the JPL horizons file
    #         orbit[0][0] = 1.0
    #         orbit[0][1] = data["Helio x"].iloc[0]  # x
    #         orbit[0][2] = data["Helio y"].iloc[0]  # y
    #         orbit[0][3] = data["Helio z"].iloc[0]  # z
    #         orbit[0][4] = data["Helio vx"].iloc[0]  # vx
    #         orbit[0][5] = data["Helio vy"].iloc[0]  # vy
    #         orbit[0][6] = data["Helio vz"].iloc[0]  # vz
    #         orbit[0][7] = 1.  # Type of orbit ID 1:Cartesian 2:Cometary 3:Keplerian
    #         orbit[0][8] = Time(data["Julian Date"].iloc[0], format='jd').to_value('mjd', 'long')  # Epoch of osculating
    #         orbit[0][9] = 1.0  # timescale type of the epochs provided; integer value: UTC: 1, UT1: 2, TT: 3, TAI: 4
    #         orbit[0][10] = mm_parser.mm_data["x7"].iloc[2]  # absolute magnitude of object (H)
    #         orbit[0][11] = 0.15  # photometric slope parameter of the target - G from HG model
    #
    #     #############################################
    #     # Integrations
    #     ############################################
    #
    #     # Open orb generates eph
    #
    #     #print(start_time)
    #
    #     # Time and observer information
    #     obscode = earth_obs  # Where are you observing from: https://minorplanetcenter.net/iau/lists/ObsCodesF.html
    #     mjds = np.arange(Time(start_time, format='isot', scale='utc').to_value('mjd', 'long'),
    #                      Time(end_time, format='isot', scale='utc').to_value('mjd', 'long'), int_step)
    #     epochs = np.array(list(zip(mjds, [1] * len(mjds))), dtype=np.double, order='F')
    #
    #     # Check output format from: https://github.com/oorb/oorb/tree/master/python
    #     print("calling oorb_ephemeris_full (n-body)")
    #     eph, err = pyoorb.pyoorb.oorb_ephemeris_full(in_orbits=orbit[0][:].reshape(1, 12),
    #                                                  in_obscode=obscode,
    #                                                  in_date_ephems=epochs,
    #                                                  in_dynmodel='N',
    #                                                  in_perturbers=perturbers)
    #     if err != 0: raise Exception("OpenOrb Exception: error code = %d" % err)
    #     print("...done")

    #     ########
    #     # JPL horizons
    #     #######
    #     # by default horizons eph have one more value than oorb eph
    #     print("Generating ephimerides data using Horizons JPL...")
    #
    #     ################################################
    #     # Earth wrt to Sun
    #     ###############################################
    #     print("Earth with respect to Sun...")
    #     obj_sun = Horizons(id='399', location=sun_obs,
    #                        epochs={'start': start_time, 'stop': end_time,
    #                                'step': str(int_step_jpl) + int_step_jpl_unit})
    #
    #     # Get the vectors table from JPL horizons
    #     eph_sun = obj_sun.vectors()
    #     print("...done")
    #
    #     ################################################
    #     # Moon wrt to Sun
    #     ###############################################
    #     print("Moon with respect to Sun...")
    #     obj_moon = Horizons(id='301', location=sun_obs,
    #                        epochs={'start': start_time, 'stop': end_time,
    #                                'step': str(int_step_jpl) + int_step_jpl_unit})
    #
    #     # Get the vectors table from JPL horizons
    #     eph_moon = obj_moon.vectors()
    #     print("...done")
    #
    #
    #     #############################################
    #     # Generate 39 element data frame containing results of the integrations: elements from 0-38
    #     #############################################
    #     """
    #     Units are au, au/day, degrees
    #     "Object id", "Julian Date", "Distance", "Helio q", "Helio e", "Helio i", "Helio Omega ", "Helio omega",
    #     "Helio M", "Helio x", "Helio y", "Helio z", "Helio vx", "Helio vy", "Helio vz", "Geo x", "Geo y", "Geo z",
    #     "Geo vx", "Geo vy", "Geo vz", "Geo q", "Geo e", "Geo i", "Geo Omega", "Geo omega", "Geo M", "Earth x (Helio)",
    #     "Earth y (Helio)", "Earth z (Helio)", "Earth vx (Helio)", "Earth vy (Helio)", "Earth vz (Helio)",
    #     "Moon x (Helio)", "Moon y (Helio)", "Moon z (Helio)", "Moon vx (Helio)", "Moon vy (Helio)",
    #     "Moon vz (Helio)", "Synodic x", "Synodic y", "Synodic z", "Eclip Long"
    #     """
    #
    #     nsteps = 0
    #     if len(eph[0]) == len(eph_sun) - 1:
    #         nsteps = len(eph[0])
    #     elif len(eph[0]) == len(eph_sun) + 1:
    #         nsteps = len(eph_sun)
    #     elif len(eph[0]) == len(eph_sun):
    #         nsteps = len(eph[0])
    #     else:
    #         print("Horizons and OOrb integration step error")
    #
    #     data_temp = {}
    #     new_data = pd.DataFrame(data_temp)
    #
    #     # element 0 - object id
    #     new_data["Object id"] = [minimoon] * nsteps
    #
    #     # element 1 - julian date
    #     new_data["Julian Date"] = [Time(eph[0, i, 0], format='mjd').to_value('jd', 'long') for i in range(nsteps)]
    #
    #     # element 2 - distance
    #     new_data["Distance"] = [np.sqrt((eph[0, i, 24] - eph_sun[i]['x']) ** 2 + (eph[0, i, 25] - eph_sun[i]['y']) ** 2
    #                                     + (eph[0, i, 26] - eph_sun[i]['z'])**2) for i in range(nsteps)]
    #
    #     # for elements 3 to 8, state vector should be converted to cometary and keplarian orbital elements
    #     orbits = np.zeros([nsteps, 1, 12], dtype=np.double, order='F')
    #     sun_c = 11  # 11 for Sun, 3 for Earth
    #     print("Getting helio keplarian osculating elements...")
    #     for i in range(nsteps):
    #         # the original orbit is in cartesian:[id x y z vx vy vz type epoch timescale H G]
    #         orbits[i, :] = [i, eph[0, i, 24], eph[0, i, 25], eph[0, i, 26], eph[0, i, 27], eph[0, i, 28],
    #                        eph[0, i, 29], 1., eph[0, i, 0], 1., mm_parser.mm_data["x7"].iloc[2], 0.15]
    #
    #     # new orbit is in cometary: [id q e i Om om tp type epoch timescale H G]
    #     new_orbits_com, err = pyoorb.pyoorb.oorb_element_transformation(in_orbits=orbits,
    #                                                                               in_element_type=2, in_center=sun_c)
    #
    #     # new orbit is in keplarian: [id a e i Om om M type epoch timescale H G]
    #     new_orbits_kep, err = pyoorb.pyoorb.oorb_element_transformation(in_orbits=orbits,
    #                                                                               in_element_type=3, in_center=sun_c)
    #     print("...done")
    #
    #     # element 3 - Helio q
    #     new_data["Helio q"] = new_orbits_com[:, 1]
    #
    #     # element 4 - Helio e
    #     new_data["Helio e"] = new_orbits_com[:, 2]
    #
    #     # element 5 - Helio i
    #     new_data["Helio i"] = np.rad2deg(new_orbits_com[:, 3])
    #
    #     # element 6 - Helio Omega
    #     new_data["Helio Omega"] = np.rad2deg(new_orbits_com[:, 4])
    #
    #     # element 7 - Helio omega
    #     new_data["Helio omega"] = np.rad2deg(new_orbits_com[:, 5])
    #
    #     # element 8 - Helio M
    #     new_data["Helio M"] = np.rad2deg(new_orbits_kep[:, 6])
    #
    #     # element 9-14 - State vector
    #     new_data["Helio x"] = eph[0, :nsteps, 24]
    #     new_data["Helio y"] = eph[0, :nsteps, 25]
    #     new_data["Helio z"] = eph[0, :nsteps, 26]
    #     new_data["Helio vx"] = eph[0, :nsteps, 27]
    #     new_data["Helio vy"] = eph[0, :nsteps, 28]
    #     new_data["Helio vz"] = eph[0, :nsteps, 29]
    #
    #     # element 15-20 - Geocentric State vector
    #     new_data["Geo x"] = eph[0, :nsteps, 24] - eph_sun[:nsteps]['x']  # sun-mm pos vec - sun-earth pos vec
    #     new_data["Geo y"] = eph[0, :nsteps, 25] - eph_sun[:nsteps]['y']
    #     new_data["Geo z"] = eph[0, :nsteps, 26] - eph_sun[:nsteps]['z']
    #     new_data["Geo vx"] = eph[0, :nsteps, 27] - eph_sun[:nsteps]['vx']  # geo is non-rotating frame vrel = vmm - ve
    #     new_data["Geo vy"] = eph[0, :nsteps, 28] - eph_sun[:nsteps]['vy']
    #     new_data["Geo vz"] = eph[0, :nsteps, 29] - eph_sun[:nsteps]['vz']
    #
    #     # for elements 21 to 26, geo state vector should be converted to cometary and keplarian orbital elements
    #     orbits_geo = np.zeros([nsteps, 1, 12], dtype=np.double, order='F')
    #     earth_c = 3
    #     print("Getting geocentric osculating keplarian elements...")
    #     for i in range(nsteps):
    #         # the original orbit is in cartesian:[id x y z vx vy vz type epoch timescale H G]
    #         orbits_geo[i, :, :] = [i, new_data["Geo x"].iloc[i], new_data["Geo y"].iloc[i], new_data["Geo z"].iloc[i],
    #                             new_data["Geo vx"].iloc[i], new_data["Geo vy"].iloc[i], new_data["Geo vz"].iloc[i],
    #                             1., eph[0, i, 0], 1, mm_parser.mm_data["x7"].iloc[2], 0.15]
    #
    #     # new orbit is in cometary: [id q e i Om om tp type epoch timescale H G]
    #     new_orbits_com_geo, err = pyoorb.pyoorb.oorb_element_transformation(in_orbits=orbits_geo,
    #                                                                               in_element_type=2, in_center=earth_c)
    #     print("...Done")
    #
    #     # element 21 - Geo q
    #     new_data["Geo q"] = new_orbits_com_geo[:, 1]
    #
    #     # element 22 - Geo e
    #     new_data["Geo e"] = new_orbits_com_geo[:, 2]
    #
    #     # element 23 - Geo i
    #     new_data["Geo i"] = np.rad2deg(new_orbits_com_geo[:, 3])
    #
    #     # element 24 - Geo Omega
    #     new_data["Geo Omega"] = np.rad2deg(new_orbits_com_geo[:, 4])
    #
    #     # element 25 - Geo omega
    #     new_data["Geo omega"] = np.rad2deg(new_orbits_com_geo[:, 5])
    #
    #     # element 26 - Geo M
    #     print("Calculating mean anomaly...")
    #     temp = np.rad2deg(get_M(new_data, new_orbits_com_geo[:, 6], grav_param))
    #     new_data["Geo M"] = temp[0, :]
    #     print("...done")
    #
    #
    #     # element 27-32 - Heliocentric state vector of Earth
    #     new_data["Earth x (Helio)"] = eph_sun[:nsteps]['x']
    #     new_data["Earth y (Helio)"] = eph_sun[:nsteps]['y']
    #     new_data["Earth z (Helio)"] = eph_sun[:nsteps]['z']
    #     new_data["Earth vx (Helio)"] = eph_sun[:nsteps]['vx']
    #     new_data["Earth vy (Helio)"] = eph_sun[:nsteps]['vy']
    #     new_data["Earth vz (Helio)"] = eph_sun[:nsteps]['vz']
    #
    #     # element 33-38 - Heliocentric state vector of moon
    #     new_data["Moon x (Helio)"] = eph_moon[:nsteps]['x']
    #     new_data["Moon y (Helio)"] = eph_moon[:nsteps]['y']
    #     new_data["Moon z (Helio)"] = eph_moon[:nsteps]['z']
    #     new_data["Moon vx (Helio)"] = eph_moon[:nsteps]['vx']
    #     new_data["Moon vy (Helio)"] = eph_moon[:nsteps]['vy']
    #     new_data["Moon vz (Helio)"] = eph_moon[:nsteps]['vz']
    #
    #
    #     print("Getting synodic x,y,z, ecliptic longitude...")
    #     earth_xyz = np.array([new_data["Earth x (Helio)"], new_data["Earth y (Helio)"], new_data["Earth z (Helio)"]])
    #     mm_xyz = np.array([new_data["Geo x"], new_data["Geo y"], new_data["Geo z"]])
    #     trans_xyz = eci_ecliptic_to_sunearth_synodic(-earth_xyz, mm_xyz)  # minus is to have sun relative to earth
    #
    #     new_data["Synodic x"] = trans_xyz[0, :]
    #     new_data["Synodic y"] = trans_xyz[1, :]
    #     new_data["Synodic z"] = trans_xyz[2, :]
    #
    #     eclip_long = get_eclip_long(trans_xyz)
    #     new_data["Eclip Long"] = eclip_long[0, :]
    #
    #     print("...done")
    #
    #     # Encapsulate comparison graphs into function to compare with fedorets data
    #     #if (minimoon != '2006 RH120') and (minimoon != '2020 CD3'):
    #     #    self.compare(eph, data, new_data)
    #
    #     return new_data

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


if __name__ == '__main__':

    # Constants
    mu_e = const.GM_earth.value  # Nominal Earth mass parameter (m3/s2)

    # Amount before and after you want oorb integrations to start (in days) with respect to Fedorets data
    leadtime = 365 * cds.d

    # Get the data for the first minimoon file
    mm_master_file_name = 'NESCv9reintv1.TCO.withH.kep.des'  # name of the minimoon file to parsed

    # name of the directory where the mm file is located,
    # also top level directory where all the integration data is located
    mm_file_dir = os.path.join(os.path.expanduser('~'), 'OneDrive', 'Thesis', 'Minimoon_Integrations',
                               'minimoon_data')

    mm_file_path = mm_file_dir + '/' + mm_master_file_name  # path for the minimoon file
    source_path = os.path.join(os.path.expanduser('~'), 'Documents', 'sean', 'minimoon_integrations',
                                    'minimoon_files')

    destination_path = os.path.join(os.path.expanduser('~'), 'Documents', 'sean', 'minimoon_integrations',
                               'minimoon_files_oorb')

    # create parser
    mm_parser = MmParser(mm_file_path, mm_file_dir, source_path)

    # organize the data in the minimoon_file
    mm_parser.organize_data()

    # fetch all the files from all the folders within the top level directory
    # mm_parser.fetch_files()

    # create an analyzer
    mm_analyzer = MmAnalyzer()

    ####################################################################
    # Integrating data to generate new data from fedorets original data, generate new data file
    ####################################################################

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

    my_master_file = pd.DataFrame(columns=['Object id', 'H', 'D', 'Capture Date', 'Helio x at Capture',
                                           'Helio y at Capture', 'Helio z at Capture', 'Helio vx at Capture',
                                           'Helio vy at Capture','Helio vz at Capture', 'Helio q at Capture',
                                           'Helio e at Capture', 'Helio i at Capture', 'Helio Omega at Capture',
                                           'Helio omega at Capture', 'Helio M at Capture', 'Geo x at Capture',
                                           'Geo y at Capture', 'Geo z at Capture', 'Geo vx at Capture', 'Geo vy at Capture',
                                           'Geo vz at Capture', 'Geo q at Capture', 'Geo e at Capture',
                                           'Geo i at Capture', 'Geo Omega at Capture', 'Geo omega at Capture',
                                           'Geo M at Capture','Moon (Helio) x at Capture', 'Moon (Helio) y at Capture',
                                           'Moon (Helio) z at Capture', 'Moon (Helio) vx at Capture',
                                           'Moon (Helio) vy at Capture', 'Moon (Helio) vz at Capture',
                                           'Capture Duration', 'Spec. En. Duration', '3 Hill Duration', 'Number of Rev',
                                           '1 Hill Duration', 'Min. Distance', 'Release Date', 'Helio x at Release',
                                           'Helio y at Release', 'Helio z at Release', 'Helio vx at Release',
                                           'Helio vy at Release','Helio vz at Release', 'Helio q at Release',
                                           'Helio e at Release', 'Helio i at Release', 'Helio Omega at Release',
                                           'Helio omega at Release', 'Helio M at Release', 'Geo x at Release',
                                           'Geo y at Release', 'Geo z at Release', 'Geo vx at Release', 'Geo vy at Release',
                                           'Geo vz at Release', 'Geo q at Release', 'Geo e at Release',
                                           'Geo i at Release', 'Geo Omega at Release', 'Geo omega at Release',
                                           'Geo M at Release','Moon (Helio) x at Release', 'Moon (Helio) y at Release',
                                           'Moon (Helio) z at Release', 'Moon (Helio) vx at Release',
                                           'Moon (Helio) vy at Release', 'Moon (Helio) vz at Release'])

    int_step = 1 / 24
    errors = []
    # Loop over all the files
    for i in range(len(mm_parser.mm_data["File Name"])):
        i += 11
        print(i)
        mm_file_name = mm_parser.mm_data["File Name"].iloc[i] + ".dat"
        temp_file = source_path + '/' + mm_file_name
        header = None
        data = mm_parser.mm_file_parse(temp_file, header)

        minimoon = str(data["Object id"].iloc[0])
        # minimoon = '2020 CD3'

        if minimoon == '2006 RH120':
            # Integration range start and end dates - if you change dates here you have to change orbital elements
            start_time = '2006-01-01T00:00:00'
            end_time = '2008-02-01T00:00:00'
        elif minimoon == '2020 CD3':
            # Integration range start and end dates
            start_time = '2018-01-01T00:00:00'
            end_time = '2020-09-01T00:00:00'
        else:
            start_time = str(
                Time(data["Julian Date"].iloc[0] * cds.d - leadtime, format="jd", scale='utc').to_value('isot'))
            end_time = str(
                Time(data["Julian Date"].iloc[-1] * cds.d + leadtime, format="jd", scale='utc').to_value('isot'))

        steps = (Time(data["Julian Date"].iloc[-1] * cds.d + leadtime, format="jd", scale='utc').to_value('jd')
                 - Time(data["Julian Date"].iloc[0] * cds.d - leadtime, format="jd", scale='utc').to_value('jd'))/int_step
        print("Number of integration steps: " + str(steps))

        if steps < 90000:  # JPL Horizons limit
            new_data = mm_analyzer.get_data_mm_oorb_w_horizons(mm_parser, data, int_step, perturbers,
                                                           start_time, end_time, mu_e, minimoon)

            mm_analyzer.H = mm_parser.mm_data['x7'].iloc[i]
            new_data.to_csv(destination_path + '/' + minimoon + '.csv', sep=' ', header=True, index=False)

            ########################################################################################################
            # generate a new master file, which contains pertinant information about the capture for each minimoon
            #########################################################################################################

            # perform a minimoon_check analysis on it
            mm_analyzer.minimoon_check(new_data, mu_e)

            hx = new_data['Helio x'].iloc[mm_analyzer.cap_idx]
            hy = new_data['Helio y'].iloc[mm_analyzer.cap_idx]
            hz = new_data['Helio z'].iloc[mm_analyzer.cap_idx]
            hvx = new_data['Helio vx'].iloc[mm_analyzer.cap_idx]
            hvy = new_data['Helio vy'].iloc[mm_analyzer.cap_idx]
            hvz = new_data['Helio vz'].iloc[mm_analyzer.cap_idx]
            #caphelstavec = [hx, hy, hz, hvx, hvy, hvz]

            hq = new_data['Helio q'].iloc[mm_analyzer.cap_idx]
            he = new_data['Helio e'].iloc[mm_analyzer.cap_idx]
            hi = new_data['Helio i'].iloc[mm_analyzer.cap_idx]
            hOm = new_data['Helio Omega'].iloc[mm_analyzer.cap_idx]
            hom = new_data['Helio omega'].iloc[mm_analyzer.cap_idx]
            hM = new_data['Helio M'].iloc[mm_analyzer.cap_idx]
            #caphelkep = [hq, he, hi, hOm, hom, hM]

            gx = new_data['Geo x'].iloc[mm_analyzer.cap_idx]
            gy = new_data['Geo y'].iloc[mm_analyzer.cap_idx]
            gz = new_data['Geo z'].iloc[mm_analyzer.cap_idx]
            gvx = new_data['Geo vx'].iloc[mm_analyzer.cap_idx]
            gvy = new_data['Geo vy'].iloc[mm_analyzer.cap_idx]
            gvz = new_data['Geo vz'].iloc[mm_analyzer.cap_idx]
            #capgeostavec = [gx, gy, gz, gvx, gvy, gvz]

            gq = new_data['Geo q'].iloc[mm_analyzer.cap_idx]
            ge = new_data['Geo e'].iloc[mm_analyzer.cap_idx]
            gi = new_data['Geo i'].iloc[mm_analyzer.cap_idx]
            gOm = new_data['Geo Omega'].iloc[mm_analyzer.cap_idx]
            gom = new_data['Geo omega'].iloc[mm_analyzer.cap_idx]
            gM = new_data['Geo M'].iloc[mm_analyzer.cap_idx]
            #capgeokep = [gq, ge, gi, gOm, gom, gM]

            hmx = new_data['Moon x (Helio)'].iloc[mm_analyzer.cap_idx]
            hmy = new_data['Moon y (Helio)'].iloc[mm_analyzer.cap_idx]
            hmz = new_data['Moon z (Helio)'].iloc[mm_analyzer.cap_idx]
            hmvx = new_data['Moon vx (Helio)'].iloc[mm_analyzer.cap_idx]
            hmvy = new_data['Moon vy (Helio)'].iloc[mm_analyzer.cap_idx]
            hmvz = new_data['Moon vz (Helio)'].iloc[mm_analyzer.cap_idx]
            #capmoonstavec = [hmx, hmy, hmz, hmvx, hmvy, hmvz]

            rhx = new_data['Helio x'].iloc[mm_analyzer.rel_idx]
            rhy = new_data['Helio y'].iloc[mm_analyzer.rel_idx]
            rhz = new_data['Helio z'].iloc[mm_analyzer.rel_idx]
            rhvx = new_data['Helio vx'].iloc[mm_analyzer.rel_idx]
            rhvy = new_data['Helio vy'].iloc[mm_analyzer.rel_idx]
            rhvz = new_data['Helio vz'].iloc[mm_analyzer.rel_idx]
            #relhelstavec = [rhx, rhy, rhz, rhvx, rhvy, rhvz]

            rhq = new_data['Helio q'].iloc[mm_analyzer.rel_idx]
            rhe = new_data['Helio e'].iloc[mm_analyzer.rel_idx]
            rhi = new_data['Helio i'].iloc[mm_analyzer.rel_idx]
            rhOm = new_data['Helio Omega'].iloc[mm_analyzer.rel_idx]
            rhom = new_data['Helio omega'].iloc[mm_analyzer.rel_idx]
            rhM = new_data['Helio M'].iloc[mm_analyzer.rel_idx]
            #relhelkep = [rhq, rhe, rhi, rhOm, rhom, rhM]

            rgx = new_data['Geo x'].iloc[mm_analyzer.rel_idx]
            rgy = new_data['Geo y'].iloc[mm_analyzer.rel_idx]
            rgz = new_data['Geo z'].iloc[mm_analyzer.rel_idx]
            rgvx = new_data['Geo vx'].iloc[mm_analyzer.rel_idx]
            rgvy = new_data['Geo vy'].iloc[mm_analyzer.rel_idx]
            rgvz = new_data['Geo vz'].iloc[mm_analyzer.rel_idx]
            #relgeostavec = [rgx, rgy, rgz, rgvx, rgvy, rgvz]

            rgq = new_data['Geo q'].iloc[mm_analyzer.rel_idx]
            rge = new_data['Geo e'].iloc[mm_analyzer.rel_idx]
            rgi = new_data['Geo i'].iloc[mm_analyzer.rel_idx]
            rgOm = new_data['Geo Omega'].iloc[mm_analyzer.rel_idx]
            rgom = new_data['Geo omega'].iloc[mm_analyzer.rel_idx]
            rgM = new_data['Geo M'].iloc[mm_analyzer.rel_idx]
            #relgeokep = [rgq, rge, rgi, rgOm, rgom, rgM]

            rhmx = new_data['Moon x (Helio)'].iloc[mm_analyzer.rel_idx]
            rhmy = new_data['Moon y (Helio)'].iloc[mm_analyzer.rel_idx]
            rhmz = new_data['Moon z (Helio)'].iloc[mm_analyzer.rel_idx]
            rhmvx = new_data['Moon vx (Helio)'].iloc[mm_analyzer.rel_idx]
            rhmvy = new_data['Moon vy (Helio)'].iloc[mm_analyzer.rel_idx]
            rhmvz = new_data['Moon vz (Helio)'].iloc[mm_analyzer.rel_idx]
            #relmoonstavec = [rhmx, rhmy, rhmz, rhmvx, rhmvy, rhmvz]

            D = getH2D(mm_analyzer.H) * 1000

            new_row = pd.DataFrame({'Object id': new_data["Object id"].iloc[0], 'H': mm_analyzer.H, 'D': D,
                       'Capture Date': mm_analyzer.capture_start, 'Helio x at Capture': hx, 'Helio y at Capture': hy,
                       'Helio z at Capture': hz, 'Helio vx at Capture': hvx, 'Helio vy at Capture': hvy,
                       'Helio vz at Capture': hvz, 'Helio q at Capture': hq, 'Helio e at Capture': he,
                       'Helio i at Capture': hi, 'Helio Omega at Capture': hOm, 'Helio omega at Capture': hom,
                       'Helio M at Capture': hM, 'Geo x at Capture': gx, 'Geo y at Capture': gy, 'Geo z at Capture': gz,
                       'Geo vx at Capture': gvx, 'Geo vy at Capture': gvy, 'Geo vz at Capture': gvz,
                       'Geo q at Capture': gq, 'Geo e at Capture': ge, 'Geo i at Capture': gi,
                       'Geo Omega at Capture': gOm, 'Geo omega at Capture': gom, 'Geo M at Capture': gM,
                       'Moon (Helio) x at Capture': hmx, 'Moon (Helio) y at Capture': hmy,
                       'Moon (Helio) z at Capture': hmz, 'Moon (Helio) vx at Capture': hmvx,
                       'Moon (Helio) vy at Capture': hmvy, 'Moon (Helio) vz at Capture': hmvz,
                       'Capture Duration': mm_analyzer.capture_duration,
                       'Spec. En. Duration': mm_analyzer.epsilon_duration,
                       '3 Hill Duration': mm_analyzer.three_eh_duration, 'Number of Rev': mm_analyzer.revolutions,
                       '1 Hill Duration': mm_analyzer.one_eh_duration, 'Min. Distance': mm_analyzer.min_dist,
                       'Release Date': mm_analyzer.capture_end, 'Helio x at Release': rhx, 'Helio y at Release': rhy,
                       'Helio z at Release': rhz, 'Helio vx at Release': rhvx, 'Helio vy at Release': rhvy,
                       'Helio vz at Release': rhvz, 'Helio q at Release': rhq, 'Helio e at Release': rhe,
                       'Helio i at Release': rhi, 'Helio Omega at Release': rhOm, 'Helio omega at Release': rhom,
                       'Helio M at Release': rhM, 'Geo x at Release': rgx, 'Geo y at Release': rgy,
                       'Geo z at Release': rgz, 'Geo vx at Release': rgvx, 'Geo vy at Release': rgvy,
                       'Geo vz at Release': rgvz, 'Geo q at Release': rgq, 'Geo e at Release': rge,
                       'Geo i at Release': rgi, 'Geo Omega at Release': rgOm, 'Geo omega at Release': rgom,
                       'Geo M at Release': rgM, 'Moon (Helio) x at Release': rhmx, 'Moon (Helio) y at Release': rhmy,
                       'Moon (Helio) z at Release': rhmz, 'Moon (Helio) vx at Release': rhmvx,
                       'Moon (Helio) vy at Release': rhmvy, 'Moon (Helio) vz at Release': rhmvz}, index=[1])

            #my_master_file = pd.concat([my_master_file, new_row], ignore_index=True)
            new_row.to_csv(destination_path + '/' + 'minimoon_master.csv', sep=' ', mode='a', header=False, index=False)

        else:
            errors.append([i, str(data["Object id"].iloc[0])])
            print("Horizons error at:" + str(i) + " for minimoon " + str(data["Object id"].iloc[0]))

