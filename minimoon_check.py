import numpy as np
from astropy.time import Time
import astropy.units as u
from space_fncs import eci_ecliptic_to_sunearth_synodic

def minimoon_check(ephs, obs_state, grav_param, start_time, int_step):
    """
    this function is meant to check if and for what period of time the generated ephemerides (from oorb's pyoorb)
    for certain test particles
    satisfied the four conditions to become a minimoon:
    1) It remains within 3 Earth Hill Radii
        - Calculated as the distance from Earth, given the geocentric cartesian coordinates
    2) It has a negative planetocentric energy, that is, its specific energy wrt to the Earth-Moon barycenter
       (and also just Earth) is negative epsilon = v^2/2 - mu/r
    3) It completes an entire revolution around Earth - tracked through the geocentric ecliptic longitude in the
       geocentric rotating frame
    4) It has at least one approach to within 1 Earth Hill Radii

    :param ephs: takes the output full ephimerides generated from OpenOrb
            x_obs, y_obs, z_obs, vx_obs, vy_obs, vz_obs: Takes the x y z and vx vy vz ephemeris of the observer in a
            heliocentric coordinate frame for the given epoch
            grav_param: Is the gravitational constant of the body in question (m3/s2)
            start_time: Start time of the integrations (As a calendar date)
            int_step: fraction of day integration steps are divided into
    :return: several statistics...
    """

    # Important constants
    mu = grav_param
    aupd = u.AU / u.d  # AU per day
    mps = u.m / u.s  # Meters per second

    # Grab the geocentric cartesian coordinates of the test particle
    steps = len(ephs)  # Number of steps in the integration
    N = 1  # Number of test particles
    strt_tm = Time(start_time, format='isot', scale='utc')

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
    cum_angle = np.zeros((N, 1))   # The cumulative angle over the temporary capture

    # Variables to provide information of condition 4 (see function definition)
    satisfied_4 = np.zeros((N, steps))  # ==1 when condition one is satisfied ==0 otherwise
    satisfied_4_overall = np.zeros((N, 1))
    min_distances = np.zeros((N, 1))  # Minimum geocentric distance reached by minimoon
    one_eh = 0.01  # One Earth-Hill radius (in AU)
    time_satisfied_4 = np.zeros((N, 1))  # The number of days for which the condition 1 was satisfied

    # Variables describing overall captured
    captured = np.zeros((N, steps))  # When the asteroid is captured (i.e. condition 1 and 2 met)
    time_captured = np.zeros((N, 1))
    first_capture = 0
    capture_date = 0
    prev_capture = 0
    release_date = 0

    #Observer state vector
    x_obs = obs_state[0, :]
    y_obs = obs_state[1, :]
    z_obs = obs_state[2, :]
    vx_obs = obs_state[3, :]
    vy_obs = obs_state[4, :]
    vz_obs = obs_state[5, :]

    # Iterate over the particles

    distance = np.zeros((1, steps))
    epsilon_j = np.zeros((1, steps))
    v_rel_j = np.zeros((1, steps))

    for j in range(0, steps):
        # Body-centered x,y,z of minimoon from openorb data
        x = ephs[j, 24] - x_obs[j]
        y = ephs[j, 25] - y_obs[j]
        z = ephs[j, 26] - z_obs[j]

        d = np.sqrt(x**2 + y**2 + z**2)  # Distance of minimoon to observer
        distance[0, j] = d

        # First and fourth conditions
        if d < three_eh:
            satisfied_1[0, j] = 1
            time_satisfied_1[0, 0] += 1 * int_step

            if d < one_eh:
                satisfied_4[0, j] = 1
                satisfied_4_overall[0, 0] = 1
                time_satisfied_4[0, 0] += 1 * int_step

        # For velocity in observer-centered frame - in meters per second
        vx = ((ephs[j, 27] - vx_obs[j]) * aupd).to(mps) / mps
        vy = ((ephs[j, 28] - vy_obs[j]) * aupd).to(mps) / mps
        vz = ((ephs[j, 29] - vz_obs[j]) * aupd).to(mps) / mps

        v = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)  # Velocity of minimoon relative observer - meters per second
        v_rel_j[0, j] = v
        r = (d * u.AU).to(u.m) / u.m  # Distance of minimoon reative to observer - in meters
        epsilon = v**2/2 - mu / r  # Specific energy relative to observer in question
        epsilon_j[0, j] = epsilon

        # Check if condition 2 from the function definition is satisfied
        if epsilon < 0:
            satisfied_2[0, j] = 1
            time_satisfied_2[0, 0] += 1 * int_step

        # Identify beginning of capture
        if satisfied_1[0, j] == 1 and satisfied_2[0, j] == 1:
            prev_capture = 1
            # Store date of capture
            if first_capture == 0:
                capture_date = strt_tm + (j * int_step) * u.day
                first_capture = 1
            captured[0, j] = 1
            time_captured[0, 0] += 1 * int_step

        # Identify end of capture
        if satisfied_1[0, j] == 0 and satisfied_2[0, j] == 0 and prev_capture == 1:
            # Store date of end of capture
            prev_capture = 0
            release_date = strt_tm + (j * int_step) * u.day

        # Check to see how many revolutions were made during capture phase
        if captured[0, j] == 1 and j > 0:
            if captured[0, j-1] == 1:
                curr_angle = ephs[j, 11]  # Grab ecliptic longitude in the topocentric frame (== earth-centered
                # if observer is at center of Earth
                prev_angle = ephs[j-1, 11]
                diff_angle = curr_angle - prev_angle  # how much object moved
                if abs(diff_angle) > 180:
                    diff_angle = -360 + diff_angle
                cum_angle[0, 0] += diff_angle
            else:
                cum_angle[0, 0] = 0  # reset if previously uncaptured

    distances[0, :] = distance[0, :]
    vs_rel[0, :] = v_rel_j[0, :]
    epsilons[0, :] = epsilon_j[0, :]
    min_distances[0, 0] = min(distance[0, :])
    revolutions[0, 0] = abs(cum_angle[0, 0] / 360.0)
    if revolutions[0, 0] >= 1:
        satisfied_3[0, 0] = 1

    if time_captured[0, 0] > 0 and satisfied_3[0, 0] == 1 and satisfied_4_overall[0, 0] == 1:
        print("Object became minimoon: YES")
        print("Start of temporary capture: " + str(capture_date))
        print("End of temporary capture: " + str(release_date))
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

    print("Minimum distance reached to observer (AU): " + str(min_distances[0, 0]))
    print("\n")

    return N

def minimoon_check_jpl(minimoon_state, grav_param, start_time, int_step_unit):
    """
    this function is meant to check if and for what period of time the generated ephemerides (from JPL horizons)
    for known bodies wrt to a known body
    satisfied the four conditions to become a minimoon:
    1) It remains within 3 Earth Hill Radii
        - Calculated as the distance from Earth, given the geocentric cartesian coordinates
    2) It has a negative planetocentric energy, that is, its specific energy wrt to the Earth-Moon barycenter
       (and also just Earth) is negative epsilon = v^2/2 - mu/r
    3) It completes an entire revolution around Earth - tracked through the geocentric ecliptic longitude in the
       geocentric rotating frame
    4) It has at least one approach to within 1 Earth Hill Radii

    :param
            x, y, z, vx, vy, vz: Takes the x y z and vx vy vz ephemeris of the of the body wrt to the observer
            grav_param: Is the gravitational constant of the body in question (m3/s2)
            start_time: Start time of the integrations (As a calendar date)
            elciptic longitude wrt to observer (degrees)
    :return: several statistics...
    """

    # Important constants
    mu = grav_param
    aupd = u.AU / u.d  # AU per day
    mps = u.m / u.s  # Meters per second

    # Convert int step to days
    if int_step_unit == 'h':
        conv_day = 1/24
    elif int_step_unit == 'm':
        conv_day = 1/(24*60)
    else:
        conv_day = 1.

    # Grab the geocentric cartesian coordinates of the test particle
    steps = len(minimoon_state[0])  # Number of steps in the integration
    N = 1  # Number of test particles
    strt_tm = Time(start_time, format='isot', scale='utc')

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
    cum_angle = np.zeros((N, 1))   # The cumulative angle over the temporary capture

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
    prev_capture = 0
    release_date = 0

    # State vector components
    x = minimoon_state[0, :]
    y = minimoon_state[1, :]
    z = minimoon_state[2, :]
    vx = minimoon_state[3, :]
    vy = minimoon_state[4, :]
    vz = minimoon_state[5, :]
    ecl_lon = minimoon_state[6, :]

    distance = np.zeros((1, steps))
    epsilon_j = np.zeros((1, steps))
    v_rel_j = np.zeros((1, steps))

    for j in range(0, steps):

        d = np.sqrt(x[j]**2 + y[j]**2 + z[j]**2)  # Distance of minimoon to observer
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
                first_capture = 1
            captured[0, j] = 1
            time_captured[0, 0] += 1 * conv_day

        # Identify end of capture
        if satisfied_1[0, j] == 0 and satisfied_2[0, j] == 0 and prev_capture == 1:
            # Store date of end of capture
            prev_capture = 0
            release_date = strt_tm + (j * conv_day) * u.day

        # Check to see how many revolutions were made during capture phase
        if captured[0, j] == 1 and j > 0:
            if captured[0, j-1] == 1:
                curr_angle = ecl_lon[j]  # Grab ecliptic longitude in the topocentric frame (== earth-centered
                # if observer is at center of Earth
                prev_angle = ecl_lon[j-1]
                diff_angle = curr_angle - prev_angle  # how much object moved
                if abs(diff_angle) > 180:
                    diff_angle = -360 + diff_angle
                cum_angle[0, 0] += diff_angle
            else:
                cum_angle[0, 0] = 0  # reset if previously uncaptured

    distances[0, :] = distance[0, :]
    vs_rel[0, :] = v_rel_j[0, :]
    epsilons[0, :] = epsilon_j[0, :]
    min_distances[0, 0] = min(distance[0, :])
    revolutions[0, 0] = abs(cum_angle / 360.0)
    if revolutions[0, 0] >= 1:
        satisfied_3[0, 0] = 1

    satisfied_3[0, 0] = 1
    if time_captured > 0 and satisfied_3[0, 0] == 1 and satisfied_4_overall[0, 0] == 1:
        print("Object became minimoon: YES")
        print("Start of temporary capture: " + str(capture_date))
        print("End of temporary capture: " + str(release_date))
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

    print("Minimum distance reached to observer (AU): " + str(min_distances[0, 0]))
    print("\n")

    return
