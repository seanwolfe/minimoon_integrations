import numpy as np

def minimoon_check(ephs):
    """
    this function is meant to check if and for what period of time the generated ephemerides for certain test particles
    satisfied the four conditions to become a minimoon:
    1) It remains within 3 Earth Hill Radii
        - Calculated as the distance from Earth, given the geocentric cartesian coordinates
    2) It has a negative planetocentric energy !!(which equation)!!
    3) It completes an entire revolution around Earth !!(which frame)!!
    4) It has at least one approach to within 1 Earth Hill Radii

    :param ephs: takes the output full ephimerides generated from OpenOrb
    :return: several statistics...
    """

    # Grab the geocentric cartesian coordinates of the test particle
    days = len(ephs[0])  # Number of days in the integration
    N = len(ephs)  # Number of test particles

    # Variables to provide information of condition 1 (see function definition)
    satisfied_1 = np.zeros((N, days))  # ==1 when condition one is satisfied ==0 otherwise
    three_eh = 0.03  # Three Earth-Hill radius (in AU)
    distances = np.zeros((N, days))  # The distance to earth (in AU) for all particles at all times
    time_satisfied_1 = 0  # The number of days for which the condition 1 was satisfied

    # Variables to provide information of condition 4 (see function definition)
    satisfied_4 = np.zeros((N, days))  # ==1 when condition one is satisfied ==0 otherwise
    min_distances = np.zeros((N, 1))  # Minimum geocentric distance reached by minimoon
    one_eh = 0.01  # One Earth-Hill radius (in AU)
    time_satisfied_4 = 0  # The number of days for which the condition 1 was satisfied

    # Iterate over the particles
    for i in range(0, N):
        distance = np.zeros((1, days))
        for j in range(0, days):
            # Geocentric x,y,z of from openorb data
            x = ephs[i][j, 24] - ephs[i][j, 30]
            y = ephs[i][j, 25] - ephs[i][j, 31]
            z = ephs[i][j, 26] - ephs[i][j, 32]

            d = np.sqrt(x**2 + y**2 + z**2)  # Distance to Earth (AU)
            distance[0, j] = d

            if d < three_eh:
                satisfied_1[i, j] = 1
                time_satisfied_1 += 1

                if d < one_eh:
                    satisfied_4[i, j] = 1
                    time_satisfied_4 += 1

        distances[i, :] = distance[0, :]

        min_distances[i, 0] = min(distance[0, :])

    print(time_satisfied_1)
    print(time_satisfied_4)
    print(min_distances)
    print(satisfied_4)

    return N
