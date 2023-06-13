import numpy as np
from astropy.time import Time
import astropy.units as u
from astropy.units import cds
cds.enable()


def get_M(new_data, tps, mu):
    """

    :param new_data: a dataframe generated as a middleman file during the integrations to rectify minimoon data from
    Fedorets
    :param tps: this is the time at perigee
    :param mu: the gravitational parameter
    :return:
    """
    mean_anomaly = np.zeros([1, len(new_data["Geo e"])])
    mu = (mu * u.m * u.m * u.m / u.s / u.s).to(u.km * u.km * u.km / u.s / u.s) / (u.km * u.km * u.km / u.s / u.s)

    for idx in range(len(new_data["Geo e"])):
        # Calculate Mean anamoly for elliptic
        tps_jd = Time(tps[idx], format='mjd', scale='utc').to_value(format='jd')
        e = new_data["Geo e"].iloc[idx]
        v = ([new_data["Geo vx"].iloc[idx], new_data["Geo vy"].iloc[idx],
              new_data["Geo vz"].iloc[idx]] * u.AU / cds.d).to(u.km / u.s) / (u.km / u.s)
        r = ([new_data["Geo x"].iloc[idx], new_data["Geo y"].iloc[idx], new_data["Geo z"].iloc[idx]] * u.AU).to(
            u.km) / u.km
        h = np.linalg.norm(np.cross(r, v))
        t = (new_data["Julian Date"].iloc[idx] - tps_jd) * 24 * 60 * 60  # convert from days to seconds
        if e < 1.:
            T = 2 * np.pi / (mu ** 2) * np.power(h / np.sqrt(1 - (e ** 2)), 3)
            if t < 0:
                mean_anomaly[0, idx] = 2*np.pi + 2 * np.pi / T * t
            else:
                mean_anomaly[0, idx] = 2 * np.pi / T * t
        # Calculate Mean anomoly for parabolic
        elif e == 1.:
            if t < 0:
                mean_anomaly[0, idx] = 2 * np.pi + mu ** 2 / np.power(h, 3) * t
            else:
                mean_anomaly[0, idx] = mu ** 2 / np.power(h, 3) * t
        # Calcluate mean anomaly for hyperbolic
        else:
            if t < 0:
                mean_anomaly[0, idx] = 2 * np.pi + mu ** 2 / np.power(h, 3) * np.power(e ** 2 - 1, 3/2) * t
            else:
                mean_anomaly[0, idx] = mu ** 2 / np.power(h, 3) * np.power(e ** 2 - 1, 3/2) * t

    return np.mod(mean_anomaly, 2*np.pi)

def getH2D(H):
    """
    convert absolute magnitude to diameter in meters, assuming a geometric albedo of 0.15
    :param H: absolute magnidte of object
    :return:D:  diameter of object in meters
    """
    a = 0.15
    D = 1347/np.sqrt(a) * np.power(10, -0.2*H)
    return D

def eci_ecliptic_to_sunearth_synodic(sun_eph, obj_xyz):
    """
    This function transforms coordinates from the ECI ecliptic plane to an Earth-centered Sun-Earth co-rotating frame,
    also referred to as a synodic frame
    :param sun_eph: the ephemeris x, y, z of the sun with respect to the ECI ecliptic frame 3 x n
    :param obj_xyz: the position of the object in the ECI ecliptic frame 3 x n
    :return: The transformed x, y, z coordinates 3 x n
    """

    trans_xyz = np.zeros((3, len(obj_xyz[0])))

    # Transform each coordinate
    for i in range(0, len(obj_xyz[0])):

        # Construct unit vector to Sun at that time
        u_s = - sun_eph[:, i] / np.linalg.norm(sun_eph[:, i])

        # Angle between x-axis in ECI and x-axis in synodic frame
        theta = np.arctan2(u_s[1],  u_s[0])

        # Rotation matrix - about z
        Rz_theta = np.array([[np.cos(theta), -np.sin(theta), 0.], [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])

        # Rotated point
        trans_xyz[:, i] = np.matmul(Rz_theta.T, obj_xyz[:, i])

    return trans_xyz


def get_eclip_long(synodic_xyz):
    """
    Function to get the ecliptic longitude with respect to earth of an object over a specified trajectory
    :param synodic_xyz: this is the x,y,z of the object with respect to Earth in an earth-sun rotating frame
    :return: returns the ecliptic longitude over time of the object in degrees
    """
    # Grab the ecliptic longitudes over the trajectory
    # For JPL Horizons
    eclip_long = np.zeros((1, len(synodic_xyz[0])))
    for j in range(0, len(synodic_xyz[0])):
        angle = np.rad2deg(np.arctan2(synodic_xyz[1, j], synodic_xyz[0, j]))
        if angle < 0:
            angle = 360 + angle
        eclip_long[0, j] = angle

    return eclip_long
