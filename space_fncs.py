import numpy as np
import astropy as ap

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
