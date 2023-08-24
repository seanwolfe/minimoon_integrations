import numpy as np
from astropy.time import Time
import astropy.units as u
from astropy.units import cds
from scipy.optimize import fsolve
from poliastro.twobody import Orbit
from poliastro.bodies import Sun, Earth, Moon
cds.enable()


def keplers_eq(E, M, e):
    return E - e * np.sin(E) - M

def get_theta_from_M(M, e):
    """
    converts mean anomaly to true anomaly, assuming elliptical orbit
    :param M: Mean anomaly
    :param e: eccentricity
    :return: theta (rad)
    """

    # Use fsolve to find the value of E
    E = fsolve(keplers_eq, x0=0, args=(M, e))
    theta = 2 * np.arctan(np.sqrt((1 + e)/(1 - e)) * np.tan(E[0]/2))

    return theta


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

def get_emb_synodic(data):
    # generate the x, y, z of the trajectory in the Sun-Earth/Moon synodic frame, centered at the earth-moon barycentre
    # calcualte the position of the earth-moon barycentr
    m_e = 5.97219e24  # mass of Earth
    m_m = 7.34767309e22  # mass of the Moon
    barycentre = (m_e * data[['Earth x (Helio)', 'Earth y (Helio)', 'Earth z (Helio)']].values + m_m * data[
        ['Moon x (Helio)', 'Moon y (Helio)', 'Moon z (Helio)']].values) \
                 / (m_m + m_e)

    # translate x, y, z to EMB
    emb_xyz = data[['Helio x', 'Helio y', 'Helio z']].values - barycentre

    # get synodic in emb frame
    emb_xyz_synodic = eci_ecliptic_to_sunearth_synodic(-barycentre.T, emb_xyz.T)

    return emb_xyz_synodic.T

def get_geo_v(dec, ra, ddec, dra, d, dd):

    geo_vx = dd * np.cos(np.deg2rad(ra)) * np.cos(np.deg2rad(dec)) - d * (np.deg2rad(dra) * np.sin(np.deg2rad(ra)) *
                np.cos(np.deg2rad(dec)) + np.deg2rad(ddec) * np.cos(np.deg2rad(ra)) * np.sin(np.deg2rad(dec)))
    geo_vy = dd * np.sin(np.deg2rad(ra)) * np.cos(np.deg2rad(dec)) + d * (
                np.deg2rad(dra) * np.cos(np.deg2rad(ra)) * np.cos(np.deg2rad(dec)) - np.deg2rad(ddec) *
                np.sin(np.deg2rad(ra)) * np.sin(np.deg2rad(dec)))
    geo_vz = dd * np.sin(np.deg2rad(dec)) + d * np.deg2rad(ddec) * np.cos(np.deg2rad(dec))

    return geo_vx, geo_vy, geo_vz

def get_r_and_v_cr3bp_from_nbody_sun_emb(h_r_TCO, h_v_TCO, h_r_E, h_v_E, h_r_M, h_v_M, date_mjd):

    # get the jacobi constant using ephimeris data
    seconds_in_day = 86400
    km_in_au = 149597870700 / 1000
    mu_s = 1.3271244e11 / np.power(km_in_au, 3)  # km^3/s^2 to AU^3/s^2
    mu_e = 3.986e5  # km^3/s^2
    mu_M = 4.9028e3  # km^3/s^2
    mu_EMS = (mu_M + mu_e) / np.power(km_in_au, 3)  # km^3/s^2 = m_E + mu_M to AU^3/s^2
    m_e = 5.97219e24  # mass of Earth kg
    m_m = 7.34767309e22  # mass of the Moon kg
    m_s = 1.9891e30  # mass of the Sun kg

    ############################################
    # Proposed method
    ###########################################

    ems_barycentre = (m_e * h_r_E + m_m * h_r_M) / (m_m + m_e)  # heliocentric position of the ems barycentre AU
    vems_barycentre = (m_e * h_v_E + m_m * h_v_M) / (m_m + m_e)  # heliocentric velocity of the ems barycentre AU/day

    r_C = (m_e + m_m) * ems_barycentre / (m_e + m_m + m_s)  # barycentre of sun-earth/moon AU
    r_sE = np.linalg.norm(ems_barycentre)  # distance between ems and sun AU
    omega = np.array([0, 0, np.sqrt((mu_s + mu_EMS) * seconds_in_day ** 2 / np.power(r_sE, 3))])  # angular velocity of sun-ems barycentre 1/day
    omega_2 = np.cross(ems_barycentre, vems_barycentre) / r_sE ** 2

    v_C = np.linalg.norm(r_C) / r_sE * vems_barycentre  # velocity of barycentre  AU/day

    r = [km_in_au * ems_barycentre[0], km_in_au * ems_barycentre[1], km_in_au * ems_barycentre[2]] << u.km  # km
    v = [km_in_au / seconds_in_day * vems_barycentre[0], km_in_au / seconds_in_day * vems_barycentre[1],
         km_in_au / seconds_in_day * vems_barycentre[2]] << u.km / u.s  # AU/day

    orb = Orbit.from_vectors(Sun, r, v, Time(date_mjd, format='mjd', scale='utc'))
    Om = np.deg2rad(orb.raan)  # in rad
    om = np.deg2rad(orb.argp)
    i = np.deg2rad(orb.inc)
    theta = np.deg2rad(orb.nu)

    # convert from heliocentric to synodic with a euler rotation
    h_R_C_peri = np.array([[-np.sin(Om) * np.cos(i) * np.sin(om) + np.cos(Om) * np.cos(om),
                            -np.sin(Om) * np.cos(i) * np.cos(om) - np.cos(Om) * np.sin(om),
                            np.sin(Om) * np.sin(i)],
                           [np.cos(Om) * np.cos(i) * np.sin(om) + np.sin(Om) * np.cos(om),
                            np.cos(Om) * np.cos(i) * np.cos(om) - np.sin(Om) * np.sin(om),
                            -np.cos(Om) * np.sin(i)],
                           [np.sin(i) * np.sin(om), np.sin(i) * np.cos(om), np.cos(i)]])

    # rotation from perihelion to location of ems_barycentre (i.e. rotation by true anomaly)
    C_peri_R_C = np.array([[np.cos(theta), -np.sin(theta), 0.],
                           [np.sin(theta), np.cos(theta), 0.],
                           [0., 0., 1.]])

    C_R_h = C_peri_R_C.T @ h_R_C_peri.T

    # translation
    C_T_h = np.array([np.linalg.norm(r_C), 0., 0.]).ravel()  # AU

    # in sun-earth/moon corotating
    C_r_TCO = C_R_h @ h_r_TCO - C_T_h  # AU
    C_ems = C_R_h @ ems_barycentre - C_T_h
    C_moon = C_R_h @ h_r_M - C_T_h

    C_v_TCO = C_R_h @ (h_v_TCO - v_C) - np.cross(omega, C_r_TCO)  # AU/day
    C_v_TCO_2 = C_R_h @ (h_v_TCO - v_C) - np.cross(C_R_h @ omega_2, C_r_TCO)  # AU/day

    ################################
    # Anderson and Lo
    ###############################

    # 1 - get state vectors (we have from proposed method)

    # 2 - Length unit
    LU = r_sE

    # 3 - Compute omega
    # omega_a = np.cross(ems_barycentre, vems_barycentre)
    # omega_a_n = omega_a / LU ** 2

    # 4 - Time and velocity unites
    # TU = 1 / np.linalg.norm(omega_a_n)
    # VU = LU / TU

    # 5 - First axis of rotated frame
    # e_1 = ems_barycentre / np.linalg.norm(ems_barycentre)

    # 6- third axis
    # e_3 = omega_a_n / np.linalg.norm(omega_a_n)

    # 7 - second axis
    # e_2 = np.cross(e_3, e_1)

    # 8 - rotation matrix
    # Q = np.array([e_1, e_2, e_3])

    # 9 - rotate postion vector
    # C_r_TCO_a = Q @ h_r_TCO

    # 10 - get velocity
    # C_v_TCO_a = Q @ (h_v_TCO - v_C) - Q @ np.cross(omega_a_n, h_r_TCO)

    # 11 - convert to nondimensional
    # C_r_TCO_a_n = C_r_TCO_a / LU
    # C_v_TCO_a_n = C_v_TCO_a / VU

    # C_r_TCO_a_n = C_r_TCO_a_n + np.array([mu, 0., 0.])

    return C_r_TCO, C_v_TCO, C_v_TCO_2, C_ems, C_moon, ems_barycentre, vems_barycentre, omega, omega_2, r_sE, mu_s, mu_EMS

def jacobi_dim_and_non_dim(C_r_TCO, C_v_TCO, h_r_TCO, ems_barycentre, mu, mu_s, mu_EMS, omega, r_sE):

    seconds_in_day = 86400
    km_in_au = 149597870700 / 1000

    v_rel = np.linalg.norm(C_v_TCO)  # AU/day

    # sun-TCO distance
    r_s = np.linalg.norm(h_r_TCO)  # AU

    # ems-TCO distance
    r_EMS = np.linalg.norm([h_r_TCO[0] - ems_barycentre[0], h_r_TCO[1] - ems_barycentre[1], h_r_TCO[2] - ems_barycentre[2]])  # AU

    constant = 0 #mu * (1 - mu) * (r_sE * km_in_au) ** 2 * np.linalg.norm(omega / seconds_in_day) ** 2

    # dimensional Jacobi constant km^2/s^2
    C_J_dimensional = ((np.linalg.norm(omega) / seconds_in_day) ** 2 * (C_r_TCO[0] ** 2 + C_r_TCO[1] ** 2) + 2 * mu_s / r_s + 2 * mu_EMS / r_EMS - (
            v_rel / seconds_in_day) ** 2) * np.power(km_in_au, 2) + constant  # might be missing a constant

    # non dimensional Jacobi constant
    x_prime = C_r_TCO[0] / r_sE
    y_prime = C_r_TCO[1] / r_sE
    z_prime = C_r_TCO[2] / r_sE

    x_dot_prime = C_v_TCO[0] / (np.linalg.norm(omega) * r_sE)
    y_dot_prime = C_v_TCO[1] / (np.linalg.norm(omega) * r_sE)
    z_dot_prime = C_v_TCO[2] / (np.linalg.norm(omega) * r_sE)
    v_prime = x_dot_prime ** 2 + y_dot_prime ** 2 + z_dot_prime ** 2
    r_s_prime = np.sqrt((x_prime + mu) ** 2 + y_prime ** 2 + z_prime ** 2)
    r_EMS_prime = np.sqrt((x_prime - (1 - mu)) ** 2 + y_prime ** 2 + z_prime ** 2)
    C_J_nondimensional = x_prime ** 2 + y_prime ** 2 + 2 * (
            1 - mu) / r_s_prime + 2 * mu / r_EMS_prime - v_prime  #+ mu * (1 - mu)

    return C_J_dimensional, C_J_nondimensional

def model(state, time, mu=0.01215):
    # Define the dynamics of the system
    # state: current state vector
    # time: current time
    # return: derivative of the state vector

    x, y, z, vx, vy, vz = state  # position and velocity

    dUdx = -(mu * (mu + x - 1))/np.power(((mu + x - 1)**2 + y**2 + z**2), (3/2)) - \
           ((1 - mu) * (mu + x))/np.power(((mu + x)**2 + y**2 + z**2),(3/2)) + x
    dUdy = - (mu * y)/np.power(((mu + x - 1)**2 + y**2 + z**2), (3/2)) - \
           ((1 - mu) * y)/np.power(((mu + x)**2 + y**2 + z**2), (3/2)) + y
    dUdz = - (mu * z)/np.power(((mu + x - 1)**2 + y**2 + z**2), (3/2)) - \
           ((1 - mu) * z)/np.power(((mu + x)**2 + y**2 + z**2), (3/2))

    dxdt = vx  # derivative of position is velocity
    dydt = vy
    dzdt = vz
    dvxdt = dUdx + 2*dydt  # derivative of velocity is acceleration
    dvydt = dUdy - 2*dxdt
    dvzdt = dUdz

    dXdt = np.array([dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt])

    return dXdt

def jacobi(res, mu):
    """

    :param res: the resultant state vector from a periodic orbit
    :param mu: the gravitational parameter
    :return: the jacobi constant
    """
    x, y, z, vx, vy, vz = res
    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2)
    U = 0.5 * (x ** 2 + y ** 2) + (1 - mu) / r1 + mu / r2

    return 2*U - vx**2 - vy**2 - vz**2
