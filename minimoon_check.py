import astropy.units as u
import pyoorb
import numpy
import os
from astropy.time import Time
import matplotlib.pyplot as plt
from astroquery.jplhorizons import Horizons

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

    return days