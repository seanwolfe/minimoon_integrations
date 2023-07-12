from astropy.time import Time
import pandas as pd
from space_fncs import get_eclip_long
import astropy.units as u
import numpy
from space_fncs import eci_ecliptic_to_sunearth_synodic
from space_fncs import get_M
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
from MmAnalyzer import MmAnalyzer
from scipy.signal import argrelextrema

cds.enable()
numpy.set_printoptions(threshold=sys.maxsize)


class MmMain():

    def __init__(self):

       return

    def integrate(self):

        # Constants
        mu_e = const.GM_earth.value  # Nominal Earth mass parameter (m3/s2)

        # Amount before and after you want oorb integrations to start (in days) with respect to Fedorets data
        leadtime = 365 * cds.d


        destination_path = os.path.join(os.getcwd(), 'Test_Set')
        destination_file = destination_path + '/minimoon_master_final.csv'

        # create parser
        mm_parser = MmParser(destination_file, "", "")

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

        my_master_file = pd.DataFrame(columns=['Object id', 'H', 'D', 'Capture Date',
                                               'Helio x at Capture', 'Helio y at Capture',
                                               'Helio z at Capture', 'Helio vx at Capture',
                                               'Helio vy at Capture', 'Helio vz at Capture',
                                               'Helio q at Capture', 'Helio e at Capture',
                                               'Helio i at Capture', 'Helio Omega at Capture',
                                               'Helio omega at Capture', 'Helio M at Capture',
                                               'Geo x at Capture', 'Geo y at Capture',
                                               'Geo z at Capture', 'Geo vx at Capture',
                                               'Geo vy at Capture', 'Geo vz at Capture',
                                               'Geo q at Capture', 'Geo e at Capture',
                                               'Geo i at Capture', 'Geo Omega at Capture',
                                               'Geo omega at Capture', 'Geo M at Capture',
                                               'Moon (Helio) x at Capture',
                                               'Moon (Helio) y at Capture',
                                               'Moon (Helio) z at Capture',
                                               'Moon (Helio) vx at Capture',
                                               'Moon (Helio) vy at Capture',
                                               'Moon (Helio) vz at Capture',
                                               'Capture Duration', 'Spec. En. Duration',
                                               '3 Hill Duration', 'Number of Rev',
                                               '1 Hill Duration', 'Min. Distance',
                                               'Release Date', 'Helio x at Release',
                                               'Helio y at Release', 'Helio z at Release',
                                               'Helio vx at Release', 'Helio vy at Release',
                                               'Helio vz at Release', 'Helio q at Release',
                                               'Helio e at Release', 'Helio i at Release',
                                               'Helio Omega at Release',
                                               'Helio omega at Release',
                                               'Helio M at Release', 'Geo x at Release',
                                               'Geo y at Release', 'Geo z at Release',
                                               'Geo vx at Release', 'Geo vy at Release',
                                               'Geo vz at Release', 'Geo q at Release',
                                               'Geo e at Release', 'Geo i at Release',
                                               'Geo Omega at Release',
                                               'Geo omega at Release', 'Geo M at Release',
                                               'Moon (Helio) x at Release',
                                               'Moon (Helio) y at Release',
                                               'Moon (Helio) z at Release',
                                               'Moon (Helio) vx at Release',
                                               'Moon (Helio) vy at Release',
                                               'Moon (Helio) vz at Release', 'Retrograde',
                                               'Became Minimoon', 'Max. Distance', 'Capture Index',
                                               'Release Index', 'X at Earth Hill', 'Y at Earth Hill',
                                               'Z at Earth Hill', 'Taxonomy', 'STC', "EMS Duration",
                                               "Periapsides in EMS", "Periapsides in 3 Hill",
                                               "Periapsides in 2 Hill", "Periapsides in 1 Hill",
                                               "STC Start", "STC Start Index", "STC End", "STC End Index",
                                               "Helio x at EMS", "Helio y at EMS", "Helio z at EMS",
                                               "Helio vx at EMS", "Helio vy at EMS", "Helio vz at EMS",
                                               "Earth x at EMS (Helio)", "Earth y at EMS (Helio)",
                                               "Earth z at EMS (Helio)", "Earth vx at EMS (Helio)",
                                               "Earth vy at EMS (Helio)", "Earth vz at EMS (Helio)",
                                               "Moon x at EMS (Helio)", "Moon y at EMS (Helio)",
                                               "Moon z at EMS (Helio)", "Moon vx at EMS (Helio)",
                                               "Moon vy at EMS (Helio)", "Moon vz at EMS (Helio)"])

        # error_dir = os.path.join(os.path.expanduser('~'), 'Documents', 'sean', 'minimoon_integrations')
        # error_file = 'errors.csv'
        # error_path = error_dir + '/' + error_file
        # my_errors = pd.read_csv(error_path, sep=",", header=None, names=["Index", "Object id"])
        # print(my_errors)

        int_step = 1 / 24
        # errors = []
        # Loop over all the files
        for i in range(1):

            # print(my_errors["Index"].iloc[i])
            # mm_file_name = mm_parser.mm_data["File Name"].iloc[my_errors["Index"].iloc[i]] + ".dat"
            # temp_file = source_path + '/' + mm_file_name
            # data = mm_parser.mm_file_parse(temp_file)
            data = []

            # minimoon = str(data["Object id"].iloc[0])
            minimoon = '2006 RH120'

            if minimoon == '2006 RH120':
                # Integration range start and end dates - if you change dates here you have to change orbital elements - is this true?
                start_time = '2005-01-01T00:00:00'
                end_time = '2009-02-01T00:00:00'
            elif minimoon == '2020 CD3':
                # Integration range start and end dates
                start_time = '2018-01-01T00:00:00'
                end_time = '2020-09-01T00:00:00'
            else:
                start_time = str(
                    Time(data["Julian Date"].iloc[0] * cds.d - leadtime, format="jd", scale='utc').to_value('isot'))
                end_time = str(
                    Time(data["Julian Date"].iloc[-1] * cds.d + leadtime, format="jd", scale='utc').to_value('isot'))

            # steps = (Time(data["Julian Date"].iloc[-1] * cds.d + leadtime, format="jd", scale='utc').to_value('jd')
            #          - Time(data["Julian Date"].iloc[0] * cds.d - leadtime, format="jd", scale='utc').to_value('jd'))/int_step
            # print("Number of integration steps: " + str(steps))
            steps = 0
            if steps < 90000:  # JPL Horizons limit

                if minimoon == '2006 RH120':
                    mm_analyzer.H = 29.9
                elif minimoon == '2020 CD3':
                    mm_analyzer.H = 31.9
                else:
                    mm_analyzer.H = mm_parser.mm_data['x7'].iloc[i]

                new_data = mm_analyzer.get_data_mm_oorb_w_horizons(mm_parser, data, int_step, perturbers,
                                                                   start_time, end_time, mu_e, minimoon)

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
                # caphelstavec = [hx, hy, hz, hvx, hvy, hvz]

                hq = new_data['Helio q'].iloc[mm_analyzer.cap_idx]
                he = new_data['Helio e'].iloc[mm_analyzer.cap_idx]
                hi = new_data['Helio i'].iloc[mm_analyzer.cap_idx]
                hOm = new_data['Helio Omega'].iloc[mm_analyzer.cap_idx]
                hom = new_data['Helio omega'].iloc[mm_analyzer.cap_idx]
                hM = new_data['Helio M'].iloc[mm_analyzer.cap_idx]
                # caphelkep = [hq, he, hi, hOm, hom, hM]

                gx = new_data['Geo x'].iloc[mm_analyzer.cap_idx]
                gy = new_data['Geo y'].iloc[mm_analyzer.cap_idx]
                gz = new_data['Geo z'].iloc[mm_analyzer.cap_idx]
                gvx = new_data['Geo vx'].iloc[mm_analyzer.cap_idx]
                gvy = new_data['Geo vy'].iloc[mm_analyzer.cap_idx]
                gvz = new_data['Geo vz'].iloc[mm_analyzer.cap_idx]
                # capgeostavec = [gx, gy, gz, gvx, gvy, gvz]

                gq = new_data['Geo q'].iloc[mm_analyzer.cap_idx]
                ge = new_data['Geo e'].iloc[mm_analyzer.cap_idx]
                gi = new_data['Geo i'].iloc[mm_analyzer.cap_idx]
                gOm = new_data['Geo Omega'].iloc[mm_analyzer.cap_idx]
                gom = new_data['Geo omega'].iloc[mm_analyzer.cap_idx]
                gM = new_data['Geo M'].iloc[mm_analyzer.cap_idx]
                # capgeokep = [gq, ge, gi, gOm, gom, gM]

                hmx = new_data['Moon x (Helio)'].iloc[mm_analyzer.cap_idx]
                hmy = new_data['Moon y (Helio)'].iloc[mm_analyzer.cap_idx]
                hmz = new_data['Moon z (Helio)'].iloc[mm_analyzer.cap_idx]
                hmvx = new_data['Moon vx (Helio)'].iloc[mm_analyzer.cap_idx]
                hmvy = new_data['Moon vy (Helio)'].iloc[mm_analyzer.cap_idx]
                hmvz = new_data['Moon vz (Helio)'].iloc[mm_analyzer.cap_idx]
                # capmoonstavec = [hmx, hmy, hmz, hmvx, hmvy, hmvz]

                rhx = new_data['Helio x'].iloc[mm_analyzer.rel_idx]
                rhy = new_data['Helio y'].iloc[mm_analyzer.rel_idx]
                rhz = new_data['Helio z'].iloc[mm_analyzer.rel_idx]
                rhvx = new_data['Helio vx'].iloc[mm_analyzer.rel_idx]
                rhvy = new_data['Helio vy'].iloc[mm_analyzer.rel_idx]
                rhvz = new_data['Helio vz'].iloc[mm_analyzer.rel_idx]
                # relhelstavec = [rhx, rhy, rhz, rhvx, rhvy, rhvz]

                rhq = new_data['Helio q'].iloc[mm_analyzer.rel_idx]
                rhe = new_data['Helio e'].iloc[mm_analyzer.rel_idx]
                rhi = new_data['Helio i'].iloc[mm_analyzer.rel_idx]
                rhOm = new_data['Helio Omega'].iloc[mm_analyzer.rel_idx]
                rhom = new_data['Helio omega'].iloc[mm_analyzer.rel_idx]
                rhM = new_data['Helio M'].iloc[mm_analyzer.rel_idx]
                # relhelkep = [rhq, rhe, rhi, rhOm, rhom, rhM]

                rgx = new_data['Geo x'].iloc[mm_analyzer.rel_idx]
                rgy = new_data['Geo y'].iloc[mm_analyzer.rel_idx]
                rgz = new_data['Geo z'].iloc[mm_analyzer.rel_idx]
                rgvx = new_data['Geo vx'].iloc[mm_analyzer.rel_idx]
                rgvy = new_data['Geo vy'].iloc[mm_analyzer.rel_idx]
                rgvz = new_data['Geo vz'].iloc[mm_analyzer.rel_idx]
                # relgeostavec = [rgx, rgy, rgz, rgvx, rgvy, rgvz]

                rgq = new_data['Geo q'].iloc[mm_analyzer.rel_idx]
                rge = new_data['Geo e'].iloc[mm_analyzer.rel_idx]
                rgi = new_data['Geo i'].iloc[mm_analyzer.rel_idx]
                rgOm = new_data['Geo Omega'].iloc[mm_analyzer.rel_idx]
                rgom = new_data['Geo omega'].iloc[mm_analyzer.rel_idx]
                rgM = new_data['Geo M'].iloc[mm_analyzer.rel_idx]
                # relgeokep = [rgq, rge, rgi, rgOm, rgom, rgM]

                rhmx = new_data['Moon x (Helio)'].iloc[mm_analyzer.rel_idx]
                rhmy = new_data['Moon y (Helio)'].iloc[mm_analyzer.rel_idx]
                rhmz = new_data['Moon z (Helio)'].iloc[mm_analyzer.rel_idx]
                rhmvx = new_data['Moon vx (Helio)'].iloc[mm_analyzer.rel_idx]
                rhmvy = new_data['Moon vy (Helio)'].iloc[mm_analyzer.rel_idx]
                rhmvz = new_data['Moon vz (Helio)'].iloc[mm_analyzer.rel_idx]
                # relmoonstavec = [rhmx, rhmy, rhmz, rhmvx, rhmvy, rhmvz]

                D = getH2D(mm_analyzer.H) * 1000

                repacked_results = mm_analyzer.short_term_capture(new_data["Object id"].iloc[0])

                # repack list according to index
                # repacked_results = [list(items) for items in zip(*results)]  # when running parallel processing

                # assign new columns to master file
                TCO_state = np.array(repacked_results[14])
                hxems = TCO_state[0]
                hyems = TCO_state[1]
                hzems = TCO_state[2]
                hvxems = TCO_state[3]
                hvyems = TCO_state[4]
                hvzems = TCO_state[5]

                Earth_state = np.array(repacked_results[15])
                hexems = Earth_state[0]
                heyems = Earth_state[1]
                hezems = Earth_state[2]
                hevxems = Earth_state[3]
                hevyems = Earth_state[4]
                hevzems = Earth_state[5]

                Moon_state = np.array(repacked_results[16])
                hmxems = Moon_state[0]
                hmyems = Moon_state[1]
                hmzems = Moon_state[2]
                hmvxems = Moon_state[3]
                hmvyems = Moon_state[4]
                hmvzems = Moon_state[5]

                # assign all the data for the minimoon in question into the master file
                new_row = pd.DataFrame({'Object id': new_data["Object id"].iloc[0], 'H': mm_analyzer.H, 'D': D,
                                        'Capture Date': mm_analyzer.capture_start, 'Helio x at Capture': hx,
                                        'Helio y at Capture': hy,
                                        'Helio z at Capture': hz, 'Helio vx at Capture': hvx,
                                        'Helio vy at Capture': hvy,
                                        'Helio vz at Capture': hvz, 'Helio q at Capture': hq, 'Helio e at Capture': he,
                                        'Helio i at Capture': hi, 'Helio Omega at Capture': hOm,
                                        'Helio omega at Capture': hom,
                                        'Helio M at Capture': hM, 'Geo x at Capture': gx, 'Geo y at Capture': gy,
                                        'Geo z at Capture': gz,
                                        'Geo vx at Capture': gvx, 'Geo vy at Capture': gvy, 'Geo vz at Capture': gvz,
                                        'Geo q at Capture': gq, 'Geo e at Capture': ge, 'Geo i at Capture': gi,
                                        'Geo Omega at Capture': gOm, 'Geo omega at Capture': gom,
                                        'Geo M at Capture': gM,
                                        'Moon (Helio) x at Capture': hmx, 'Moon (Helio) y at Capture': hmy,
                                        'Moon (Helio) z at Capture': hmz, 'Moon (Helio) vx at Capture': hmvx,
                                        'Moon (Helio) vy at Capture': hmvy, 'Moon (Helio) vz at Capture': hmvz,
                                        'Capture Duration': mm_analyzer.capture_duration,
                                        'Spec. En. Duration': mm_analyzer.epsilon_duration,
                                        '3 Hill Duration': mm_analyzer.three_eh_duration,
                                        'Number of Rev': mm_analyzer.revolutions,
                                        '1 Hill Duration': mm_analyzer.one_eh_duration,
                                        'Min. Distance': mm_analyzer.min_dist,
                                        'Release Date': mm_analyzer.capture_end, 'Helio x at Release': rhx,
                                        'Helio y at Release': rhy,
                                        'Helio z at Release': rhz, 'Helio vx at Release': rhvx,
                                        'Helio vy at Release': rhvy,
                                        'Helio vz at Release': rhvz, 'Helio q at Release': rhq,
                                        'Helio e at Release': rhe,
                                        'Helio i at Release': rhi, 'Helio Omega at Release': rhOm,
                                        'Helio omega at Release': rhom,
                                        'Helio M at Release': rhM, 'Geo x at Release': rgx, 'Geo y at Release': rgy,
                                        'Geo z at Release': rgz, 'Geo vx at Release': rgvx, 'Geo vy at Release': rgvy,
                                        'Geo vz at Release': rgvz, 'Geo q at Release': rgq, 'Geo e at Release': rge,
                                        'Geo i at Release': rgi, 'Geo Omega at Release': rgOm,
                                        'Geo omega at Release': rgom,
                                        'Geo M at Release': rgM, 'Moon (Helio) x at Release': rhmx,
                                        'Moon (Helio) y at Release': rhmy,
                                        'Moon (Helio) z at Release': rhmz, 'Moon (Helio) vx at Release': rhmvx,
                                        'Moon (Helio) vy at Release': rhmvy, 'Moon (Helio) vz at Release': rhmvz,
                                        'Retrograde': mm_analyzer.retrograde,
                                        'Became Minimoon': mm_analyzer.minimoon_flag,
                                        'Max. Distance': mm_analyzer.max_dist, 'Capture Index': mm_analyzer.cap_idx,
                                        'Release Index': mm_analyzer.rel_idx, 'X at Earth Hill': mm_analyzer.x_eh,
                                        'Y at Earth Hill': mm_analyzer.y_eh, 'Z at Earth Hill': mm_analyzer.z_eh,
                                        'Taxonomy': 'U', 'STC': mm_analyzer.stc, "EMS Duration": mm_analyzer.t_ems,
                                        "Periapsides in EMS": mm_analyzer.peri_ems,
                                        "Periapsides in 3 Hill": mm_analyzer.peri_3hill,
                                        "Periapsides in 2 Hill": mm_analyzer.peri_2hill,
                                        "Periapsides in 1 Hill": mm_analyzer.peri_1hill,
                                        "STC Start": mm_analyzer.stc_start,
                                        "STC Start Index": mm_analyzer.stc_start_idx,
                                        "STC End": mm_analyzer.stc_end, "STC End Index": mm_analyzer.stc_end_idx,
                                        "Helio x at EMS": hxems,
                                        "Helio y at EMS": hyems, "Helio z at EMS": hzems, "Helio vx at EMS": hvxems,
                                        "Helio vy at EMS": hvyems, "Helio vz at EMS": hvzems,
                                        "Earth x at EMS (Helio)": hexems,
                                        "Earth y at EMS (Helio)": heyems, "Earth z at EMS (Helio)": hezems,
                                        "Earth vx at EMS (Helio)": hevxems, "Earth vy at EMS (Helio)": hevyems,
                                        "Earth vz at EMS (Helio)": hevzems, "Moon x at EMS (Helio)": hmxems,
                                        "Moon y at EMS (Helio)": hmyems, "Moon z at EMS (Helio)": hmzems,
                                        "Moon vx at EMS (Helio)": hmvxems, "Moon vy at EMS (Helio)": hmvyems,
                                        "Moon vz at EMS (Helio)": hmvzems, "EMS Start": mm_analyzer.ems_start,
                                        "EMS Start Index": mm_analyzer.ems_start_idx, "EMS End": mm_analyzer.ems_end,
                                        "EMS End Index": mm_analyzer.ems_end_idx}, index=[1])

                # use the initial row to update the taxonomy
                new_row2 = pd.DataFrame({'Object id': new_data["Object id"].iloc[0], 'H': mm_analyzer.H, 'D': D,
                                         'Capture Date': mm_analyzer.capture_start, 'Helio x at Capture': hx,
                                         'Helio y at Capture': hy,
                                         'Helio z at Capture': hz, 'Helio vx at Capture': hvx,
                                         'Helio vy at Capture': hvy,
                                         'Helio vz at Capture': hvz, 'Helio q at Capture': hq, 'Helio e at Capture': he,
                                         'Helio i at Capture': hi, 'Helio Omega at Capture': hOm,
                                         'Helio omega at Capture': hom,
                                         'Helio M at Capture': hM, 'Geo x at Capture': gx, 'Geo y at Capture': gy,
                                         'Geo z at Capture': gz,
                                         'Geo vx at Capture': gvx, 'Geo vy at Capture': gvy, 'Geo vz at Capture': gvz,
                                         'Geo q at Capture': gq, 'Geo e at Capture': ge, 'Geo i at Capture': gi,
                                         'Geo Omega at Capture': gOm, 'Geo omega at Capture': gom,
                                         'Geo M at Capture': gM,
                                         'Moon (Helio) x at Capture': hmx, 'Moon (Helio) y at Capture': hmy,
                                         'Moon (Helio) z at Capture': hmz, 'Moon (Helio) vx at Capture': hmvx,
                                         'Moon (Helio) vy at Capture': hmvy, 'Moon (Helio) vz at Capture': hmvz,
                                         'Capture Duration': mm_analyzer.capture_duration,
                                         'Spec. En. Duration': mm_analyzer.epsilon_duration,
                                         '3 Hill Duration': mm_analyzer.three_eh_duration,
                                         'Number of Rev': mm_analyzer.revolutions,
                                         '1 Hill Duration': mm_analyzer.one_eh_duration,
                                         'Min. Distance': mm_analyzer.min_dist,
                                         'Release Date': mm_analyzer.capture_end, 'Helio x at Release': rhx,
                                         'Helio y at Release': rhy,
                                         'Helio z at Release': rhz, 'Helio vx at Release': rhvx,
                                         'Helio vy at Release': rhvy,
                                         'Helio vz at Release': rhvz, 'Helio q at Release': rhq,
                                         'Helio e at Release': rhe,
                                         'Helio i at Release': rhi, 'Helio Omega at Release': rhOm,
                                         'Helio omega at Release': rhom,
                                         'Helio M at Release': rhM, 'Geo x at Release': rgx, 'Geo y at Release': rgy,
                                         'Geo z at Release': rgz, 'Geo vx at Release': rgvx, 'Geo vy at Release': rgvy,
                                         'Geo vz at Release': rgvz, 'Geo q at Release': rgq, 'Geo e at Release': rge,
                                         'Geo i at Release': rgi, 'Geo Omega at Release': rgOm,
                                         'Geo omega at Release': rgom,
                                         'Geo M at Release': rgM, 'Moon (Helio) x at Release': rhmx,
                                         'Moon (Helio) y at Release': rhmy,
                                         'Moon (Helio) z at Release': rhmz, 'Moon (Helio) vx at Release': rhmvx,
                                         'Moon (Helio) vy at Release': rhmvy, 'Moon (Helio) vz at Release': rhmvz,
                                         'Retrograde': mm_analyzer.retrograde,
                                         'Became Minimoon': mm_analyzer.minimoon_flag,
                                         'Max. Distance': mm_analyzer.max_dist, 'Capture Index': mm_analyzer.cap_idx,
                                         'Release Index': mm_analyzer.rel_idx, 'X at Earth Hill': mm_analyzer.x_eh,
                                         'Y at Earth Hill': mm_analyzer.y_eh, 'Z at Earth Hill': mm_analyzer.z_eh,
                                         'Taxonomy': mm_analyzer.taxonomy(new_data, new_row), 'STC': mm_analyzer.stc,
                                         "EMS Duration": mm_analyzer.t_ems,
                                         "Periapsides in EMS": mm_analyzer.peri_ems,
                                         "Periapsides in 3 Hill": mm_analyzer.peri_3hill,
                                         "Periapsides in 2 Hill": mm_analyzer.peri_2hill,
                                         "Periapsides in 1 Hill": mm_analyzer.peri_1hill,
                                         "STC Start": mm_analyzer.stc_start,
                                         "STC Start Index": mm_analyzer.stc_start_idx,
                                         "STC End": mm_analyzer.stc_end, "STC End Index": mm_analyzer.stc_end_idx,
                                         "Helio x at EMS": hxems,
                                         "Helio y at EMS": hyems, "Helio z at EMS": hzems, "Helio vx at EMS": hvxems,
                                         "Helio vy at EMS": hvyems, "Helio vz at EMS": hvzems,
                                         "Earth x at EMS (Helio)": hexems,
                                         "Earth y at EMS (Helio)": heyems, "Earth z at EMS (Helio)": hezems,
                                         "Earth vx at EMS (Helio)": hevxems, "Earth vy at EMS (Helio)": hevyems,
                                         "Earth vz at EMS (Helio)": hevzems, "Moon x at EMS (Helio)": hmxems,
                                         "Moon y at EMS (Helio)": hmyems, "Moon z at EMS (Helio)": hmzems,
                                         "Moon vx at EMS (Helio)": hmvxems, "Moon vy at EMS (Helio)": hmvyems,
                                         "Moon vz at EMS (Helio)": hmvzems, "EMS Start": mm_analyzer.ems_start,
                                         "EMS Start Index": mm_analyzer.ems_start_idx, "EMS End": mm_analyzer.ems_end,
                                         "EMS End Index": mm_analyzer.ems_end_idx}, index=[1])

                # pd.set_option('display.max_columns', None)
                # pd.set_option('display.max_rows', None)
                # pd.set_option('display.float_format', lambda x: '%.5f' % x)
                new_row2.to_csv(destination_path + '/' + 'minimoon_master_final.csv', sep=' ', mode='a', header=False,
                                index=False)

            else:
                # errors.append([i, str(data["Object id"].iloc[0])])
                print("Horizons error at:" + str(i) + " for minimoon " + str(data["Object id"].iloc[0]))

        # print(errors)

        return

    def add_new_row(self):
        return

    def add_new_column(self):

# Constants
    mu_e = const.GM_earth.value  # Nominal Earth mass parameter (m3/s2)

    # Amount before and after you want oorb integrations to start (in days) with respect to Fedorets data
    leadtime = 365 * cds.d

    # Get the data for the first minimoon file
    # mm_master_file_name = 'Test_Set/NESCv9reintv1.TCO.withH.kep.des'  # name of the minimoon file to parsed

    # name of the directory where the mm file is located,
    # also top level directory where all the integration data is located
    # mm_file_dir = os.path.join(os.path.expanduser('~'), 'OneDrive', 'Thesis', 'Minimoon_Integrations',
    #                           'minimoon_data')
    # mm_file_dir = os.path.join(os.path.expanduser('~'), 'Documents', 'sean', 'minimoon_integrations')
    # mm_file_path = mm_file_dir + '/' + mm_master_file_name  # path for the minimoon file
    # source_path = os.path.join('/media', 'aeromec', 'a309bc3f-5615-4f84-b401-c01b43bd2be3',
    #                            'aeromec', 'minimoon_files')
    destination_path = os.path.join(os.getcwd(), 'Test_Set')
    destination_file = destination_path + '/minimoon_master_final.csv'

    # create parser
    mm_parser = MmParser(destination_file, "", "")

    # organize the data in the minimoon_file
    # mm_parser.organize_data()

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

    my_master_file = pd.DataFrame(columns=['Object id', 'H', 'D', 'Capture Date',
                                                             'Helio x at Capture', 'Helio y at Capture',
                                                             'Helio z at Capture', 'Helio vx at Capture',
                                                             'Helio vy at Capture', 'Helio vz at Capture',
                                                             'Helio q at Capture', 'Helio e at Capture',
                                                             'Helio i at Capture', 'Helio Omega at Capture',
                                                             'Helio omega at Capture', 'Helio M at Capture',
                                                             'Geo x at Capture', 'Geo y at Capture',
                                                             'Geo z at Capture', 'Geo vx at Capture',
                                                             'Geo vy at Capture', 'Geo vz at Capture',
                                                             'Geo q at Capture', 'Geo e at Capture',
                                                             'Geo i at Capture', 'Geo Omega at Capture',
                                                             'Geo omega at Capture', 'Geo M at Capture',
                                                             'Moon (Helio) x at Capture',
                                                             'Moon (Helio) y at Capture',
                                                             'Moon (Helio) z at Capture',
                                                             'Moon (Helio) vx at Capture',
                                                             'Moon (Helio) vy at Capture',
                                                             'Moon (Helio) vz at Capture',
                                                             'Capture Duration', 'Spec. En. Duration',
                                                             '3 Hill Duration', 'Number of Rev',
                                                             '1 Hill Duration', 'Min. Distance',
                                                             'Release Date', 'Helio x at Release',
                                                             'Helio y at Release', 'Helio z at Release',
                                                             'Helio vx at Release', 'Helio vy at Release',
                                                             'Helio vz at Release', 'Helio q at Release',
                                                             'Helio e at Release', 'Helio i at Release',
                                                             'Helio Omega at Release',
                                                             'Helio omega at Release',
                                                             'Helio M at Release', 'Geo x at Release',
                                                             'Geo y at Release', 'Geo z at Release',
                                                             'Geo vx at Release', 'Geo vy at Release',
                                                             'Geo vz at Release', 'Geo q at Release',
                                                             'Geo e at Release', 'Geo i at Release',
                                                             'Geo Omega at Release',
                                                             'Geo omega at Release', 'Geo M at Release',
                                                             'Moon (Helio) x at Release',
                                                             'Moon (Helio) y at Release',
                                                             'Moon (Helio) z at Release',
                                                             'Moon (Helio) vx at Release',
                                                             'Moon (Helio) vy at Release',
                                                             'Moon (Helio) vz at Release', 'Retrograde',
                                                             'Became Minimoon', 'Max. Distance', 'Capture Index',
                                                             'Release Index', 'X at Earth Hill', 'Y at Earth Hill',
                                                             'Z at Earth Hill', 'Taxonomy', 'STC', "EMS Duration",
                                                             "Periapsides in EMS", "Periapsides in 3 Hill",
                                                             "Periapsides in 2 Hill", "Periapsides in 1 Hill",
                                                             "STC Start", "STC Start Index", "STC End", "STC End Index",
                                                             "Helio x at EMS", "Helio y at EMS", "Helio z at EMS",
                                                             "Helio vx at EMS", "Helio vy at EMS", "Helio vz at EMS",
                                                             "Earth x at EMS (Helio)", "Earth y at EMS (Helio)",
                                                             "Earth z at EMS (Helio)", "Earth vx at EMS (Helio)",
                                                             "Earth vy at EMS (Helio)", "Earth vz at EMS (Helio)",
                                                             "Moon x at EMS (Helio)", "Moon y at EMS (Helio)",
                                                             "Moon z at EMS (Helio)", "Moon vx at EMS (Helio)",
                                                             "Moon vy at EMS (Helio)", "Moon vz at EMS (Helio)"])

    # error_dir = os.path.join(os.path.expanduser('~'), 'Documents', 'sean', 'minimoon_integrations')
    # error_file = 'errors.csv'
    # error_path = error_dir + '/' + error_file
    # my_errors = pd.read_csv(error_path, sep=",", header=None, names=["Index", "Object id"])
    # print(my_errors)

    int_step = 1 / 24
    # errors = []
    # Loop over all the files
    for i in range(1):

        # print(my_errors["Index"].iloc[i])
        # mm_file_name = mm_parser.mm_data["File Name"].iloc[my_errors["Index"].iloc[i]] + ".dat"
        # temp_file = source_path + '/' + mm_file_name
        # data = mm_parser.mm_file_parse(temp_file)
        data = []

        # minimoon = str(data["Object id"].iloc[0])
        minimoon = '2006 RH120'

        if minimoon == '2006 RH120':
            # Integration range start and end dates - if you change dates here you have to change orbital elements - is this true?
            start_time = '2005-01-01T00:00:00'
            end_time = '2009-02-01T00:00:00'
        elif minimoon == '2020 CD3':
            # Integration range start and end dates
            start_time = '2018-01-01T00:00:00'
            end_time = '2020-09-01T00:00:00'
        else:
            start_time = str(
                Time(data["Julian Date"].iloc[0] * cds.d - leadtime, format="jd", scale='utc').to_value('isot'))
            end_time = str(
                Time(data["Julian Date"].iloc[-1] * cds.d + leadtime, format="jd", scale='utc').to_value('isot'))

        # steps = (Time(data["Julian Date"].iloc[-1] * cds.d + leadtime, format="jd", scale='utc').to_value('jd')
        #          - Time(data["Julian Date"].iloc[0] * cds.d - leadtime, format="jd", scale='utc').to_value('jd'))/int_step
        # print("Number of integration steps: " + str(steps))
        steps = 0
        if steps < 90000:  # JPL Horizons limit

            if minimoon == '2006 RH120':
                mm_analyzer.H = 29.9
            elif minimoon == '2020 CD3':
                mm_analyzer.H = 31.9
            else:
                mm_analyzer.H = mm_parser.mm_data['x7'].iloc[i]

            new_data = mm_analyzer.get_data_mm_oorb_w_horizons(mm_parser, data, int_step, perturbers,
                                                           start_time, end_time, mu_e, minimoon)



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

            repacked_results = mm_analyzer.short_term_capture(new_data["Object id"].iloc[0])

            # repack list according to index
            # repacked_results = [list(items) for items in zip(*results)]  # when running parallel processing

            # assign new columns to master file
            TCO_state = np.array(repacked_results[14])
            hxems = TCO_state[0]
            hyems = TCO_state[1]
            hzems = TCO_state[2]
            hvxems = TCO_state[3]
            hvyems = TCO_state[4]
            hvzems = TCO_state[5]

            Earth_state = np.array(repacked_results[15])
            hexems = Earth_state[0]
            heyems = Earth_state[1]
            hezems = Earth_state[2]
            hevxems = Earth_state[3]
            hevyems = Earth_state[4]
            hevzems = Earth_state[5]

            Moon_state = np.array(repacked_results[16])
            hmxems = Moon_state[0]
            hmyems = Moon_state[1]
            hmzems = Moon_state[2]
            hmvxems = Moon_state[3]
            hmvyems = Moon_state[4]
            hmvzems = Moon_state[5]

            # assign all the data for the minimoon in question into the master file
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
                       'Moon (Helio) vy at Release': rhmvy, 'Moon (Helio) vz at Release': rhmvz,
                       'Retrograde': mm_analyzer.retrograde, 'Became Minimoon': mm_analyzer.minimoon_flag,
                       'Max. Distance': mm_analyzer.max_dist, 'Capture Index': mm_analyzer.cap_idx,
                       'Release Index': mm_analyzer.rel_idx, 'X at Earth Hill': mm_analyzer.x_eh,
                       'Y at Earth Hill': mm_analyzer.y_eh, 'Z at Earth Hill': mm_analyzer.z_eh,
                       'Taxonomy': 'U', 'STC': mm_analyzer.stc, "EMS Duration": mm_analyzer.t_ems,
                       "Periapsides in EMS": mm_analyzer.peri_ems, "Periapsides in 3 Hill": mm_analyzer.peri_3hill,
                       "Periapsides in 2 Hill": mm_analyzer.peri_2hill, "Periapsides in 1 Hill": mm_analyzer.peri_1hill,
                       "STC Start": mm_analyzer.stc_start, "STC Start Index": mm_analyzer.stc_start_idx,
                       "STC End": mm_analyzer.stc_end, "STC End Index": mm_analyzer.stc_end_idx, "Helio x at EMS": hxems,
                       "Helio y at EMS": hyems, "Helio z at EMS": hzems, "Helio vx at EMS": hvxems,
                       "Helio vy at EMS": hvyems, "Helio vz at EMS": hvzems, "Earth x at EMS (Helio)": hexems,
                       "Earth y at EMS (Helio)": heyems, "Earth z at EMS (Helio)": hezems,
                       "Earth vx at EMS (Helio)": hevxems, "Earth vy at EMS (Helio)": hevyems,
                       "Earth vz at EMS (Helio)": hevzems, "Moon x at EMS (Helio)": hmxems,
                       "Moon y at EMS (Helio)": hmyems, "Moon z at EMS (Helio)": hmzems,
                       "Moon vx at EMS (Helio)": hmvxems, "Moon vy at EMS (Helio)": hmvyems,
                       "Moon vz at EMS (Helio)": hmvzems, "EMS Start": mm_analyzer.ems_start,
                       "EMS Start Index": mm_analyzer.ems_start_idx, "EMS End": mm_analyzer.ems_end,
                       "EMS End Index": mm_analyzer.ems_end_idx}, index=[1])


            # use the initial row to update the taxonomy
            new_row2 = pd.DataFrame({'Object id': new_data["Object id"].iloc[0], 'H': mm_analyzer.H, 'D': D,
                                    'Capture Date': mm_analyzer.capture_start, 'Helio x at Capture': hx,
                                    'Helio y at Capture': hy,
                                    'Helio z at Capture': hz, 'Helio vx at Capture': hvx, 'Helio vy at Capture': hvy,
                                    'Helio vz at Capture': hvz, 'Helio q at Capture': hq, 'Helio e at Capture': he,
                                    'Helio i at Capture': hi, 'Helio Omega at Capture': hOm,
                                    'Helio omega at Capture': hom,
                                    'Helio M at Capture': hM, 'Geo x at Capture': gx, 'Geo y at Capture': gy,
                                    'Geo z at Capture': gz,
                                    'Geo vx at Capture': gvx, 'Geo vy at Capture': gvy, 'Geo vz at Capture': gvz,
                                    'Geo q at Capture': gq, 'Geo e at Capture': ge, 'Geo i at Capture': gi,
                                    'Geo Omega at Capture': gOm, 'Geo omega at Capture': gom, 'Geo M at Capture': gM,
                                    'Moon (Helio) x at Capture': hmx, 'Moon (Helio) y at Capture': hmy,
                                    'Moon (Helio) z at Capture': hmz, 'Moon (Helio) vx at Capture': hmvx,
                                    'Moon (Helio) vy at Capture': hmvy, 'Moon (Helio) vz at Capture': hmvz,
                                    'Capture Duration': mm_analyzer.capture_duration,
                                    'Spec. En. Duration': mm_analyzer.epsilon_duration,
                                    '3 Hill Duration': mm_analyzer.three_eh_duration,
                                    'Number of Rev': mm_analyzer.revolutions,
                                    '1 Hill Duration': mm_analyzer.one_eh_duration,
                                    'Min. Distance': mm_analyzer.min_dist,
                                    'Release Date': mm_analyzer.capture_end, 'Helio x at Release': rhx,
                                    'Helio y at Release': rhy,
                                    'Helio z at Release': rhz, 'Helio vx at Release': rhvx, 'Helio vy at Release': rhvy,
                                    'Helio vz at Release': rhvz, 'Helio q at Release': rhq, 'Helio e at Release': rhe,
                                    'Helio i at Release': rhi, 'Helio Omega at Release': rhOm,
                                    'Helio omega at Release': rhom,
                                    'Helio M at Release': rhM, 'Geo x at Release': rgx, 'Geo y at Release': rgy,
                                    'Geo z at Release': rgz, 'Geo vx at Release': rgvx, 'Geo vy at Release': rgvy,
                                    'Geo vz at Release': rgvz, 'Geo q at Release': rgq, 'Geo e at Release': rge,
                                    'Geo i at Release': rgi, 'Geo Omega at Release': rgOm, 'Geo omega at Release': rgom,
                                    'Geo M at Release': rgM, 'Moon (Helio) x at Release': rhmx,
                                    'Moon (Helio) y at Release': rhmy,
                                    'Moon (Helio) z at Release': rhmz, 'Moon (Helio) vx at Release': rhmvx,
                                    'Moon (Helio) vy at Release': rhmvy, 'Moon (Helio) vz at Release': rhmvz,
                                    'Retrograde': mm_analyzer.retrograde, 'Became Minimoon': mm_analyzer.minimoon_flag,
                                    'Max. Distance': mm_analyzer.max_dist, 'Capture Index': mm_analyzer.cap_idx,
                                    'Release Index': mm_analyzer.rel_idx, 'X at Earth Hill': mm_analyzer.x_eh,
                                    'Y at Earth Hill': mm_analyzer.y_eh, 'Z at Earth Hill': mm_analyzer.z_eh,
                                    'Taxonomy': mm_analyzer.taxonomy(new_data, new_row), 'STC': mm_analyzer.stc, "EMS Duration": mm_analyzer.t_ems,
                                    "Periapsides in EMS": mm_analyzer.peri_ems,
                                    "Periapsides in 3 Hill": mm_analyzer.peri_3hill,
                                    "Periapsides in 2 Hill": mm_analyzer.peri_2hill,
                                    "Periapsides in 1 Hill": mm_analyzer.peri_1hill,
                                    "STC Start": mm_analyzer.stc_start, "STC Start Index": mm_analyzer.stc_start_idx,
                                    "STC End": mm_analyzer.stc_end, "STC End Index": mm_analyzer.stc_end_idx,
                                    "Helio x at EMS": hxems,
                                    "Helio y at EMS": hyems, "Helio z at EMS": hzems, "Helio vx at EMS": hvxems,
                                    "Helio vy at EMS": hvyems, "Helio vz at EMS": hvzems,
                                    "Earth x at EMS (Helio)": hexems,
                                    "Earth y at EMS (Helio)": heyems, "Earth z at EMS (Helio)": hezems,
                                    "Earth vx at EMS (Helio)": hevxems, "Earth vy at EMS (Helio)": hevyems,
                                    "Earth vz at EMS (Helio)": hevzems, "Moon x at EMS (Helio)": hmxems,
                                    "Moon y at EMS (Helio)": hmyems, "Moon z at EMS (Helio)": hmzems,
                                    "Moon vx at EMS (Helio)": hmvxems, "Moon vy at EMS (Helio)": hmvyems,
                                    "Moon vz at EMS (Helio)": hmvzems, "EMS Start": mm_analyzer.ems_start,
                                    "EMS Start Index": mm_analyzer.ems_start_idx, "EMS End": mm_analyzer.ems_end,
                                    "EMS End Index": mm_analyzer.ems_end_idx}, index=[1])

            # pd.set_option('display.max_columns', None)
            # pd.set_option('display.max_rows', None)
            # pd.set_option('display.float_format', lambda x: '%.5f' % x)
            new_row2.to_csv(destination_path + '/' + 'minimoon_master_final.csv', sep=' ', mode='a', header=False, index=False)

        else:
            # errors.append([i, str(data["Object id"].iloc[0])])
            print("Horizons error at:" + str(i) + " for minimoon " + str(data["Object id"].iloc[0]))

    # print(errors)