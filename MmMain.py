import pandas as pd
import numpy
from space_fncs import getH2D
import numpy as np
import os
from astropy.time import Time
import sys
from astropy.units import cds
from MM_Parser import MmParser
from MmAnalyzer import MmAnalyzer
import multiprocessing
from MmPopulation import MmPopulation
import matplotlib.pyplot as plt
from space_fncs import eci_ecliptic_to_sunearth_synodic
from astropy import constants as const
import astropy.units as u
from poliastro.twobody import Orbit
from poliastro.bodies import Sun, Earth, Moon
from space_fncs import get_theta_from_M
# import pyoorb
import matplotlib.ticker as ticker
from space_fncs import get_emb_synodic
from scipy.signal import argrelextrema
from space_fncs import get_r_and_v_cr3bp_from_nbody_sun_emb
from space_fncs import jacobi_dim_and_non_dim
from space_fncs import helio_to_earthmoon_corotating
from space_fncs import helio_to_earthmoon_corotating_vec
from space_fncs import pseudo_potential
from space_fncs import jacobi_earth_moon
import string
from itertools import product

cds.enable()
numpy.set_printoptions(threshold=sys.maxsize)


class MmMain():

    def __init__(self):

        return

    def integrate(self, master_path, mu_e, leadtime, perturbers, int_step):
        """ untested, to be used in conjunction with reintegrated data, not original fedorets data"""

        # create parser
        mm_parser = MmParser("", "", "")

        # create an analyzer
        mm_analyzer = MmAnalyzer()

        # get the master file - you need a list of initial orbits to integrate with openorb (pyorb)
        master = mm_parser.parse_master(master_path)

        leadtime = leadtime * cds.d

        ####################################################################
        # Integrating data to generate new data from fedorets original data, generate new data file
        ####################################################################

        # catch objects that should be integrated with larger step because number of samples is too large for JPL
        errors = []

        # Loop over all the files
        for i in range(len(master['Object id'])):

            data = mm_parser.mm_file_parse_new(os.path.join(os.getcwd(), 'minimoon_files_oorb') + '/' +
                                               str(master['Object id'].iloc[i]))
            minimoon = str(master["Object id"].iloc[0])
            start_time = str(Time(data["Julian Date"].iloc[0] * cds.d - leadtime,
                                  format="jd", scale='utc').to_value('isot'))
            end_time = str(Time(data["Julian Date"].iloc[-1] * cds.d + leadtime,
                                format="jd", scale='utc').to_value('isot'))
            steps = (Time(data["Julian Date"].iloc[-1] * cds.d + leadtime, format="jd", scale='utc').to_value('jd')
                     - Time(data["Julian Date"].iloc[0] * cds.d - leadtime, format="jd", scale='utc').to_value(
                        'jd')) / int_step

            print("Number of integration steps: " + str(steps))
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

                # add a new row of data to the master file
                mm_main.add_new_row(new_data)

            else:
                errors.append([i, str(data["Object id"].iloc[0])])
                print("Horizons error at:" + str(i) + " for minimoon " + str(data["Object id"].iloc[0]))

        return errors

    def integrate_parallel(self, idx, object_id):

        # print(idx)
        print(object_id)
        # create an analyzer
        mm_analyzer = MmAnalyzer()

        # Constants
        mu_e = const.GM_earth.value  # Nominal Earth mass parameter (m3/s2)

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
        moon = 0
        perturbers = [mercury, venus, earth, mars, jupiter, saturn, uranus, neptune, pluto, moon]

        # go through all the files of test particles
        population_dir = os.path.join(os.getcwd(), 'minimoon_files_oorb')
        name = "minimoon_master_final.csv"
        file_path = os.path.join(population_dir, name)
        mm_parser = MmParser("", "", "")

        # get master
        master = mm_parser.parse_master(file_path)
        # get old data
        old_name = str(object_id) + '.csv'
        file_path_old_data = os.path.join(population_dir, old_name)
        old_data = mm_parser.mm_file_parse_new(file_path_old_data)

        ####################################################################
        # Integrating data to generate new data from reintegrated data, generate new data file - check perturbers
        ####################################################################

        start_date = old_data['Julian Date'].iloc[0]
        end_date = old_data['Julian Date'].iloc[-1]
        int_step_rev = round(1 / (old_data['Julian Date'].iloc[1] - old_data['Julian Date'].iloc[0]))
        print(int_step_rev)
        int_step = 1 / int_step_rev
        start_time = Time(start_date, format="jd", scale='utc')
        end_time = Time(end_date, format="jd", scale='utc')
        steps = int((end_time.to_value('jd') - start_time.to_value('jd')) / int_step)
        master_i = master[master['Object id'] == object_id]
        new_data = mm_analyzer.get_data_mm_oorb(master_i, old_data, int_step, perturbers, start_time, end_time, mu_e)

        print("Number of integration steps: " + str(steps))

        destination_path = os.path.join(os.getcwd(), 'no_moon_files')
        # new_data.to_csv(destination_path + '/' + str(object_id) + '.csv', sep=' ', header=True, index=False)

        # H = master_i['H'].iloc[0]

        # add a new row of data to the master file
        # mm_main.add_new_row(new_data, mu_e, H, destination_path)

        return

    def add_new_row(self, new_data, mu_e, H, destination_path):

        ########################################################################################################
        # generate a new master file, which contains pertinant information about the capture for each minimoon
        #########################################################################################################

        # create an analyzer
        mm_analyzer = MmAnalyzer()

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

        D = getH2D(H) * 1000

        repacked_results = mm_analyzer.short_term_capture(new_data["Object id"].iloc[0])

        # repack list according to index
        # repacked_results = [list(items) for items in zip(*results)]  # when running parallel processing just this function

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
        new_row = pd.DataFrame({'Object id': new_data["Object id"].iloc[0], 'H': H, 'D': D,
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
        new_row2 = pd.DataFrame({'Object id': new_data["Object id"].iloc[0], 'H': H, 'D': D,
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

        new_row2.to_csv(destination_path + '/' + 'minimoon_master_final.csv', sep=' ', mode='a', header=False,
                        index=False)

        return new_row2

    def change_existing_column(self, master_path, population_dir, mu_e):

        # create parser
        mm_parser = MmParser(master_path, "", "")

        # create an analyzer
        mm_analyzer = MmAnalyzer()

        # get the master file - you need a list of initial orbits to integrate with openorb (pyorb)
        master = mm_parser.parse_master(master_path)

        new_column = np.zeros((len(master['Object id']),))

        # if adding data from short-term capture, uncomment these lines, and grab the data you want to add:
        ################################################# - here
        # parrelelized version of short term capture
        # pool = multiprocessing.Pool()
        # results = pool.map(mm_analyzer.short_term_capture, master['Object id'])
        # pool.close()

        # repack list according to index
        # repacked_results = [list(items) for items in zip(*results)]  # when running parallel processing

        # stc =  repacked_results[0]
        # tems = repacked_results[1]  # time in ems
        # pems = repacked_results[2]  # periapsides in ems
        # pth = repacked_results[3]  # periapsides in 3 hill
        # ptwoh = repacked_results[4]  # periapsides in 2 hill
        # poh = repacked_results[5]  # periapsides in one hill
        # sstart = repacked_results[6]  # start of stc
        # starti = repacked_results[7]  # start index of stc
        # send = repacked_results[8]  # end of stc
        # sendi = repacked_results[9]  # end index of stc
        # estart = repacked_results[10]  # start of ems stay
        # estarti = repacked_results[11]  # ems start index
        # eend = repacked_results[12]  # end ems
        # eendi = repacked_results[13]  # end ems index

        # assign new columns to master file
        # TCO_state = np.array(repacked_results[14])
        # hxems = TCO_state[0]
        # hyems = TCO_state[1]
        # hzems = TCO_state[2]
        # hvxems = TCO_state[3]
        # hvyems = TCO_state[4]
        # hvzems = TCO_state[5]

        # Earth_state = np.array(repacked_results[15])
        # hexems = Earth_state[0]
        # heyems = Earth_state[1]
        # hezems = Earth_state[2]
        # hevxems = Earth_state[3]
        # hevyems = Earth_state[4]
        # hevzems = Earth_state[5]

        # Moon_state = np.array(repacked_results[16])
        # hmxems = Moon_state[0]
        # hmyems = Moon_state[1]
        # hmzems = Moon_state[2]
        # hmvxems = Moon_state[3]
        # hmvyems = Moon_state[4]
        # hmvzems = Moon_state[5]

        # set the new column
        # new_column =
        ############################################### - to here

        # if adding data from minimoon_check, uncomment the following lines and grab data you want
        ####################################################### from here

        # for idx in range(len(master['Object id'])):

        # go through all the files of test particles
        # for root, dirs, files in os.walk(population_dir):
        #     find files that are minimoons
        # name = str(master['Object id'].iloc[idx]) + ".csv"
        #
        # if name in files:
        #     file_path = os.path.join(root, name)

        # read the file
        # data = mm_parser.mm_file_parse_new(file_path)

        # perform a minimoon_check analysis on it
        # mm_analyzer.minimoon_check(data, mu_e)
        #
        # hx = data['Helio x'].iloc[mm_analyzer.cap_idx]
        # hy = data['Helio y'].iloc[mm_analyzer.cap_idx]
        # hz = data['Helio z'].iloc[mm_analyzer.cap_idx]
        # hvx = data['Helio vx'].iloc[mm_analyzer.cap_idx]
        # hvy = data['Helio vy'].iloc[mm_analyzer.cap_idx]
        # hvz = data['Helio vz'].iloc[mm_analyzer.cap_idx]
        #
        # hq = data['Helio q'].iloc[mm_analyzer.cap_idx]
        # he = data['Helio e'].iloc[mm_analyzer.cap_idx]
        # hi = data['Helio i'].iloc[mm_analyzer.cap_idx]
        # hOm = data['Helio Omega'].iloc[mm_analyzer.cap_idx]
        # hom = data['Helio omega'].iloc[mm_analyzer.cap_idx]
        # hM = data['Helio M'].iloc[mm_analyzer.cap_idx]
        #
        # gx = data['Geo x'].iloc[mm_analyzer.cap_idx]
        # gy = data['Geo y'].iloc[mm_analyzer.cap_idx]
        # gz = data['Geo z'].iloc[mm_analyzer.cap_idx]
        # gvx = data['Geo vx'].iloc[mm_analyzer.cap_idx]
        # gvy = data['Geo vy'].iloc[mm_analyzer.cap_idx]
        # gvz = data['Geo vz'].iloc[mm_analyzer.cap_idx]
        #
        # gq = data['Geo q'].iloc[mm_analyzer.cap_idx]
        # ge = data['Geo e'].iloc[mm_analyzer.cap_idx]
        # gi = data['Geo i'].iloc[mm_analyzer.cap_idx]
        # gOm = data['Geo Omega'].iloc[mm_analyzer.cap_idx]
        # gom = data['Geo omega'].iloc[mm_analyzer.cap_idx]
        # gM = data['Geo M'].iloc[mm_analyzer.cap_idx]
        #
        # hmx = data['Moon x (Helio)'].iloc[mm_analyzer.cap_idx]
        # hmy = data['Moon y (Helio)'].iloc[mm_analyzer.cap_idx]
        # hmz = data['Moon z (Helio)'].iloc[mm_analyzer.cap_idx]
        # hmvx = data['Moon vx (Helio)'].iloc[mm_analyzer.cap_idx]
        # hmvy = data['Moon vy (Helio)'].iloc[mm_analyzer.cap_idx]
        # hmvz = data['Moon vz (Helio)'].iloc[mm_analyzer.cap_idx]
        #
        # rhx = data['Helio x'].iloc[mm_analyzer.rel_idx]
        # rhy = data['Helio y'].iloc[mm_analyzer.rel_idx]
        # rhz = data['Helio z'].iloc[mm_analyzer.rel_idx]
        # rhvx = data['Helio vx'].iloc[mm_analyzer.rel_idx]
        # rhvy = data['Helio vy'].iloc[mm_analyzer.rel_idx]
        # rhvz = data['Helio vz'].iloc[mm_analyzer.rel_idx]
        #
        # rhq = data['Helio q'].iloc[mm_analyzer.rel_idx]
        # rhe = data['Helio e'].iloc[mm_analyzer.rel_idx]
        # rhi = data['Helio i'].iloc[mm_analyzer.rel_idx]
        # rhOm = data['Helio Omega'].iloc[mm_analyzer.rel_idx]
        # rhom = data['Helio omega'].iloc[mm_analyzer.rel_idx]
        # rhM = data['Helio M'].iloc[mm_analyzer.rel_idx]
        #
        # rgx = data['Geo x'].iloc[mm_analyzer.rel_idx]
        # rgy = data['Geo y'].iloc[mm_analyzer.rel_idx]
        # rgz = data['Geo z'].iloc[mm_analyzer.rel_idx]
        # rgvx = data['Geo vx'].iloc[mm_analyzer.rel_idx]
        # rgvy = data['Geo vy'].iloc[mm_analyzer.rel_idx]
        # rgvz = data['Geo vz'].iloc[mm_analyzer.rel_idx]
        #
        # rgq = data['Geo q'].iloc[mm_analyzer.rel_idx]
        # rge = data['Geo e'].iloc[mm_analyzer.rel_idx]
        # rgi = data['Geo i'].iloc[mm_analyzer.rel_idx]
        # rgOm = data['Geo Omega'].iloc[mm_analyzer.rel_idx]
        # rgom = data['Geo omega'].iloc[mm_analyzer.rel_idx]
        # rgM = data['Geo M'].iloc[mm_analyzer.rel_idx]
        #
        # rhmx = data['Moon x (Helio)'].iloc[mm_analyzer.rel_idx]
        # rhmy = data['Moon y (Helio)'].iloc[mm_analyzer.rel_idx]
        # rhmz = data['Moon z (Helio)'].iloc[mm_analyzer.rel_idx]
        # rhmvx = data['Moon vx (Helio)'].iloc[mm_analyzer.rel_idx]
        # rhmvy = data['Moon vy (Helio)'].iloc[mm_analyzer.rel_idx]
        # rhmvz = data['Moon vz (Helio)'].iloc[mm_analyzer.rel_idx]
        #
        # name = new_data["Object id"].iloc[0]
        # H = mm_analyzer.H
        # D = getH2D(mm_analyzer.H) * 1000
        # cs = mm_analyzer.capture_starty
        # cd = mm_analyzer.capture_duration
        # espd = mm_analyzer.epsilon_duration
        # tehd = mm_analyzer.three_eh_duration
        # rev = mm_analyzer.revolutions
        # oehd = mm_analyzer.one_eh_duration
        # min_d = mm_analyzer.min_dist
        # ce = mm_analyzer.capture_end
        # retro = mm_analyzer.retrograde
        # minif = mm_analyzer.minimoon_flag
        # max_d = mm_analyzer.max_dist
        # cap_i = mm_analyzer.cap_idx
        # rel_i = mm_analyzer.rel_idx
        # x_eh = mm_analyzer.x_eh
        # y_eh = mm_analyzer.y_eh
        # z_eh = mm_analyzer.z_eh
        # designation = mm_analyzer.taxonomy(data, master)
        #
        # set to desired data
        # new_column[i] = oehd

        ############################################## to here

        # update that column in the master
        # master[] = new_column

        # write the master to csv - only if your sure you have the right data, otherwise in will be over written
        # master.to_csv(master_path, sep=' ', header=True, index=False)

        return

    def add_new_column(self, master_path, dest_path):

        # create parser
        mm_parser = MmParser(master_path, "", "")

        # create an analyzer
        mm_analyzer = MmAnalyzer()

        # get the master file - you need a list of initial orbits to integrate with openorb (pyorb)
        master = mm_parser.parse_master_new_new_new(dest_path)


        ########################################
        # Minimum Apparent magnitude as seen from Sun-Earth L_1
        #########################################

        # parallel implementation
        pool = multiprocessing.Pool()
        results = pool.map(mm_analyzer.minimum_apparent_magnitude, master['Object id'])
        pool.close()

        # repack list according to index
        repacked_results = [list(items) for items in zip(*results)]  # when running parallel processing

        # create your columns according to the data in results
        master['Min_SunEarthL1_V'] = repacked_results[0]  # min apparent mag.
        master['Min_SunEarthL1_V_index'] = repacked_results[1]  # corresponding index

        master.to_csv(dest_path, sep=' ', header=True, index=False)


        #####################################
        # Earth helio at capture
        ####################################

        # helio_E_x = master['Helio x at Capture'] - master['Geo x at Capture']
        # helio_E_y = master['Helio y at Capture'] - master['Geo y at Capture']
        # helio_E_z = master['Helio z at Capture'] - master['Geo z at Capture']
        # helio_E_vx = master['Helio vx at Capture'] - master['Geo vx at Capture']
        # helio_E_vy = master['Helio vy at Capture'] - master['Geo vy at Capture']
        # helio_E_vz = master['Helio vz at Capture'] - master['Geo vz at Capture']
        # desired_cols = ['Object id', '1 Hill Duration', 'Min. Distance', 'EMS Duration', 'Retrograde', 'STC',
        #                 'Became Minimoon', 'Taxonomy',
        #                 '3 Hill Duration', 'Helio x at Capture', 'Helio y at Capture',
        #                 'Helio z at Capture', 'Helio vx at Capture',
        #                 'Helio vy at Capture', 'Helio vz at Capture',
        #                 'Moon (Helio) x at Capture',
        #                 'Moon (Helio) y at Capture',
        #                 'Moon (Helio) z at Capture',
        #                 'Moon (Helio) vx at Capture',
        #                 'Moon (Helio) vy at Capture',
        #                 'Moon (Helio) vz at Capture', 'Capture Date', "Helio x at EMS", "Helio y at EMS",
        #                 "Helio z at EMS",
        #                 "Helio vx at EMS", "Helio vy at EMS", "Helio vz at EMS",
        #                 "Earth x at EMS (Helio)", "Earth y at EMS (Helio)",
        #                 "Earth z at EMS (Helio)", "Earth vx at EMS (Helio)",
        #                 "Earth vy at EMS (Helio)", "Earth vz at EMS (Helio)",
        #                 "Moon x at EMS (Helio)", "Moon y at EMS (Helio)",
        #                 "Moon z at EMS (Helio)", "Moon vx at EMS (Helio)",
        #                 "Moon vy at EMS (Helio)", "Moon vz at EMS (Helio)", "Entry Date to EMS"]
        #
        # cluster_df = master[desired_cols]
        # cluster_df.loc[:, ('Earth (Helio) x at Capture', 'Earth (Helio) y at Capture', 'Earth (Helio) z at Capture',
        #                    'Earth (Helio) vx at Capture', 'Earth (Helio) vy at Capture', 'Earth (Helio) vz at Capture')] \
        #     = np.array([helio_E_x, helio_E_y, helio_E_z, helio_E_vx, helio_E_vy, helio_E_vz]).T
        #
        # outlier_list = ['NESC00003lpo', 'NESC00000gRv', 'NESC00007H9p', 'NESC0000xeFs', 'NESC0000j1j1', 'NESC0000Fn0J',
        #                 'NESC0000EM8J', 'NESC0000EEZg', 'NESC0000BvQW', 'NESC0000BRzH', 'NESC0000aZ2t', 'NESC0000AY6C']
        #
        # cluster_df = cluster_df[~cluster_df['Object id'].isin(outlier_list)]
        # cluster_df.reset_index(drop=True, inplace=True)
        #
        # cluster_df.to_csv('cluster_df.csv', sep=' ', header=True, index=False)

        #######################################
        # parrelelized version of short term capture
        #########################################
        # pool = multiprocessing.Pool()
        # results = pool.map(mm_analyzer.short_term_capture, master['Object id'])  # input your function
        # pool.close()

        ########################################
        # alpha beta jacobi
        #######################################
        # for root, dirs, files in os.walk(dest_path):
        #
        #     for file in files:
        #         if file == 'minimoon_master_final (copy).csv' or file == 'minimoon_master_final.csv' or\
        #                 file == 'minimoon_master_final_previous.csv' or file == 'NESCv9reintv1.TCO.withH.kep.des':
        #             pass
        #         else:
        #             data = mm_parser.mm_file_parse_new(dest_path + '/' + file)
        #             object_id = data['Object id'].iloc[0]
        #             res = mm_analyzer.alpha_beta_jacobi(object_id)

        # parallel version of jacobi alpha beta
        # stc_pop = master[master['STC'] == True]
        # print(master)
        # for idx, row in stc_pop.iterrows():
        #     res = mm_analyzer.alpha_beta_jacobi(row['Object id'])
        #     res = mm_analyzer.short_term_capture(master['Object id'].iloc[i])
        # pool = multiprocessing.Pool()
        # results = pool.map(mm_analyzer.alpha_beta_jacobi, master['Object id'])
        # pool.close()

        # repack list according to index
        # repacked_results = [list(items) for items in zip(*results)]  # when running parallel processing

        # create your columns according to the data in results
        # master['Entry Date to EMS'] = repacked_results[10]  # start of ems stay
        # master['Entry to EMS Index'] = repacked_results[11]  # ems start index
        # master['Exit Date to EMS'] = repacked_results[12]  # end ems
        # master['Exit Index to EMS'] = repacked_results[13]  # end ems index
        #
        # pd.set_option('display.max_rows', None)
        # print(master['Entry Date to EMS'])

        # master['Dimensional Jacobi'] = repacked_results[0]
        # master['Non-Dimensional Jacobi'] = repacked_results[1]
        # master['Alpha_I'] = repacked_results[2]
        # master['Beta_I'] = repacked_results[3]
        # master['Theta_M'] = repacked_results[4]

        #################################################
        # cluster data
        ################################################

        # for idx, row in master.iterrows():
        #     idx += 2
        #     res = mm_analyzer.get_cluster_data(master['Object id'].iloc[idx])

        # write the master to csv - only if your sure you have the right data, otherwise in will be over written
        # master.to_csv(dest_path, sep=' ', header=True, index=False)

        return

    @staticmethod
    def cluster_viz_main(master_file):

        mm_pop = MmPopulation(master_file)
        mm_pop.cluster_viz()

    @staticmethod
    def stc_viz_main(master_file):

        mm_pop = MmPopulation(master_file)
        mm_pop.pop_viz()

    @staticmethod
    def alphabetastc(master_file):

        mm_parser = MmParser("", "", "")
        master = mm_parser.parse_master(master_file)

        bad_stc = master[(master['STC'] == True) & (master['Beta_I'] < -180)]
        population_dir = os.path.join(os.getcwd(), 'minimoon_files_oorb')

        for index2, row2 in bad_stc.iterrows():

            object_id = row2['Object id']
            print(row2['Object id'])

            xs = []
            ys = []
            zs = []
            vxs = []
            vys = []
            vzs = []
            xsdim = []
            ysdim = []
            zsdim = []
            vxsdim = []
            vysdim = []
            vzsdim = []
            xsdima = []
            ysdima = []
            zsdima = []
            vxsdima = []
            vysdima = []
            vzsdima = []
            ems = []
            distances = []
            coherence = []
            coherencea = []
            coherencedim = []
            hrx = []
            hry = []
            hrz = []
            hvx = []
            hvy = []
            hvz = []
            coherenceh = []

            for root, dirs, files in os.walk(population_dir):
                # find files that are minimoons
                name = str(object_id) + ".csv"

                if name in files:
                    file_path = os.path.join(root, name)

                    # read the file
                    data = mm_parser.mm_file_parse_new(file_path)

                    # get the jacobi constant using ephimeris data
                    seconds_in_day = 86400
                    km_in_au = 149597870700 / 1000

                    mu_s = 1.3271244e11 / np.power(km_in_au, 3)  # km^3/s^2 to AU^3/s^2
                    mu_e = 3.986e5  # km^3/s^2
                    mu_M = 4.9028e3  # km^3/s^2
                    mu_EMS = (mu_M + mu_e) / np.power(km_in_au, 3)  # km^3/s^2 = m_E + mu_M to AU^3/s^2
                    mu = mu_EMS / (mu_EMS + mu_s)
                    m_e = 5.97219e24  # mass of Earth kg
                    m_m = 7.34767309e22  # mass of the Moon kg
                    m_s = 1.9891e30  # mass of the Sun kg
                    r_ems = 0.0038752837677

                    for index, row in data.iterrows():
                        # print(row)
                        x = row['Helio x']  # AU
                        y = row['Helio y']
                        z = row['Helio z']
                        vx = row['Helio vx']  # AU/day
                        vy = row['Helio vy']
                        vz = row['Helio vz']
                        vx_M = row["Moon vx (Helio)"]  # AU/day
                        vy_M = row["Moon vy (Helio)"]
                        vz_M = row["Moon vz (Helio)"]
                        x_M = row["Moon x (Helio)"]  # AU
                        y_M = row["Moon y (Helio)"]
                        z_M = row["Moon z (Helio)"]
                        x_E = row["Earth x (Helio)"]
                        y_E = row["Earth y (Helio)"]
                        z_E = row["Earth z (Helio)"]
                        vx_E = row["Earth vx (Helio)"]  # AU/day
                        vy_E = row["Earth vy (Helio)"]
                        vz_E = row["Earth vz (Helio)"]
                        date = row['Julian Date']  # Julian date

                        if not np.isnan(row2['Entry to EMS Index']):
                            date_mjd = Time(date, format='jd').to_value('mjd')

                            h_r_TCO = np.array([x, y, z]).ravel()  # AU
                            h_r_M = np.array([x_M, y_M, z_M]).ravel()
                            h_r_E = np.array([x_E, y_E, z_E]).ravel()
                            h_v_TCO = np.array([vx, vy, vz]).ravel()  # AU/day
                            h_v_M = np.array([vx_M, vy_M, vz_M]).ravel()
                            h_v_E = np.array([vx_E, vy_E, vz_E]).ravel()

                            ems_barycentre = (m_e * h_r_E + m_m * h_r_M) / (
                                    m_m + m_e)  # heliocentric position of the ems barycentre AU
                            vems_barycentre = (m_e * h_v_E + m_m * h_v_M) / (
                                    m_m + m_e)  # heliocentric velocity of the ems barycentre AU/day

                            r_C = (m_e + m_m) * ems_barycentre / (m_e + m_m + m_s)  # barycentre of sun-earth/moon AU
                            r_sE = np.linalg.norm(ems_barycentre)  # distance between ems and sun AU
                            omega = np.sqrt(
                                (mu_s + mu_EMS) / np.power(r_sE, 3))  # angular velocity of sun-ems barycentre 1/s
                            omega_2 = np.cross(ems_barycentre, vems_barycentre) / r_sE ** 2

                            v_C = np.linalg.norm(r_C) / r_sE * vems_barycentre  # velocity of barycentre  AU/day

                            r = [km_in_au * ems_barycentre[0], km_in_au * ems_barycentre[1],
                                 km_in_au * ems_barycentre[2]] << u.km  # km
                            v = [km_in_au / seconds_in_day * vems_barycentre[0],
                                 km_in_au / seconds_in_day * vems_barycentre[1],
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

                            # v_rel in Jacobi constant
                            C_v_TCO = C_R_h @ (h_v_TCO - v_C) - np.cross(np.array([0, 0, omega * seconds_in_day]),
                                                                         C_r_TCO)  # AU/day
                            C_v_TCO_2 = C_R_h @ (h_v_TCO - v_C) - np.cross(C_R_h @ omega_2, C_r_TCO)  # AU/day

                            # non dimensional Jacobi constant
                            x_prime = C_r_TCO[0] / r_sE
                            y_prime = C_r_TCO[1] / r_sE
                            z_prime = C_r_TCO[2] / r_sE

                            x_dot_prime = C_v_TCO[0] / (omega * seconds_in_day * r_sE)
                            y_dot_prime = C_v_TCO[1] / (omega * seconds_in_day * r_sE)
                            z_dot_prime = C_v_TCO[2] / (omega * seconds_in_day * r_sE)

                            x_dot_prime_2 = C_v_TCO_2[0] / (np.linalg.norm(omega_2) * r_sE)
                            y_dot_prime_2 = C_v_TCO_2[1] / (np.linalg.norm(omega_2) * r_sE)
                            z_dot_prime_2 = C_v_TCO_2[2] / (np.linalg.norm(omega_2) * r_sE)

                            ################################
                            # Anderson and Lo
                            ###############################

                            # 1 - get state vectors (we have from proposed method)

                            # 2 - Length unit
                            # LU = r_sE

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

                            xs.append(x_prime)
                            ys.append(y_prime)
                            zs.append(z_prime)
                            vxs.append(x_dot_prime)
                            vys.append(y_dot_prime)
                            vzs.append(z_dot_prime)
                            xsdim.append(C_r_TCO[0])
                            ysdim.append(C_r_TCO[1])
                            zsdim.append(C_r_TCO[2])
                            vxsdim.append(C_v_TCO_2[0])
                            vysdim.append(C_v_TCO_2[1])
                            vzsdim.append(C_v_TCO_2[2])
                            xsdima.append(x_prime)
                            ysdima.append(y_prime)
                            zsdima.append(z_prime)
                            vxsdima.append(x_dot_prime_2)
                            vysdima.append(y_dot_prime_2)
                            vzsdima.append(z_dot_prime_2)
                            hrx.append(x)
                            hry.append(y)
                            hrz.append(z)
                            hvx.append(vx)
                            hvy.append(vy)
                            hvz.append(vz)
                            ems.append(C_R_h @ ems_barycentre - C_T_h)
                            distances.append(np.linalg.norm(h_r_TCO - ems_barycentre))

                            ridim = np.array([xsdim[index - 1], ysdim[index - 1], zsdim[index - 1]])
                            rip1dim = np.array([xsdim[index], ysdim[index], zsdim[index]])
                            vidim = np.array([vxsdim[index - 1], vysdim[index - 1], vzsdim[index - 1]])
                            vip1dim = np.array([vxsdim[index], vysdim[index], vzsdim[index]])
                            vimdim = (vidim + vip1dim) / np.linalg.norm(vidim + vip1dim)
                            didim = (rip1dim - ridim) / np.linalg.norm(rip1dim - ridim)

                            ri = np.array([xs[index - 1], ys[index - 1], zs[index - 1]])
                            rip1 = np.array([xs[index], ys[index], zs[index]])
                            vi = np.array([vxs[index - 1], vys[index - 1], vzs[index - 1]])
                            vip1 = np.array([vxs[index], vys[index], vzs[index]])
                            vim = (vi + vip1) / np.linalg.norm(vi + vip1)
                            di = (rip1 - ri) / np.linalg.norm(rip1 - ri)

                            ria = np.array([xsdima[index - 1], ysdima[index - 1], zsdima[index - 1]])
                            rip1a = np.array([xsdima[index], ysdima[index], zsdima[index]])
                            via = np.array([vxsdima[index - 1], vysdima[index - 1], vzsdima[index - 1]])
                            vip1a = np.array([vxsdima[index], vysdima[index], vzsdima[index]])
                            vima = (via + vip1a) / np.linalg.norm(via + vip1a)
                            dia = (rip1a - ria) / np.linalg.norm(rip1a - ria)

                            riah = np.array([hrx[index - 1], hry[index - 1], hrz[index - 1]])
                            rip1ah = np.array([hrx[index], hry[index], hrz[index]])
                            viah = np.array([hvx[index - 1], hvy[index - 1], hvz[index - 1]])
                            vip1ah = np.array([hvx[index], hvy[index], hvz[index]])
                            vimah = (viah + vip1ah) / np.linalg.norm(viah + vip1ah)
                            diah = (rip1ah - riah) / np.linalg.norm(rip1ah - riah)

                            coherence.append(di @ vim)  # omega
                            coherencea.append(dia @ vima)  # omega 2
                            coherencedim.append(didim @ vimdim)  # omega non-dimmed omega
                            coherenceh.append(diah @ vimah)  # heliocentric

            if xs:
                in_ems_idxs = int(row2['Entry to EMS Index'])
                fig3 = plt.figure()
                ax = fig3.add_subplot(111, projection='3d')
                vel_scale = 1
                ut, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
                xw = 0.0038752837677 / r_sE * np.cos(ut) * np.sin(v) + 1
                yw = 0.0038752837677 / r_sE * np.sin(ut) * np.sin(v)
                zw = 0.0038752837677 / r_sE * np.cos(v)
                ax.plot_wireframe(xw, yw, zw, color="b", alpha=0.1, label='SOI of EMS')
                ax.scatter(1, 0, 0, color='blue', s=50, label='Earth-Moon barycentre')
                ax.plot3D(xs, ys, zs, color='grey', zorder=10)
                ax.scatter(xs[in_ems_idxs], ys[in_ems_idxs], zs[in_ems_idxs], color='red', s=10,
                           label='$\omega_{SE}$ at SOI of EMS')
                # ax.scatter(xs[in_ems_idxs + 100], ys[in_ems_idxs + 100], zs[in_ems_idxs + 100], color='orange', s=10)
                ax.plot3D([xs[in_ems_idxs], xs[in_ems_idxs] + vel_scale * vxs[in_ems_idxs]],
                          [ys[in_ems_idxs], ys[in_ems_idxs] + vel_scale * vys[in_ems_idxs]],
                          [zs[in_ems_idxs], zs[in_ems_idxs] + vel_scale * vzs[in_ems_idxs]], color='red', zorder=15)
                # ax.plot3D([xs[in_ems_idxs + 100], xs[in_ems_idxs + 100] + vel_scale * vxs[in_ems_idxs + 100]],
                #           [ys[in_ems_idxs + 100], ys[in_ems_idxs + 100] + vel_scale * vys[in_ems_idxs + 100]],
                #           [zs[in_ems_idxs + 100], zs[in_ems_idxs + 100] + vel_scale * vzs[in_ems_idxs + 100]], color='orange',
                #           zorder=15)
                # ax.plot3D(xsdima, ysdima, zsdima)
                ax.scatter(xsdima[in_ems_idxs], ysdima[in_ems_idxs], zsdima[in_ems_idxs], color='blue', s=10,
                           label='$\omega^{\prime}_{SE}$ at SOI of EMS')
                ax.plot3D([xsdima[in_ems_idxs], xsdima[in_ems_idxs] + vel_scale * vxsdima[in_ems_idxs]],
                          [ysdima[in_ems_idxs], ysdima[in_ems_idxs] + vel_scale * vysdima[in_ems_idxs]],
                          [zsdima[in_ems_idxs], zsdima[in_ems_idxs] + vel_scale * vzsdima[in_ems_idxs]], color='blue',
                          zorder=25)

                ax.set_xlim([0.99, 1.01])
                ax.set_ylim([-0.01, 0.01])
                ax.set_zlim([-0.01, 0.01])
                num_ticks = 3
                ax.xaxis.set_major_locator(ticker.MaxNLocator(num_ticks))
                ax.yaxis.set_major_locator(ticker.MaxNLocator(num_ticks))
                ax.zaxis.set_major_locator(ticker.MaxNLocator(num_ticks))
                ax.set_xlabel('Synodic x ($\emptyset$)')
                ax.set_ylabel('Synodic y ($\emptyset$)')
                ax.set_zlabel('Synodic z ($\emptyset$)')
                ax.legend()

                fig = plt.figure()
                plt.scatter(1, 0, color='blue', s=80, label='Earth-Moon barycentre')
                plt.scatter(xs[in_ems_idxs], ys[in_ems_idxs], color='red', zorder=10, s=10,
                            label='$\omega_{SE}$ at SOI of EMS')
                plt.scatter(xsdima[in_ems_idxs], ysdima[in_ems_idxs], color='blue', s=30, zorder=5,
                            label='$\omega^{\prime}_{SE}$ at SOI of EMS')
                plt.plot([xs[in_ems_idxs], xs[in_ems_idxs] + vxs[in_ems_idxs]],
                         [ys[in_ems_idxs], ys[in_ems_idxs] + vys[in_ems_idxs]], color='red', zorder=15)
                plt.plot([xsdima[in_ems_idxs], xsdima[in_ems_idxs] + vxsdima[in_ems_idxs]],
                         [ysdima[in_ems_idxs], ysdima[in_ems_idxs] + vysdima[in_ems_idxs]], color='blue', zorder=5)
                plt.plot(xs, ys, color='grey', label='STC Trajectory')
                c1 = plt.Circle((1, 0), radius=0.0038752837677 / r_sE, alpha=0.1)
                plt.gca().add_artist(c1)
                plt.xlabel('Synodic x ($\emptyset$)')
                plt.ylabel('Synodix y ($\emptyset$)')
                plt.xlim([0.985, 1.015])
                plt.ylim([-0.015, 0.015])
                plt.legend()

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

                fig3 = plt.figure()
                print(row2['Object id'])
                print(row2['STC'])
                print("Coherence with omega 1: " + str(coherence[in_ems_idxs + 1]))
                print("Coherence with omega 2: " + str(coherencea[in_ems_idxs + 1]))
                print("Coherence of non-dim vel omega 1: " + str(coherencedim[in_ems_idxs + 1]))
                print("Coherence of helio vel: " + str(coherenceh[in_ems_idxs + 1]))
                plt.plot([i for i in range(1, len(coherence) + 1)], coherence, linewidth=1, zorder=5, color='red',
                         label='${}^Cv_{TCO\emptyset}$ with $\omega_{SE}$')
                plt.plot([i for i in range(1, len(coherencea) + 1)], coherencea, linewidth=1, zorder=10, color='blue',
                         label='${}^Cv_{TCO\emptyset}$ with $\omega^{\prime}_{SE}$')
                plt.plot([i for i in range(1, len(coherencedim) + 1)], coherencedim, linewidth=1, zorder=3,
                         color='green', label='${}^Cv_{TCO}$ with $\omega^{\prime}_{SE}$')
                plt.plot([i for i in range(1, len(coherenceh) + 1)], coherenceh, linewidth=3, zorder=2, color='orange',
                         label='${}^hv_{TCO}$')
                plt.scatter(in_ems_idxs, coherence[in_ems_idxs], s=20, zorder=5, color='red')
                plt.scatter(in_ems_idxs, coherencea[in_ems_idxs], s=20, zorder=10, color='blue')
                plt.scatter(in_ems_idxs, coherencedim[in_ems_idxs], s=20, zorder=3, color='green')
                plt.scatter(in_ems_idxs, coherenceh[in_ems_idxs], s=20, zorder=2, color='orange')
                plt.xlabel('Time-step')
                plt.ylabel('Coherence')
                plt.legend()

                # fig3 = plt.figure()
                # ax = fig3.add_subplot(111, projection='3d')
                # vel_scale = 1
                # ut, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
                # xw = 0.0038752837677 * np.cos(ut) * np.sin(v) + ems[in_ems_idxs][0]
                # yw = 0.0038752837677 * np.sin(ut) * np.sin(v) + ems[in_ems_idxs][1]
                # zw = 0.0038752837677 * np.cos(v) + + ems[in_ems_idxs][2]
                # ax.plot_wireframe(xw, yw, zw, color="b", alpha=0.1)
                # ax.scatter(+ ems[in_ems_idxs][0], + ems[in_ems_idxs][1], + ems[in_ems_idxs][2], color='blue', s=10)
                # ax.plot3D(xsdim, ysdim, zsdim, color='grey', zorder=10)
                # ax.scatter(xsdim[in_ems_idxs], ysdim[in_ems_idxs], zsdim[in_ems_idxs], color='red', s=10)
                # ax.scatter(xsdim[in_ems_idxs + 10], ysdim[in_ems_idxs + 10], zsdim[in_ems_idxs + 10], color='orange', s=10)
                # ax.plot3D([xsdim[in_ems_idxs], xsdim[in_ems_idxs] + vel_scale * vxsdim[in_ems_idxs]],
                #           [ysdim[in_ems_idxs], ysdim[in_ems_idxs] + vel_scale * vysdim[in_ems_idxs]],
                #           [zsdim[in_ems_idxs], zsdim[in_ems_idxs] + vel_scale * vzsdim[in_ems_idxs]], color='red', zorder=15)
                # ax.plot3D([xsdim[in_ems_idxs + 10], xsdim[in_ems_idxs + 10] + vel_scale * vxsdim[in_ems_idxs + 10]],
                #           [ysdim[in_ems_idxs + 10], ysdim[in_ems_idxs + 10] + vel_scale * vysdim[in_ems_idxs + 10]],
                #           [zsdim[in_ems_idxs + 10], zsdim[in_ems_idxs + 10] + vel_scale * vzsdim[in_ems_idxs + 10]], color='orange',
                #           zorder=15)
                #
                # ax.set_xlim([ems[in_ems_idxs][0] - 0.01, ems[in_ems_idxs][0] + 0.01])
                # ax.set_ylim([ems[in_ems_idxs][1] - 0.01, ems[in_ems_idxs][1] + 0.01])
                # ax.set_zlim([ems[in_ems_idxs][2] - 0.01, ems[in_ems_idxs][2] + 0.01])

                # fig3 = plt.figure()
                # ax = fig3.add_subplot(111, projection='3d')
                # vel_scale = 1
                # ut, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
                # xw = 0.0038752837677  * np.cos(ut) * np.sin(v) + data['Earth x (Helio)'].iloc[in_ems_idxs]
                # yw = 0.0038752837677  * np.sin(ut) * np.sin(v) + data['Earth y (Helio)'].iloc[in_ems_idxs]
                # zw = 0.0038752837677  * np.cos(v) + data['Earth z (Helio)'].iloc[in_ems_idxs]
                # ax.plot_wireframe(xw, yw, zw, color="b", alpha=0.1)
                # ax.scatter(data['Earth x (Helio)'].iloc[in_ems_idxs], data['Earth y (Helio)'].iloc[in_ems_idxs], data['Earth z (Helio)'].iloc[in_ems_idxs], color='blue', s=10)
                # ax.plot3D(data['Earth x (Helio)'], data['Earth y (Helio)'], data['Earth z (Helio)'], color='grey', zorder=10)
                # ax.scatter(data['Helio x'].iloc[in_ems_idxs], data['Helio y'].iloc[in_ems_idxs], data['Helio z'].iloc[in_ems_idxs], color='red', s=10)
                # ax.scatter(xs[in_ems_idxs + 10], ys[in_ems_idxs + 10], zs[in_ems_idxs + 10], color='orange', s=10)
                # ax.plot3D([data['Earth x (Helio)'].iloc[in_ems_idxs], data['Earth x (Helio)'].iloc[in_ems_idxs] + vel_scale * data['Earth vx (Helio)'].iloc[in_ems_idxs]],
                #           [data['Earth y (Helio)'].iloc[in_ems_idxs], data['Earth y (Helio)'].iloc[in_ems_idxs] + vel_scale * data['Earth vy (Helio)'].iloc[in_ems_idxs]],
                #           [data['Earth z (Helio)'].iloc[in_ems_idxs], data['Earth z (Helio)'].iloc[in_ems_idxs] + vel_scale * data['Earth vz (Helio)'].iloc[in_ems_idxs]], color='red', zorder=15)
                # ax.scatter(data['Earth x (Helio)'].iloc[in_ems_idxs], data['Earth y (Helio)'].iloc[in_ems_idxs],
                #            data['Earth z (Helio)'].iloc[in_ems_idxs], color='blue', s=10)
                # ax.plot3D(data['Moon x (Helio)'], data['Moon y (Helio)'], data['Moon z (Helio)'], color='red',
                #           zorder=10)
                # ax.scatter(data['Moon x (Helio)'].iloc[in_ems_idxs], data['Moon y (Helio)'].iloc[in_ems_idxs],
                #            data['Moon z (Helio)'].iloc[in_ems_idxs], color='orange', s=10)
                # ax.scatter(xs[in_ems_idxs + 10], ys[in_ems_idxs + 10], zs[in_ems_idxs + 10], color='orange', s=10)
                # ax.plot3D([data['Moon x (Helio)'].iloc[in_ems_idxs],
                #            data['Moon x (Helio)'].iloc[in_ems_idxs] + vel_scale * data['Moon vx (Helio)'].iloc[
                #                in_ems_idxs]],
                #           [data['Moon y (Helio)'].iloc[in_ems_idxs],
                #            data['Moon y (Helio)'].iloc[in_ems_idxs] + vel_scale * data['Moon vy (Helio)'].iloc[
                #                in_ems_idxs]],
                #           [data['Moon z (Helio)'].iloc[in_ems_idxs],
                #            data['Moon z (Helio)'].iloc[in_ems_idxs] + vel_scale * data['Moon vz (Helio)'].iloc[
                #                in_ems_idxs]], color='orange', zorder=15)
                #
                # ax.set_xlim([0.1, 0.4])
                # ax.set_ylim([-0.9, 1.1])
                # ax.set_zlim([-0.01, 0.01])
                plt.show()

    @staticmethod
    def no_moon_pop(master_file_nomoon, master_file):

        mm_pop = MmPopulation(master_file)
        mm_parser = MmParser("", "", "")
        path_nomoon = os.path.join('/media', 'aeromec', 'data', 'minimoon_files_oorb_nomoon')
        path_moon = os.path.join(os.getcwd(), 'minimoon_files_oorb')
        mm_pop_nomoon = mm_parser.parse_master_previous(master_file_nomoon)
        pd.set_option('display.max_rows', None)
        # print(mm_pop_nomoon.iloc[0])

        # Statistics on transitions
        stcstc_pop, stcnonstc_pop, nonstcnonstc_pop, nonstcstc_pop = mm_pop.no_moon_table(mm_pop, mm_pop_nomoon)

        # Jacobi constant graphs
        fig = plt.figure()
        plt.scatter(stcnonstc_pop['3 Hill Duration'], stcnonstc_pop['Non-Dimensional Jacobi'], s=1, color='red',
                    label='Became Non-STC without influence of the Moon')
        plt.scatter(stcstc_pop['3 Hill Duration'], stcstc_pop['Non-Dimensional Jacobi'], s=1, color='blue',
                    label='Remained STC without influence of the Moon')
        plt.plot(np.linspace(0, 1000, 200), 2.9999 * np.linspace(1, 1, 200), linestyle='--', color='green', linewidth=1)
        plt.xlabel('Capture Duration (days)')
        plt.ylabel('Jacobi Constant ($\emptyset$)')
        plt.xlim([0, 1000])
        plt.ylim([2.9985, 3.0015])
        plt.legend()

        # fig = plt.figure()
        # plt.scatter(stcstc_pop['3 Hill Duration'], stcstc_pop['Non-Dimensional Jacobi'], s=1, color='blue')
        # plt.plot(np.linspace(0, 1000, 200), 2.9999 * np.linspace(1, 1, 200), linestyle='--',
        #          color='green', linewidth=1)
        # plt.xlabel('Capture Duration (days)')
        # plt.ylabel('Jacobi Constant ($\emptyset$)')
        # plt.xlim([0, 1000])
        # plt.ylim([2.9985, 3.0015])
        #
        # plt.show()

        # Examine the planar population of STCs
        stc_stayed_stc_planar, stc_became_nonstc_planar = mm_pop.planar_stc(mm_pop, mm_pop_nomoon, path_moon,
                                                                            path_nomoon)

        stc_stayed_stc_planar = nonstcstc_pop
        stc_became_nonstc_planar = stcnonstc_pop

        vels_stayed = []
        vels_became_nonstc = []
        thills_stayed = []
        thills_became = []
        for idx3, master in stc_stayed_stc_planar.iterrows():

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
            date_ems = master['Entry Date to EMS']  # Julian date

            if not np.isnan(date_ems):

                date_mjd = Time(date_ems, format='jd').to_value('mjd')

                h_r_TCO = np.array([x, y, z]).ravel()  # AU
                h_r_M = np.array([x_M, y_M, z_M]).ravel()
                h_r_E = np.array([x_E, y_E, z_E]).ravel()
                h_v_TCO = np.array([vx, vy, vz]).ravel()  # AU/day
                h_v_M = np.array([vx_M, vy_M, vz_M]).ravel()
                h_v_E = np.array([vx_E, vy_E, vz_E]).ravel()

                C_r_TCO, C_v_TCO, C_v_TCO_2, C_ems, C_moon, ems_barycentre, vems_barycentre, omega, omega_2, r_sE, mu_s, mu_EMS = (
                    get_r_and_v_cr3bp_from_nbody_sun_emb(h_r_TCO, h_v_TCO, h_r_E, h_v_E, h_r_M, h_v_M, date_mjd))

                if True:  # np.linalg.norm(C_v_TCO_2[2]) < 2.2e-6:
                    val = np.linalg.norm(C_v_TCO_2) / r_sE / np.linalg.norm(
                        omega_2)  # * np.sin(np.deg2rad(abs(90 + master['Beta_I'])))
                    vels_stayed.append(val)
                    thills_stayed.append(master['Alpha_I'])
                    print(str(master['Object id']) + "'s vel: " + str(val))

        for idx3, master in stc_became_nonstc_planar.iterrows():

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
            date_ems = master['Entry Date to EMS']  # Julian date

            if not np.isnan(date_ems):
                date_mjd = Time(date_ems, format='jd').to_value('mjd')

                h_r_TCO = np.array([x, y, z]).ravel()  # AU
                h_r_M = np.array([x_M, y_M, z_M]).ravel()
                h_r_E = np.array([x_E, y_E, z_E]).ravel()
                h_v_TCO = np.array([vx, vy, vz]).ravel()  # AU/day
                h_v_M = np.array([vx_M, vy_M, vz_M]).ravel()
                h_v_E = np.array([vx_E, vy_E, vz_E]).ravel()

                C_r_TCO, C_v_TCO, C_v_TCO_2, C_ems, C_moon, ems_barycentre, vems_barycentre, omega, omega_2, r_sE, mu_s, mu_EMS = (
                    get_r_and_v_cr3bp_from_nbody_sun_emb(h_r_TCO, h_v_TCO, h_r_E, h_v_E, h_r_M, h_v_M, date_mjd))

                if True:  # np.linalg.norm(C_v_TCO_2[2]) < 2.2e-6:
                    val = np.linalg.norm(C_v_TCO_2) / r_sE / np.linalg.norm(
                        omega_2)  # * np.sin(np.deg2rad(abs(90 + master['Beta_I'])))
                    vels_became_nonstc.append(val)
                    thills_became.append(master['Alpha_I'])
                    print(str(master['Object id']) + "'s vel: " + str(val))

        fig = plt.figure()
        plt.scatter(thills_stayed, vels_stayed, s=5, color='blue')
        plt.scatter(thills_became, vels_became_nonstc, s=5, color='red')
        plt.show()

        # stcs_non.to_csv('minimoon_files_oorb/minimoon_master_nonstcs_nomoon.csv', sep=' ', header=True, index=False)
        # stcs_non.to_csv('minimoon_files_oorb/minimoon_master_nonstcs_nomoon.csv', sep=' ', header=True, index=False)

    """
            stc_to_non_stcs = pd.DataFrame(columns=['Object id', 'H', 'D', 'Capture Date',
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
                                                             "Moon vy at EMS (Helio)", "Moon vz at EMS (Helio)",
                                                             'Entry Date to EMS', 'Entry to EMS Index',
                                                             'Exit Date to EMS', 'Exit Index to EMS',
                                                             "Dimensional Jacobi", "Non-Dimensional Jacobi", 'Alpha_I',
                                                             'Beta_I', 'Theta_M'])




        
        vels_stayed_non = []
        vels_became_nonstc_non2 = []
        thills_stayed_non = []
        thills_became_non2 = []
        for idx3, master in stcs_non.iterrows():

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
            date_ems = master['Entry Date to EMS']  # Julian date

            if not np.isnan(date_ems):

                date_mjd = Time(date_ems, format='jd').to_value('mjd')

                h_r_TCO = np.array([x, y, z]).ravel()  # AU
                h_r_M = np.array([x_M, y_M, z_M]).ravel()
                h_r_E = np.array([x_E, y_E, z_E]).ravel()
                h_v_TCO = np.array([vx, vy, vz]).ravel()  # AU/day
                h_v_M = np.array([vx_M, vy_M, vz_M]).ravel()
                h_v_E = np.array([vx_E, vy_E, vz_E]).ravel()

                C_r_TCO, C_v_TCO, C_v_TCO_2, C_ems, C_moon, ems_barycentre, vems_barycentre, omega, omega_2, r_sE, mu_s, mu_EMS = (
                    get_r_and_v_cr3bp_from_nbody_sun_emb(h_r_TCO, h_v_TCO, h_r_E, h_v_E, h_r_M, h_v_M, date_mjd))

                if master['Beta_I'] > -180 and master['Beta_I'] < 0: #if np.linalg.norm(C_v_TCO_2[2]) < 2.2e-6:
                    vels_stayed_non.append(
                        np.linalg.norm(C_v_TCO_2[:3]) * np.cos(np.deg2rad(abs(90 + master['Beta_I']))))
                    thills_stayed_non.append(master['3 Hill Duration'])

        for idx3, master in stcs_non2.iterrows():

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
            date_ems = master['Entry Date to EMS']  # Julian date

            if not np.isnan(date_ems):

                date_mjd = Time(date_ems, format='jd').to_value('mjd')

                h_r_TCO = np.array([x, y, z]).ravel()  # AU
                h_r_M = np.array([x_M, y_M, z_M]).ravel()
                h_r_E = np.array([x_E, y_E, z_E]).ravel()
                h_v_TCO = np.array([vx, vy, vz]).ravel()  # AU/day
                h_v_M = np.array([vx_M, vy_M, vz_M]).ravel()
                h_v_E = np.array([vx_E, vy_E, vz_E]).ravel()

                C_r_TCO, C_v_TCO, C_v_TCO_2, C_ems, C_moon, ems_barycentre, vems_barycentre, omega, omega_2, r_sE, mu_s, mu_EMS = (
                    get_r_and_v_cr3bp_from_nbody_sun_emb(h_r_TCO, h_v_TCO, h_r_E, h_v_E, h_r_M, h_v_M,
                                                         date_mjd))

                if master['Beta_I'] > -180 and master['Beta_I'] < 0: #if np.linalg.norm(C_v_TCO_2[2]) < 2.2e-6:
                    vels_became_nonstc_non2.append(
                        np.linalg.norm(C_v_TCO_2[:3]) * np.cos(np.deg2rad(abs(90 + master['Beta_I']))))
                    thills_became_non2.append(master['3 Hill Duration'])

        fig = plt.figure()
        plt.scatter(thills_stayed_non, vels_stayed_non, s=1, color='blue')
        plt.scatter(thills_became_non2, vels_became_nonstc_non2, s=1, color='red')

        # Set the bin size
        bin_size = 0.00001

        # Calculate the number of bins based on data range and bin size
        data_range_stc = max(vels_stayed_non) - min(vels_stayed_non)
        num_bins_stc = int(data_range_stc / bin_size)
        data_range_nonstc = max(vels_became_nonstc_non2) - min(vels_became_nonstc_non2)
        num_bins_nonstc = int(data_range_nonstc / bin_size)

        fig = plt.figure()
        plt.hist(vels_stayed_non, bins=num_bins_stc, label='STCs', edgecolor="#038cfc",
                 color="#03b1fc")
        plt.hist(vels_became_nonstc_non2, bins=num_bins_nonstc, label='Non-STCs', edgecolor="#ed0000",
                 color="#f54747")
        # plt.xlim([0, 60])
        # plt.ylim([0, 3500])
        plt.xlabel('Duration in the SOI of EMS (days)')
        plt.ylabel('Count')
        plt.legend()

        plt.show()
        print(stcs_non['STC'])
        print(stcs_non2['STC'])
        print(len(stcs_non))
        print(len(stcs_non2))
        
            
            # constants
            three_eh = 0.03
            r_ems = 0.0038752837677  # sphere of influence of earth-moon system
            two_hill = 0.02
            one_hill = 0.01

            print("Analyzing the Short-Term Capture Statistics of minimoon with moon: " + str(stc_name))

            # get data with respect to the earth-moon barycentre in a co-rotating frame
            emb_xyz_synodic = get_emb_synodic(data_moon)

            stc_moon_dist_moon = [np.linalg.norm(np.array([rows['Helio x'] - rows['Moon x (Helio)'],
                                                          rows['Helio y'] - rows['Moon y (Helio)'],
                                                          rows['Helio z'] - rows['Moon z (Helio)']])) for i, rows in data_moon.iterrows()]

            # the sampling interval of the moon was 1/23, not 1/24
            stc_moon_dist_nomoon = [np.linalg.norm(np.array([rows2['Helio x'] - rows2['Moon x (Helio)'],
                                                          rows2['Helio y'] - rows2['Moon y (Helio)'],
                                                          rows2['Helio z'] - rows2['Moon z (Helio)']])) for i, rows2 in data_nomoon.iterrows()]

            distance_emb_synodic = np.sqrt(
                emb_xyz_synodic[:, 0] ** 2 + emb_xyz_synodic[:, 1] ** 2 + emb_xyz_synodic[:, 2] ** 2)

            # identify when inside the 3 earth hill sphere
            three_hill_under = np.NAN * np.zeros((len(distance_emb_synodic),))
            three_hill_idxs = [index for index, value in enumerate(distance_emb_synodic) if value <= three_eh]
            for index in three_hill_idxs:
                three_hill_under[index] = 1


            captured_distance = distance_emb_synodic * three_hill_under

            # identify periapses that exist in the 3 earth hill sphere
            local_minima_indices = argrelextrema(captured_distance, np.less)[0]
            local_dist = captured_distance[local_minima_indices]
            time = data_moon["Julian Date"] - data_moon["Julian Date"].iloc[0]
            local_time = time.iloc[local_minima_indices]

            # identify when inside the sphere of influence of the EMS
            in_ems = np.NAN * np.zeros((len(distance_emb_synodic),))
            in_ems_idxs = [index for index, value in enumerate(distance_emb_synodic) if value <= r_ems]
            for index in in_ems_idxs:
                in_ems[index] = 1

            captured_distance_ems = distance_emb_synodic * in_ems

            # identify periapses that exist in the EMS SOI
            local_minima_indices_ems = argrelextrema(captured_distance_ems, np.less)[0]
            local_dist_ems = captured_distance_ems[local_minima_indices_ems]
            local_time_ems = time.iloc[local_minima_indices_ems]


            ems_line = r_ems * np.ones(len(time), )
            three_eh_line = three_eh * np.ones(len(time), )


            stc = False
            # decide if short-term capture or not
            if len(three_hill_idxs) >= 2:
                if len(local_minima_indices_ems) >= 2:
                    stc = True

            print(str(stc_name) + ": " + str(stc))

            # Data of interest (for the master):
            # Whether a STC took place or not
            # Time spent in SOI EMS
            time_ste = data_moon["Julian Date"].iloc[1] - data_moon["Julian Date"].iloc[0]

            print("Analyzing the Short-Term Capture Statistics of minimoon without moon: " + str(stc_name_no_moon))

            # get data with respect to the earth-moon barycentre in a co-rotating frame
            emb_xyz_synodicnm = get_emb_synodic(data_nomoon)

            distance_emb_synodicnm = np.sqrt(
                emb_xyz_synodicnm[:, 0] ** 2 + emb_xyz_synodicnm[:, 1] ** 2 + emb_xyz_synodicnm[:, 2] ** 2)

            # identify when inside the 3 earth hill sphere
            three_hill_undernm = np.NAN * np.zeros((len(distance_emb_synodicnm),))
            three_hill_idxsnm = [index for index, value in enumerate(distance_emb_synodicnm) if value <= three_eh]
            for index in three_hill_idxsnm:
                three_hill_undernm[index] = 1
            captured_distancenm = distance_emb_synodicnm * three_hill_undernm

            # identify periapses that exist in the 3 earth hill sphere
            local_minima_indicesnm = argrelextrema(captured_distancenm, np.less)[0]
            local_distnm = captured_distancenm[local_minima_indicesnm]
            timenm = data_nomoon["Julian Date"] - data_nomoon["Julian Date"].iloc[0]
            local_timenm = timenm.iloc[local_minima_indicesnm]

            # identify when inside the sphere of influence of the EMS
            in_emsnm = np.NAN * np.zeros((len(distance_emb_synodicnm),))
            in_ems_idxsnm = [index for index, value in enumerate(distance_emb_synodicnm) if value <= r_ems]
            for index in in_ems_idxsnm:
                in_emsnm[index] = 1
            captured_distance_emsnm = distance_emb_synodicnm * in_emsnm

            # identify periapses that exist in the EMS SOI
            local_minima_indices_emsnm = argrelextrema(captured_distance_emsnm, np.less)[0]
            local_dist_emsnm = captured_distance_emsnm[local_minima_indices_emsnm]
            local_time_emsnm = timenm.iloc[local_minima_indices_emsnm]

            stcnm = False
            # decide if short-term capture or not
            if len(three_hill_idxsnm) >= 2:
                if len(local_minima_indices_emsnm) >= 2:
                    stcnm = True

            print(str(stc_name_no_moon) + ": " + str(stcnm))

            # Data of interest (for the master):
            # Whether a STC took place or not
            # Time spent in SOI EMS
            time_stepnm = data_nomoon["Julian Date"].iloc[1] - data_nomoon["Julian Date"].iloc[0]
            time_SOI_EMSnm = time_stepnm * len(in_ems_idxsnm)

            print(1/time_stepnm)
            print(1/time_ste)
            print(data_nomoon["Julian Date"].iloc[0])
            print(data_moon["Julian Date"].iloc[0])

            fig2 = plt.figure()
            plt.plot(time, data_moon['Moon x (Helio)'], linestyle='-', color='blue')
            plt.plot(time, data_moon['Moon y (Helio)'], linestyle='-', color='green')
            plt.plot(time, data_moon['Moon z (Helio)'], linestyle='-', color='red')
            plt.plot(timenm, data_nomoon['Moon x (Helio)'], linestyle='--', color='blue')
            plt.plot(timenm, data_nomoon['Moon y (Helio)'], linestyle='--', color='green')
            plt.plot(timenm, data_nomoon['Moon z (Helio)'], linestyle='--', color='red')

            fig = plt.figure()
            plt.plot(time, captured_distance, color='#5599ff', linewidth=3, zorder=5,
                     label='Inside 3 Earth Hill')
            plt.plot(time, captured_distance_ems, color='red', linewidth=5, zorder=6, label='Inside SOI of EMS')
            plt.plot(time, distance_emb_synodic, color='grey',
                     linewidth=1, zorder=7, label='STC with Moon')
            plt.scatter(local_time, local_dist, color='#ff80ff', zorder=8,
                        label='Periapsides Inside 3 Earth Hill')
            plt.scatter(local_time_ems, local_dist_ems, color='blue', zorder=9,
                        label='Periapsides Inside SOI of EMS')
            # if in_ems_idxs:
            #     plt.scatter(time.iloc[in_ems_idxs[0]], distance_emb_synodic[in_ems_idxs[0]], zorder=15)
            plt.plot(timenm, captured_distancenm, color='#5599ff', linewidth=3, zorder=5)
            plt.plot(timenm, captured_distance_emsnm, color='red', linewidth=5, zorder=6)
            plt.plot(timenm, distance_emb_synodicnm, color='orange',
                     linewidth=1, zorder=7, label='STC without Moon')
            plt.scatter(local_timenm, local_distnm, color='#ff80ff', zorder=8)
            plt.scatter(local_time_emsnm, local_dist_emsnm, color='blue', zorder=9)
            # if in_ems_idxsnm:
            #     plt.scatter(timenm.iloc[in_ems_idxsnm[0]], distance_emb_synodicnm[in_ems_idxsnm[0]], zorder=15)
            plt.plot(time, three_eh_line, linestyle='--', color='red', zorder=4, label='3 Earth Hill')
            plt.plot(time, ems_line, linestyle='--', color='green', zorder=3, label='SOI of EMS')
            plt.plot(time, stc_moon_dist_moon, linestyle='--', color='red', label='STC Moon Distance')
            # plt.plot(timenm, stc_moon_dist_nomoon, linestyle='--', color='blue', label='STC Moon Distance without Moon')
            plt.legend()
            plt.xlabel('Time (days)')
            plt.ylabel('Distance from Earth/Moon Barycentre (AU)')
            plt.title(str(stc_name_no_moon))
            # plt.ylim([0, 0.06])
            # plt.xlim([0, time.iloc[-1]])
            # plt.savefig("figures/" + str(stc_name_no_moon) + "_distance_nomoon.svg", format="svg")
            # plt.savefig("figures/" + str(stc_name_no_moon) + "_distance_nomoon.png", format="png")
            plt.show()
    """

    @staticmethod
    def jacobi_variation():

        actual_planar_ids = ['2006 RH120', 'NESC00000Opf', 'NESC00001xp6', 'NESC00003HO8', 'NESC00004Hzu',
                             'NESC00004m1B',
                             'NESC00004zBZ',
                             'NESC00009F39', 'NESC0000as6C', 'NESC0000AWYz', 'NESC0000BHG1', 'NESC0000CdOz',
                             'NESC0000dbfP',
                             'NESC0000dPxh', 'NESC0000dR3v', 'NESC0000ds7v', 'NESC0000dw0G', 'NESC0000eGj2',
                             'NESC0000EXSB',
                             'NESC0000m2AL', 'NESC0000nlWD', 'NESC0000qF2S', 'NESC0000u8R8', 'NESC0000wMjh',
                             'NESC0000yn24',
                             'NESC0000zHqv']

        destination_path = os.path.join(os.getcwd(), 'minimoon_files_oorb')

        mm_parser = MmParser("", "", "")

        jacobis = []
        fig, ax = plt.subplots()
        for idx, id in enumerate(actual_planar_ids):

            file_path = destination_path + '/' + id + '.csv'
            print(id)

            data = mm_parser.mm_file_parse_new(file_path)
            jacobi = []
            for j, master in data.iterrows():
                # print(j)

                x = master['Helio x']  # AU
                y = master['Helio y']
                z = master['Helio z']
                vx = master['Helio vx']  # AU/day
                vy = master['Helio vy']
                vz = master['Helio vz']
                vx_M = master['Moon vx (Helio)']  # AU/day
                vy_M = master['Moon vy (Helio)']
                vz_M = master['Moon vz (Helio)']
                x_M = master['Moon x (Helio)']  # AU
                y_M = master['Moon y (Helio)']
                z_M = master['Moon z (Helio)']
                x_E = master['Earth x (Helio)']
                y_E = master['Earth y (Helio)']
                z_E = master['Earth z (Helio)']
                vx_E = master['Earth vx (Helio)']  # AU/day
                vy_E = master['Earth vy (Helio)']
                vz_E = master['Earth vz (Helio)']
                date_ems = master['Julian Date']  # Julian date

                date_mjd = Time(date_ems, format='jd').to_value('mjd')
                h_r_TCO = np.array([x, y, z]).ravel()  # AU
                h_r_M = np.array([x_M, y_M, z_M]).ravel()
                h_r_E = np.array([x_E, y_E, z_E]).ravel()
                h_v_TCO = np.array([vx, vy, vz]).ravel()  # AU/day
                h_v_M = np.array([vx_M, vy_M, vz_M]).ravel()
                h_v_E = np.array([vx_E, vy_E, vz_E]).ravel()

                C_r_TCO, C_v_TCO, C_v_TCO_2, C_ems, C_moon, ems_barycentre, vems_barycentre, omega, omega_2, r_sE, mu_s, mu_EMS = (
                    get_r_and_v_cr3bp_from_nbody_sun_emb(h_r_TCO, h_v_TCO, h_r_E, h_v_E, h_r_M, h_v_M, date_mjd))

                mu = mu_EMS / (mu_EMS + mu_s)

                good_r = C_r_TCO
                good_v = C_v_TCO_2
                good_omega = omega_2
                C_J_dimensional, C_J_nondimensional = jacobi_dim_and_non_dim(good_r, good_v, h_r_TCO, ems_barycentre,
                                                                             mu, mu_s, mu_EMS, good_omega, r_sE)
                jacobi.append(C_J_nondimensional)

            time = np.linspace(0, data['Julian Date'].iloc[-1] - data['Julian Date'].iloc[0], len(data['Julian Date']))

            window_size = 600
            weights = np.ones(window_size) / window_size
            new_jacob = np.convolve(jacobi, weights, mode='valid')

            # plt.plot(time, jacobi, label=id)
            plt.plot(time[:len(new_jacob)], new_jacob, label=id)

            ax.get_yaxis().get_major_formatter().set_useOffset(False)
            if idx > 4:
                plt.plot(time[:len(new_jacob)], 2.999 * np.ones((len(time[:len(new_jacob)]),)), color='black')
                plt.plot(time[:len(new_jacob)], 3.001 * np.ones((len(time[:len(new_jacob)]),)), color='black')
                plt.xlabel('Time (Days)')
                plt.ylabel('Jacobi Constant ($\emptyset$)')
                plt.legend()
                plt.show()

            jacobis.append(jacobi)

    @staticmethod
    def BCR4BP():

        data_stc = pd.read_csv('Databases/stc_nonstc2.csv', sep=' ', header=0, names=['Object id', 'EM Syn. at SOIEMS x',
                                                                            'EM Syn. at SOIEMS y',
                                                                            'EM Syn. at SOIEMS z',
                                                                            'Moon at SOIEMS x', 'Moon at SOIEMS y',
                                                                            'Moon at SOIEMS z',
                                                                            'EM Syn. at SOIEMS vx',
                                                                            'EM Syn. at SOIEMS vy',
                                                                            'EM Syn. at SOIEMS vz', 'STC without Moon',
                                                                            'Index',
                                                                            "First Perigee Distance",
                                                                            "Time Between Perigees"])

        moon_vel = []
        posvelratio = []
        for k, master_k in data_stc.iterrows():
            tco_moon = np.array([master_k["Moon at SOIEMS x"] - master_k["EM Syn. at SOIEMS x"],
                                 master_k["Moon at SOIEMS y"] - master_k["EM Syn. at SOIEMS y"],
                                 master_k["Moon at SOIEMS z"] - master_k['EM Syn. at SOIEMS z']])
            u_tco_moon = tco_moon / np.linalg.norm(tco_moon)
            vel_tco = np.array(
                [master_k["EM Syn. at SOIEMS vx"], master_k["EM Syn. at SOIEMS vy"], master_k['EM Syn. at SOIEMS vz']])
            r_tco = np.array(
                [master_k['EM Syn. at SOIEMS x'], master_k['EM Syn. at SOIEMS y'], master_k['EM Syn. at SOIEMS z']])
            moon_vel.append((np.dot(u_tco_moon, vel_tco) + (
                        np.linalg.norm(vel_tco) - np.dot(-r_tco / np.linalg.norm(r_tco), vel_tco))) / master_k[
                                'Time Between Perigees'] * 8760)
            posvelratio.append(np.linalg.norm(tco_moon) / np.linalg.norm(vel_tco))

        data_stc['New Index'] = moon_vel
        # data_stc['Ratio'] = posvelratio
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        fig = plt.figure()
        # data_stc = data_stc[(data_stc['EM Syn. at SOIEMS x'] < 0) & (data_stc['STC without Moon'] == False)]
        stc = data_stc[data_stc['STC without Moon'] == True]
        nonstc = data_stc[data_stc['STC without Moon'] == False]
        plt.scatter(stc["EM Syn. at SOIEMS x"], stc['New Index'], color='blue', s=5)
        plt.scatter(nonstc["EM Syn. at SOIEMS x"], nonstc['New Index'], color='red', s=5)
        plt.show()
        print(data_stc.loc[:, ('Object id', 'STC without Moon', 'New Index', 'Time Between Perigees')])
        input()
        # actual_planar_ids = ['NESC00000Opf', 'NESC00001xp6', 'NESC00003HO8', 'NESC00004Hzu', 'NESC00004m1B',
        #                      'NESC00004zBZ', 'NESC00009F39', 'NESC0000as6C', 'NESC0000AWYz', 'NESC0000BHG1',
        #                      'NESC0000CdOz', 'NESC0000dbfP', 'NESC0000dPxh', 'NESC0000dR3v', 'NESC0000ds7v',
        #                      'NESC0000dw0G', 'NESC0000eGj2', 'NESC0000EXSB', 'NESC0000m2AL', 'NESC0000nlWD',
        #                      'NESC0000qF2S', 'NESC0000u8R8', 'NESC0000wMjh', 'NESC0000yn24', 'NESC0000zHqv']

        # master file with moon and without moon
        mm_parser = MmParser("", "", "")
        destination_path = os.path.join(os.getcwd(), 'minimoon_files_oorb')
        path_nomoon = os.path.join(os.getcwd(), 'Test_Set_nomoon')
        master_file = mm_parser.parse_master(destination_path + '/minimoon_master_final.csv')
        # master_file_nomoon = mm_parser.parse_master_previous(path_nomoon + '/minimoon_master_final.csv')
        # mm_pop = MmPopulation(destination_path + '/minimoon_master_final.csv')

        # Statistics on transitions
        # stcstc_pop, stcnonstc_pop, nonstcnonstc_pop, nonstcstc_pop = mm_pop.no_moon_table(mm_pop, master_file_nomoon)
        # stcstc_pop.to_csv('stcstc.csv', sep=' ', header=True, index=False)
        # stcnonstc_pop.to_csv('stcnonstc.csv', sep=' ', header=True, index=False)
        # nonstcnonstc_pop.to_csv('nonstcnonstc.csv', sep=' ', header=True, index=False)
        # nonstcstc_pop.to_csv('nonstcstc.csv', sep=' ', header=True, index=False)

        # stcstc_file = 'stcstc.csv'
        # nonstcnonstc_file = 'nonstcnonstc.csv'
        stcnonstc_file = 'Databases/stcnonstc.csv'
        # nonstcstc_file = 'nonstcstc.csv'
        #
        # mm_parser = MmParser("", "", "")
        # stcstc_pop = mm_parser.parse_master(os.path.join(os.getcwd(), stcstc_file))
        # nonstcnonstc_pop = mm_parser.parse_master(os.path.join(os.getcwd(), nonstcnonstc_file))
        stcnonstc_pop = mm_parser.parse_master(os.path.join(os.getcwd(), stcnonstc_file))
        # nonstcstc_pop = mm_parser.parse_master(os.path.join(os.getcwd(), nonstcstc_file))
        no_moon_master = mm_parser.parse_master_previous(
            os.path.join(os.getcwd(), 'Test_Set_nomoon', 'minimoon_master_final.csv'))

        # master_file = stcnonstc_pop

        # actual_planar_ids = ['NESC00000Opf', 'NESC00001xp6', 'NESC00003HO8', 'NESC00004Hzu', 'NESC00004m1B',
        #                      'NESC00004zBZ', 'NESC00009F39', 'NESC0000as6C', 'NESC0000AWYz', 'NESC0000BHG1',
        #                      'NESC0000CdOz', 'NESC0000dbfP', 'NESC0000dPxh', 'NESC0000dR3v', 'NESC0000ds7v',
        #                      'NESC0000dw0G', 'NESC0000eGj2', 'NESC0000EXSB', 'NESC0000m2AL', 'NESC0000nlWD',
        #                      'NESC0000qF2S', 'NESC0000u8R8', 'NESC0000wMjh', 'NESC0000yn24', 'NESC0000zHqv']

        # planar_data['Object id'] = actual_planar_ids
        # planar_data['Metric'] = moon_vel
        # planar_data.to_csv('planar_datas.csv', sep=' ', header=True, index=False)

        # df = pd.read_csv('stc_nonstc.csv', sep=" ", header=0, names=['Object id', 'EM Syn. at SOIEMS x',
        #                                                              'EM Syn. at SOIEMS y', 'EM Syn. at SOIEMS z',
        #                                                              'Moon at SOIEMS x', 'Moon at SOIEMS y',
        #                                                              'Moon at SOIEMS z',
        #                                                              'EM Syn. at SOIEMS vx', 'EM Syn. at SOIEMS vy',
        #                                                              'EM Syn. at SOIEMS vz', 'STC without Moon'])

        important_params = []
        stc_pop = master_file[master_file['STC'] == True]
        # last stopped NESC00003SVp
        peris = []
        dts = []
        for idx, id in enumerate(data_stc['Object id'].iloc[0:]):

            file_path = destination_path + '/' + id + '.csv'
            print(id)

            ########################################
            # Integrate Initializations
            #########################################
            no_moon_path = os.path.join(os.getcwd(), 'no_moon_files', id + '.csv')
            if os.path.isfile(no_moon_path):
                no_moon_data = mm_parser.mm_file_parse_new_no_moon(no_moon_path)
            else:
                mm_main.integrate_parallel(idx, id)
                no_moon_data = mm_parser.mm_file_parse_new(no_moon_path)
                nmx = []
                nmy = []
                nmz = []
                nmvx = []
                nmvy = []
                nmvz = []
                nmomx = []
                nmomy = []
                nmomz = []
                nmetco = []
                nmmtco = []
                nmstco = []
                nmsunx = []
                nmsuny = []
                nmsunz = []
                nmmoonx = []
                nmmoony = []
                mnmoonz = []
                mnemd = []
                emomx = []
                emomy = []
                emomz = []
                for i, master in no_moon_data.iterrows():
                    x = master['Helio x']
                    y = master['Helio y']
                    z = master['Helio z']
                    vx = master['Helio vx']  # AU/day
                    vy = master['Helio vy']
                    vz = master['Helio vz']
                    vx_M = master['Moon vx (Helio)']  # AU/day
                    vy_M = master['Moon vy (Helio)']
                    vz_M = master['Moon vz (Helio)']
                    x_M = master['Moon x (Helio)']  # AU
                    y_M = master['Moon y (Helio)']
                    z_M = master['Moon z (Helio)']
                    x_E = master['Earth x (Helio)']
                    y_E = master['Earth y (Helio)']
                    z_E = master['Earth z (Helio)']
                    vx_E = master['Earth vx (Helio)']  # AU/day
                    vy_E = master['Earth vy (Helio)']
                    vz_E = master['Earth vz (Helio)']
                    date_ems = master['Julian Date']  # Julian date

                    date_mjd = Time(date_ems, format='jd').to_value('mjd')
                    h_r_TCO = np.array([x, y, z]).ravel()  # AU
                    h_r_M = np.array([x_M, y_M, z_M]).ravel()
                    h_r_E = np.array([x_E, y_E, z_E]).ravel()
                    h_v_TCO = np.array([vx, vy, vz]).ravel()  # AU/day
                    h_v_M = np.array([vx_M, vy_M, vz_M]).ravel()
                    h_v_E = np.array([vx_E, vy_E, vz_E]).ravel()

                    EMS_rp_TCO, EMS_vp_TCO, EMS_omega_SEMSM, r_ETCO, r_MTCO, r_STCO, EMS_rp_SUN, EMS_rp_M, r_EM, hp_omega_EMSM = (
                        helio_to_earthmoon_corotating(h_r_TCO, h_v_TCO, h_r_E, h_v_E, h_r_M, h_v_M,
                                                      date_mjd))

                    nmx.append(EMS_rp_TCO[0])
                    nmy.append(EMS_rp_TCO[1])
                    nmz.append(EMS_rp_TCO[2])
                    nmvx.append(EMS_vp_TCO[0])
                    nmvy.append(EMS_vp_TCO[1])
                    nmvz.append(EMS_vp_TCO[2])
                    nmomx.append(EMS_omega_SEMSM[0])
                    nmomy.append(EMS_omega_SEMSM[1])
                    nmomz.append(EMS_omega_SEMSM[2])
                    nmetco.append(r_ETCO)
                    nmmtco.append(r_MTCO)
                    nmstco.append(r_STCO)
                    nmsunx.append(EMS_rp_SUN[0])
                    nmsuny.append(EMS_rp_SUN[1])
                    nmsunz.append(EMS_rp_SUN[2])
                    nmmoonx.append(EMS_rp_M[0])
                    nmmoony.append(EMS_rp_M[1])
                    mnmoonz.append(EMS_rp_M[2])
                    mnemd.append(r_EM)
                    emomx.append(hp_omega_EMSM[0])
                    emomy.append(hp_omega_EMSM[1])
                    emomz.append(hp_omega_EMSM[2])

                no_moon_data['Earth-Moon Synodic x'] = nmx
                no_moon_data['Earth-Moon Synodic y'] = nmy
                no_moon_data['Earth-Moon Synodic z'] = nmz
                no_moon_data['Earth-Moon Synodic vx'] = nmvx
                no_moon_data['Earth-Moon Synodic vy'] = nmvy
                no_moon_data['Earth-Moon Synodic vx'] = nmvz
                no_moon_data['Earth-Moon Synodic Omega x'] = nmomx
                no_moon_data['Earth-Moon Synodic Omega y'] = nmomy
                no_moon_data['Earth-Moon Synodic Omega z'] = nmomz
                no_moon_data['Earth-TCO Distance'] = nmetco
                no_moon_data['Moon-TCO Distance'] = nmmtco
                no_moon_data['Sun-TCO Distance'] = nmstco
                no_moon_data['Earth-Moon Synodic Sun x'] = nmsunx
                no_moon_data['Earth-Moon Synodic Sun y'] = nmsuny
                no_moon_data['Earth-Moon Synodic Sun z'] = nmsunz
                no_moon_data['Earth-Moon Synodic Moon x'] = nmmoonx
                no_moon_data['Earth-Moon Synodic Moon y'] = nmmoony
                no_moon_data['Earth-Moon Synodic Moon z'] = mnmoonz
                no_moon_data['Earth-Moon Distance'] = mnemd
                no_moon_data['Moon around EMS Omega x'] = emomx
                no_moon_data['Moon around EMS Omega y'] = emomy
                no_moon_data['Moon around EMS Omega z'] = emomz

                no_moon_data.to_csv('no_moon_files/' + id + '.csv', sep=' ', header=True, index=False)

            data_full = mm_parser.mm_file_parse_new(file_path)
            start_offset = 0
            end_offset = 3000
            start_moon = int(
                master_file.loc[master_file['Object id'] == id, "Entry to EMS Index"].iloc[0] - start_offset)
            # end = int(master_file.loc[master_file['Object id'] == id, "Entry to EMS Index"].iloc[0] + end_offset)
            end_moon = int(master_file.loc[master_file['Object id'] == id, "Exit Index to EMS"].iloc[0])
            # start = 0
            # end = -1
            if ((not np.isnan(no_moon_master.loc[no_moon_master['Object id'] == id, "EMS Start Index"].iloc[0])) and
                    (not np.isnan(no_moon_master.loc[no_moon_master['Object id'] == id, "EMS End Index"].iloc[0]))):
                start_nomoon = int(
                    no_moon_master.loc[no_moon_master['Object id'] == id, "EMS Start Index"].iloc[0] - start_offset)
                # end = int(master_file.loc[master_file['Object id'] == id, "Entry to EMS Index"].iloc[0] + end_offset)
                end_nomoon = int(no_moon_master.loc[no_moon_master['Object id'] == id, "EMS End Index"].iloc[0])

            else:
                start_nomoon = 0
                end_nomoon = -1

            # start = 0
            # end = -1

            start_moon = int(
                master_file.loc[master_file['Object id'] == id, "Entry to EMS Index"].iloc[0] - start_offset)
            # end = int(master_file.loc[master_file['Object id'] == id, "Entry to EMS Index"].iloc[0] + end_offset)
            end_moon = int(master_file.loc[master_file['Object id'] == id, "Exit Index to EMS"].iloc[0])
            # data = data_full.iloc[start_moon:end_moon]
            # data_nomoon = no_moon_data.iloc[start_moon:end_moon]
            # start_moon = 0
            data = data_full.iloc[start_moon:end_moon]
            data_nomoon = no_moon_data.iloc[start_moon:end_moon]

            vels_i = []
            poss_i = []
            poss_m_i = []
            hamiltonians = []
            sun_pos = []
            omegas = []
            j_ems_d = []
            j_ems_nd = []
            first_time = True
            first_exit = 10
            for j, master in data.iterrows():
                x = master['Helio x']
                y = master['Helio y']
                z = master['Helio z']
                vx = master['Helio vx']  # AU/day
                vy = master['Helio vy']
                vz = master['Helio vz']
                vx_M = master['Moon vx (Helio)']  # AU/day
                vy_M = master['Moon vy (Helio)']
                vz_M = master['Moon vz (Helio)']
                x_M = master['Moon x (Helio)']  # AU
                y_M = master['Moon y (Helio)']
                z_M = master['Moon z (Helio)']
                x_E = master['Earth x (Helio)']
                y_E = master['Earth y (Helio)']
                z_E = master['Earth z (Helio)']
                vx_E = master['Earth vx (Helio)']  # AU/day
                vy_E = master['Earth vy (Helio)']
                vz_E = master['Earth vz (Helio)']
                date_ems = master['Julian Date']  # Julian date

                date_mjd = Time(date_ems, format='jd').to_value('mjd')
                h_r_TCO = np.array([x, y, z]).ravel()  # AU
                h_r_M = np.array([x_M, y_M, z_M]).ravel()
                h_r_E = np.array([x_E, y_E, z_E]).ravel()
                h_v_TCO = np.array([vx, vy, vz]).ravel()  # AU/day
                h_v_M = np.array([vx_M, vy_M, vz_M]).ravel()
                h_v_E = np.array([vx_E, vy_E, vz_E]).ravel()

                EMS_rp_TCO, EMS_vp_TCO, EMS_omega_SEMSM, r_ETCO, r_MTCO, r_STCO, EMS_rp_SUN, EMS_rp_M, r_EM, hp_omega_EMSM = (
                    helio_to_earthmoon_corotating(h_r_TCO, h_v_TCO, h_r_E, h_v_E, h_r_M, h_v_M,
                                                  date_mjd))

                # hamiltonian = pseudo_potential(EMS_rp_TCO, EMS_vp_TCO, hp_omega_EMSM, r_STCO, r_ETCO, r_MTCO, EMS_rp_SUN, r_EM)
                # r_TCO = np.linalg.norm(EMS_rp_TCO)
                # if r_TCO < 0.0038752837677:
                #     jacobi_em_dim, jacobi_em_non_dim = jacobi_earth_moon(EMS_rp_TCO, EMS_vp_TCO, r_ETCO, r_MTCO, hp_omega_EMSM, r_EM)
                # else:
                #     C_r_TCO, C_v_TCO, C_v_TCO_2, C_ems, C_moon, ems_barycentre, vems_barycentre, omega, omega_2, r_sE, mu_s, mu_EMS = get_r_and_v_cr3bp_from_nbody_sun_emb(h_r_TCO, h_v_TCO, h_r_E, h_v_E, h_r_M, h_v_M, date_mjd)
                #     mu = mu_EMS / (mu_EMS + mu_s)
                #     jacobi_em_dim, jacobi_em_non_dim = jacobi_dim_and_non_dim(C_r_TCO, C_v_TCO, h_r_TCO, ems_barycentre, mu, mu_s, mu_EMS, omega_2, r_sE)
                #
                # if r_TCO < 0.0038752837677 and first_time is True:
                #     pass
                # elif r_TCO > 0.0038752837677 and first_time is True:
                #     first_exit = j - start_moon
                #     first_time = False
                #
                # else:
                #     pass

                poss_i.append(EMS_rp_TCO)
                sun_pos.append(r_STCO)
                poss_m_i.append(EMS_rp_M)
                vels_i.append(EMS_vp_TCO)
                # hamiltonians.append(hamiltonian)
                omegas.append(EMS_omega_SEMSM)
                # j_ems_d.append(jacobi_em_dim)
                # j_ems_nd.append(jacobi_em_non_dim)

            poss_i = np.array(poss_i)
            poss_m_i = np.array(poss_m_i)
            vels_i = np.array(vels_i)
            # fig3 = plt.figure()
            # ax = fig3.add_subplot(111, projection='3d')
            # ax.plot3D(poss_i[0, :], poss_i[1, :], poss_i[2, :])
            # ax.plot3D(poss_m_i[0, :], poss_m_i[1, :], poss_m_i[2, :])

            fig = plt.figure()
            end_offset = end_offset
            # sc = plt.scatter(poss_i[:first_exit, 0], poss_i[:first_exit, 1], c=j_ems_d[:first_exit], cmap='coolwarm', s=15, zorder=20)
            plt.plot(poss_i[:end_offset, 0], poss_i[:end_offset, 1], color='blue', zorder=18)
            plt.plot(data_nomoon['Earth-Moon Synodic x'].iloc[:end_offset],
                     data_nomoon['Earth-Moon Synodic y'].iloc[:end_offset], linewidth=1, zorder=25, color='orange')
            # plt.plot(poss_i[:, 0], poss_i[:, 1], label='Trajectory', color='grey', zorder=20)
            plt.scatter(poss_m_i[:, 0], poss_m_i[:, 1], s=1, label='Moon', color='red', zorder=15)
            plt.scatter(0, 0, s=20, label='Earth', color='Blue', zorder=10)
            plt.scatter(poss_i[0 + start_offset, 0], poss_i[0 + start_offset, 1], s=30, label='Entry', color='green',
                        zorder=25)
            # plt.scatter(poss_i[end_offset, 0], poss_i[end_offset, 1], s=30, label='Exit', color='red', zorder=30)
            c1 = plt.Circle((0, 0), radius=0.0038752837677, alpha=0.1, label='SOI of the EMS', zorder=5)
            plt.gca().add_artist(c1)
            c2 = plt.Circle((np.mean(poss_m_i[:, 0]), np.mean(poss_m_i[:, 1])), radius=0.00044118275, alpha=0.3,
                            label='SOI of the Moon', zorder=8, color='black')
            plt.gca().add_artist(c2)
            plt.gca().set_aspect('equal')
            # cbar = fig.colorbar(sc, label='Dimensional Jacobi')
            plt.legend()
            plt.xlabel('x (au)')
            plt.ylabel('y (au)')
            plt.title(id)

            fig = plt.figure()
            ax3 = fig.add_subplot()
            ax3.plot(np.linspace(0, len(poss_i), len(poss_i)), np.linalg.norm(poss_i, axis=1), label='EMS-STC Distance')
            ax3.plot(np.linspace(0, len(data_nomoon['Earth-Moon Synodic x']),
                                 len(data_nomoon['Earth-Moon Synodic x'])),
                     np.linalg.norm(np.array([data_nomoon['Earth-Moon Synodic x'],
                                              data_nomoon['Earth-Moon Synodic y'],
                                              data_nomoon['Earth-Moon Synodic z']]).T, axis=1),
                     label='EMS-STC Distance', color='red')
            # ax4 = ax3.twinx()
            # ax4.plot(np.linspace(0, len(poss_i), len(poss_i)), j_ems_d, color='tab:purple', label='Hamiltonian')
            # ax4.set_ylabel('Jacobi', color='tab:purple')
            ax3.set_ylabel('STC-EMS Distance (AU)')
            ax3.set_xlabel('Time (h)')
            plt.show()
            """
            fig = plt.figure()
            ax10 = fig.add_subplot()
            master_i = master_file[master_file['Object id'] == id]
            ax10.plot(np.linspace(0, len(j_ems_d), len(j_ems_d)),
                      master_i['Dimensional Jacobi'].iloc[0] * np.ones((len(j_ems_d),)),
                      label='Sun-EMS Jacobi at SOI of EMS', linestyle='--', color='blue')
            ax10.plot(np.linspace(0, len(j_ems_d), len(j_ems_d)), j_ems_d, label='Earth-Moon Jacobi',
                      color='blue')
            ax11 = ax10.twinx()
            ax11.plot(np.linspace(0, len(j_ems_d), len(j_ems_d)),
                      master_i['Non-Dimensional Jacobi'].iloc[0] * np.ones((len(j_ems_d),)),
                      label='Sun-EMS Jacobi (ND) at SOI of EMS', linestyle='--', color='tab:red')
            ax11.plot(np.linspace(0, len(j_ems_d), len(j_ems_d)), j_ems_nd, label='Earth-Moon Jacobi (ND)',
                      color='tab:red')
            ax10.set_xlabel('Time (h)')
            ax10.set_ylabel('Dimensional Jacobi Constant')
            ax11.set_ylabel('Non-Dimensional Jacobi Constant')

            fig = plt.figure()
            sc = plt.scatter(poss_i[:, 0], poss_i[:, 1], c=hamiltonians, cmap='gist_rainbow', s=15, zorder=20)
            # plt.plot(poss_i[:, 0], poss_i[:, 1], label='Trajectory', color='grey', zorder=20)
            plt.scatter(poss_m_i[:, 0], poss_m_i[:, 1], s=1, label='Moon', color='red', zorder=15)
            plt.scatter(0, 0, s=20, label='Earth', color='Blue', zorder=10)
            plt.scatter(poss_i[0 + start_offset, 0], poss_i[0 + start_offset, 1], s=30, label='Entry', color='green', zorder=25)
            plt.scatter(poss_i[-1, 0], poss_i[-1, 1], s=30, label='Exit', color='red', zorder=30)
            c1 = plt.Circle((0, 0), radius=0.0038752837677, alpha=0.1, label='SOI of the EMS', zorder=5)
            plt.gca().add_artist(c1)
            c2 = plt.Circle((np.mean(poss_m_i[:, 0]), np.mean(poss_m_i[:, 1])), radius=0.00044118275, alpha=0.3, label='SOI of the Moon', zorder=8, color='black')
            plt.gca().add_artist(c2)
            plt.gca().set_aspect('equal')
            cbar = fig.colorbar(sc, label='Hamiltonian')
            plt.legend()
            plt.xlabel('x (au)')
            plt.ylabel('y (au)')
            plt.title(id)
            fig = plt.figure()
            ax1 = fig.add_subplot()
            ax1.plot(np.linspace(0, len(poss_i), len(poss_i)), np.linalg.norm(poss_i, axis=1), label='Earth-STC Distance')
            ax2 = ax1.twinx()
            ax2.plot(np.linspace(0, len(poss_i), len(poss_i)), hamiltonians, color='tab:purple', label='Hamiltonian')
            ax2.set_ylabel('Hamiltonian ($AU^2/d^2$)', color='tab:purple')
            ax1.set_ylabel('Earth-STC Distance (AU)')
            ax1.set_xlabel('Time (h)')

            fig = plt.figure()
            ax3 = fig.add_subplot()
            ax3.plot(np.linspace(0, len(poss_i), len(poss_i)), np.linalg.norm(poss_i - poss_m_i, axis=1),  label='Moon-STC Distance')
            ax4 = ax3.twinx()
            ax4.plot(np.linspace(0, len(poss_i), len(poss_i)), hamiltonians, color='tab:purple', label='Hamiltonian')
            ax4.set_ylabel('Hamiltonian ($AU^2/d^2$)', color='tab:purple')
            ax3.set_ylabel('Moon-STC Distance (AU)')
            ax3.set_xlabel('Time (h)')

            fig = plt.figure()
            ax5 = fig.add_subplot()
            ax5.plot(np.linspace(0, len(sun_pos), len(sun_pos)), sun_pos, label='Sun-STC Distance')
            ax6 = ax5.twinx()
            ax6.plot(np.linspace(0, len(poss_i), len(poss_i)), hamiltonians, color='tab:purple', label='Hamiltonian')
            ax6.set_ylabel('Hamiltonian ($AU^2/d^2$)', color='tab:purple')
            ax5.set_ylabel('Sun-STC Distance (AU)')
            ax5.set_xlabel('Time (h)')

            fig = plt.figure()
            ax7 = fig.add_subplot()
            ax7.plot(np.linspace(0, len(omegas), len(omegas)), np.linalg.norm(omegas, axis=1), label='Sun-STC Distance')
            ax8 = ax7.twinx()
            ax8.plot(np.linspace(0, len(omegas), len(omegas)), hamiltonians, color='tab:purple', label='Hamiltonian')
            ax8.set_ylabel('Hamiltonian ($AU^2/d^2$)', color='tab:purple')
            ax7.set_ylabel('Sun-STC Distance (AU)')
            ax7.set_xlabel('Time (h)')
            
            plt.show()
            """
            no_moon_pos = np.array([no_moon_data['Earth-Moon Synodic x'], no_moon_data['Earth-Moon Synodic y'],
                                    no_moon_data['Earth-Moon Synodic z']])
            distance = np.linalg.norm(no_moon_pos.T, axis=1)

            moon_pos = np.array([poss_i[:, 0], poss_i[:, 1], poss_i[:, 2]])
            moon_distance = np.linalg.norm(moon_pos.T, axis=1)

            # fig = plt.figure()
            # plt.plot(np.linspace(0, len(distance), len(distance)), distance)
            # plt.show()

            # Define the threshold
            threshold = 0.0038752837677

            # Find local minima
            moon_minima_indices = argrelextrema(moon_distance, np.less, order=1)

            # Filter minima below the threshold
            moon_minima_below_threshold = [index for index in moon_minima_indices[0] if
                                           moon_distance[index] < threshold]

            # first peri distance
            first_peri = moon_distance[moon_minima_below_threshold[0]]

            # time between peri
            dt = moon_minima_below_threshold[1] - moon_minima_below_threshold[0]

            # Find local minima
            minima_indices = argrelextrema(distance, np.less, order=1)

            # Filter minima below the threshold
            minima_below_threshold = [index for index in minima_indices[0] if distance[index] < threshold]

            # Count the number of such minima
            num_minima_below_threshold = len(minima_below_threshold)

            if num_minima_below_threshold >= 2:
                still_stc = True
            else:
                still_stc = False
            important_params.append(
                [id, poss_i[0, 0], poss_i[0, 1], poss_i[0, 2], poss_m_i[0, 0], poss_m_i[0, 1], poss_m_i[0, 2],
                 vels_i[0, 0], vels_i[0, 1], vels_i[0, 2], still_stc])

            peris.append(first_peri)
            dts.append(dt)

        # data_stc['First Perigee Distance'] = peris
        # data_stc['Time Between Perigees'] = dts
        # data_stc.to_csv('stc_nonstc2.csv', sep=' ', header=True, index=False)
        # df = pd.DataFrame(important_params, columns=['Object id', 'EM Syn. at SOIEMS x', 'EM Syn. at SOIEMS y', 'EM Syn. at SOIEMS z',
        #                                              'Moon at SOIEMS x', 'Moon at SOIEMS y', 'Moon at SOIEMS z',
        #                                              'EM Syn. at SOIEMS vx', 'EM Syn. at SOIEMS vy',
        #                                              'EM Syn. at SOIEMS vz', 'STC without Moon'])

        # df = pd.DataFrame({'Object id': id, 'EM Syn. at SOIEMS x': poss_i[0, 0],
        #                    'EM Syn. at SOIEMS y': poss_i[0, 1], 'EM Syn. at SOIEMS z': poss_i[0, 2],
        #                    'Moon at SOIEMS x': poss_m_i[0, 0], 'Moon at SOIEMS y': poss_m_i[0, 1],
        #                    'Moon at SOIEMS z': poss_m_i[0, 2], 'EM Syn. at SOIEMS vx': vels_i[0, 0],
        #                    'EM Syn. at SOIEMS vy': vels_i[0, 1], 'EM Syn. at SOIEMS vz': vels_i[0, 2],
        #                    'STC without Moon': still_stc}, index=[1])
        # df.to_csv('stc_nonstc.csv', sep=" ", mode='a', header=False, index=False)

    @staticmethod
    def severity_check():

        destination_path = os.path.join(os.getcwd(), 'minimoon_files_oorb')
        destination_file = destination_path + '/minimoon_master_final.csv'
        mm_parser = MmParser("", "", "")
        master = mm_parser.parse_master(destination_file)

        # three hill discrepency
        three_hill_severity = len(master[master['Max. Distance'] > 0.03])
        print('The three hill severity is: {}'.format(three_hill_severity))

        # energy discrepency
        energy_defaults = 0

        rev_defaults = 0

        # retrograde and prograde discrepency
        retro_pro_defaults = 0

        for idx, row in master.iterrows():
            # get file
            obj_id = row['Object id']

            # obj_id = 'NESC000001td'
            file_path = destination_path + '/' + obj_id + '.csv'
            print(obj_id)
            data = mm_parser.mm_file_parse_new(file_path)
            capture_data = data.iloc[row['Capture Index']:row['Release Index']]

            # pro retro discrepency
            eclip_long_diff = pd.Series(capture_data['Eclip Long']).diff().dropna()
            positive = eclip_long_diff[eclip_long_diff > 0]
            positive_filtered = positive[positive < 1]
            negative = eclip_long_diff[eclip_long_diff < 0]
            negative_filtered = negative[negative > -1]
            if len(positive_filtered) > 0 and len(negative_filtered) > 0:
                retro_pro_defaults += 1

            # energy discrepency
            vx = capture_data['Geo vx']
            vy = capture_data['Geo vy']
            vz = capture_data['Geo vz']
            v = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2) * 1730840
            r = capture_data['Distance'] * 149597870700
            mu = const.GM_earth.value  # Nominal Earth mass parameter (m3/s2)
            energy = v ** 2 / 2 - mu / r
            filtered_energy = energy[abs(energy / 1000 / 1000) < 0.7]
            filtered_energy_diff = pd.Series(energy).diff().dropna()
            if np.any(filtered_energy > 0) and not np.any(
                    filtered_energy_diff[abs(filtered_energy_diff) / 1000 / 1000 > 0.005]):
                energy_defaults += 1

            # revolution discrepencies
            xz_angle = np.rad2deg(np.arctan2(capture_data['Synodic z'], capture_data['Synodic x']))
            xz_angle_corrected = pd.Series(xz_angle).apply(lambda x: x + 360 if x < 0 else x)
            xz_angle_corrected_diff = pd.Series(xz_angle_corrected).diff().dropna()
            xz_eclip_long = xz_angle_corrected_diff.apply(lambda x: x - 360 if ((abs(x) > 200) and (x > 0)) else x)
            xz_eclip_long_corrected = pd.Series(xz_eclip_long).apply(
                lambda x: x + 360 if (((abs(x) > 200)) and (x < 0)) else x)

            yz_angle = np.rad2deg(np.arctan2(capture_data['Synodic y'], capture_data['Synodic z']))
            yz_angle_corrected = pd.Series(yz_angle).apply((lambda x: x + 360 if x < 0 else x))
            yz_angle_corrected_diff = pd.Series(yz_angle_corrected).diff().dropna()
            yz_eclip_long = yz_angle_corrected_diff.apply(lambda x: x - 360 if ((abs(x) > 200) and (x > 0)) else x)
            yz_eclip_long_corrected = pd.Series(yz_eclip_long).apply(
                lambda x: x + 360 if (((abs(x) > 200)) and (x < 0)) else x)

            rev_xz = abs(sum(xz_eclip_long_corrected)) / 360
            rev_yz = abs(sum(yz_eclip_long_corrected)) / 360

            if row['Number of Rev'] >= 1 or rev_yz >= 1 or rev_xz >= 1:
                if row['Number of Rev'] >= 1 and rev_yz >= 1 and rev_xz >= 1:
                    pass
                else:
                    rev_defaults += 1

            print('The three hill severity is: {}'.format(three_hill_severity))
            print("Number of Prograde/Retrogade motions: {}".format(retro_pro_defaults))
            print("Number of Energy greater than zero: {}".format(energy_defaults))
            print("Number of Revolutions in other frames: {}".format(rev_defaults))

        return

    @staticmethod
    def chyba_criteria():

        destination_path = os.path.join(os.getcwd(), 'minimoon_files_oorb')
        destination_file = destination_path + '/minimoon_master_final.csv'
        mm_parser = MmParser("", "", "")
        master = mm_parser.parse_master(destination_file)

        min_energys = []
        avg_zs = []
        avg_vzs = []
        min_l2tcas = []
        winding_diffs = []

        for idx, row in master.iterrows():
            # get file
            obj_id = row['Object id']

            file_path = destination_path + '/' + obj_id + '.csv'
            print(idx)
            print(obj_id)
            data = mm_parser.mm_file_parse_new(file_path)
            capture_data = data.iloc[row['Capture Index']:row['Release Index']]

            # minimum energy
            vx = capture_data['Geo vx']
            vy = capture_data['Geo vy']
            vz = capture_data['Geo vz']
            v = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2) * 1730840
            r = capture_data['Distance'] * 149597870700
            mu = const.GM_earth.value  # Nominal Earth mass parameter (m3/s2)
            energy = v ** 2 / 2 - mu / r
            filtered_energy = energy[abs(energy / 1000 / 1000) < 0.7]
            min_energy = min(filtered_energy / 1000 / 1000)  # km^2 / s^2
            min_energys.append(min_energy)

            # Average vz and z
            avg_z = np.mean(capture_data['Geo z'])
            avg_vz = np.mean(capture_data['Geo vz'])
            avg_zs.append(avg_z)
            avg_vzs.append(avg_vz)


            # peri-L2 distance
            moon_tca_pos = np.array(
                [capture_data['Helio x'], capture_data['Helio y'], capture_data['Helio z']]) - np.array(
                [capture_data['Moon x (Helio)'], capture_data['Moon y (Helio)'], capture_data['Moon z (Helio)']])

            moon_l2_dist = 0.00043249279
            earth_moon_pos = np.array(
                [capture_data['Earth x (Helio)'], capture_data['Earth y (Helio)'], capture_data['Earth z (Helio)']]) - np.array(
                [capture_data['Moon x (Helio)'], capture_data['Moon y (Helio)'], capture_data['Moon z (Helio)']])
            em_pos_unit = earth_moon_pos / np.linalg.norm(earth_moon_pos, axis=0)
            moon_l2_pos = moon_l2_dist * em_pos_unit
            l2_tca_pos = moon_tca_pos - moon_l2_pos
            l2_tca_dist = np.linalg.norm(l2_tca_pos, axis=0)
            min_l2 = min(l2_tca_dist)
            min_l2tcas.append(min_l2)

            # winding numbers

            # transform whole traj to earth-moon corotating
            h_r_TCA = np.array([capture_data['Helio x'], capture_data['Helio y'], capture_data['Helio z']])
            h_v_TCA = np.array([capture_data['Helio vx'], capture_data['Helio vy'], capture_data['Helio vz']])
            h_r_E = np.array(
                [capture_data['Earth x (Helio)'], capture_data['Earth y (Helio)'], capture_data['Earth z (Helio)']])
            h_v_E = np.array(
                [capture_data['Earth vx (Helio)'], capture_data['Earth vy (Helio)'], capture_data['Earth vz (Helio)']])
            h_r_M = np.array(
                [capture_data['Moon x (Helio)'], capture_data['Moon y (Helio)'], capture_data['Moon z (Helio)']])
            h_v_M = np.array(
                [capture_data['Moon vx (Helio)'], capture_data['Moon vy (Helio)'], capture_data['Moon vz (Helio)']])

            pos_TCA, pos_Moon, pos_Earth = helio_to_earthmoon_corotating_vec(h_r_TCA, h_v_TCA, h_r_E, h_v_E, h_r_M, h_v_M,
                                                        capture_data['Julian Date'])



            # get earth tca eclip long
            earth_xy_angle = np.rad2deg(np.arctan2(pos_TCA[:, 1] - pos_Earth[:, 1], pos_TCA[:, 0] - pos_Earth[:, 0]))
            earth_xy_angle_corrected = pd.Series(earth_xy_angle).apply(lambda x: x + 360 if x < 0 else x)
            earth_xy_angle_corrected_diff = pd.Series(earth_xy_angle_corrected).diff().dropna()
            earth_xy_eclip_long = earth_xy_angle_corrected_diff.apply(lambda x: x - 360 if ((abs(x) > 200) and (x > 0)) else x)
            earth_xy_eclip_long_corrected = pd.Series(earth_xy_eclip_long).apply(
                lambda x: x + 360 if (((abs(x) > 200)) and (x < 0)) else x)


            # get moon tca eclip long
            moon_xy_angle = np.rad2deg(np.arctan2(pos_TCA[:, 1] - pos_Moon[:, 1], pos_TCA[:, 0] - pos_Moon[:, 0]))
            moon_xy_angle_corrected = pd.Series(moon_xy_angle).apply(lambda x: x + 360 if x < 0 else x)
            moon_xy_angle_corrected_diff = pd.Series(moon_xy_angle_corrected).diff().dropna()
            moon_xy_eclip_long = moon_xy_angle_corrected_diff.apply(
                lambda x: x - 360 if ((abs(x) > 200) and (x > 0)) else x)
            moon_xy_eclip_long_corrected = pd.Series(moon_xy_eclip_long).apply(
                lambda x: x + 360 if (((abs(x) > 200)) and (x < 0)) else x)

            # get rev around earth
            earth_rev_xy = abs(sum(earth_xy_eclip_long_corrected)) / 360

            # get rev around moon
            moon_rev_xy = abs(sum(moon_xy_eclip_long_corrected)) / 360

            # get winding number
            winding_diffs.append(abs(earth_rev_xy - moon_rev_xy))


        master['Minimum Energy'] = min_energys
        master['Peri-EM-L2'] = min_l2tcas
        master['Average Geo z'] = avg_zs
        master['Average Geo vz'] = avg_vzs
        master['Winding Difference'] = winding_diffs
        master.to_csv('minimoon_master_new.csv', sep=' ', header=True, index=False)


    @staticmethod
    def case_study_hists():

        destination_path = os.path.join(os.getcwd(), 'minimoon_files_oorb')
        destination_file = destination_path + '/minimoon_master_new.csv'
        mm_parser = MmParser("", "", "")
        master = mm_parser.parse_master_new_new(destination_file)

        """
        #min energy
        bin_size_en = 0.01

        # Calculate the number of bins based on data range and bin size
        data_range_en = max(master['Minimum Energy']) - min(master['Minimum Energy'])
        num_bins_en = int(data_range_en / bin_size_en)

        fig = plt.figure()
        plt.hist(master['Minimum Energy'], bins=num_bins_en, edgecolor="#448dc0", color="#1f77b4")
        plt.xlim([-0.7, 0])
        plt.ylim([0, 2300])
        plt.xlabel('Minimum Planetocentric Energy During Capture $(km^2/s^2)$')
        plt.ylabel('Count')

        # Peri-EM-L2
        bin_size_l2 = 0.0001

        # Calculate the number of bins based on data range and bin size
        data_range_l2 = max(master['Peri-EM-L2']) - min(master['Peri-EM-L2'])
        num_bins_l2 = int(data_range_l2 / bin_size_l2)

        fig = plt.figure()
        plt.hist(master['Peri-EM-L2'], bins=num_bins_l2, edgecolor="#448dc0", color="#1f77b4")
        plt.xlim([0, 0.02])
        plt.ylim([0, 900])
        plt.xlabel('Minimum Distance to Earth-Moon $L_2$ (au)')
        plt.ylabel('Count')


        # Average Geo z
        bin_size_z = 0.0001

        # Calculate the number of bins based on data range and bin size
        data_range_z = max(master['Average Geo z']) - min(master['Average Geo z'])
        num_bins_z = int(data_range_z / bin_size_z)

        fig = plt.figure()
        plt.hist(master['Average Geo z'], bins=num_bins_z, edgecolor="#448dc0", color="#1f77b4")
        plt.xlim([-0.015, 0.015])
        plt.ylim([0, 800])
        plt.xlabel('Average Geocentric $z$ During Capture (au)')
        plt.ylabel('Count')

        # Average Geo z
        bin_size_vz = 0.00001

        # Calculate the number of bins based on data range and bin size
        data_range_vz = max(master['Average Geo vz']) - min(master['Average Geo vz'])
        num_bins_vz = int(data_range_vz / bin_size_vz)

        fig = plt.figure()
        plt.hist(master['Average Geo vz'], bins=num_bins_vz, edgecolor="#448dc0", color="#1f77b4")
        plt.xlim([-0.0004, 0.0004])
        plt.ylim([0, 5500])
        plt.xlabel('Average Geocentric $v_z$ During Capture (au/day)')
        plt.ylabel('Count')

        # Winding number
        bin_size_wn = 0.05

        # Calculate the number of bins based on data range and bin size
        data_range_wn = max(master['Winding Difference']) - min(master['Winding Difference'])
        num_bins_wn = int(data_range_wn / bin_size_wn)

        fig = plt.figure()
        plt.hist(master['Winding Difference'], bins=num_bins_wn, edgecolor="#448dc0", color="#1f77b4")
        plt.xlim([0, 6.5])
        plt.ylim([0, 8000])
        plt.xlabel('|Rev. Around Moon - Rev. Around Earth|')
        plt.ylabel('Count')
        # plt.show()
        """

        # Taxonomy design
        metrics = ['Capture Duration', 'Minimum Energy', 'Peri-EM-L2', 'Average Geo z', 'Average Geo vz', 'Winding Difference']
        # metrics = ['Capture Duration', 'Minimum Energy', 'Peri-EM-L2']
        n = len(metrics)
        thresholds = [300, -0.3, 0.001, 0.001, 0.00005, 1]
        positives = ['greater', 'less', 'less', 'abs. less', 'abs. less', 'less']
        letters = list(string.ascii_uppercase)[:n]
        n_letters = ['n' + letter for idx, letter in enumerate(letters)]

        # Generate all permutations
        classes = list(product([0, 1], repeat=len(n_letters)))

        # Map permutations to letters
        letter_classes = [[n_letters[i] if val == 1 else letters[i] for i, val in enumerate(p)] for p in classes]

        pops = []
        for metric, threshold, positive in zip(metrics, thresholds, positives):

            if positive == 'greater':
                pos_pop = master[master[metric] >= threshold]
                neg_pop = master[master[metric] < threshold]
                pos_percent = len(pos_pop[metric]) / len(master[metric]) * 100
                print('There is {0}% of the population with {1} greater than {2}, {3}% with less.'.format(pos_percent, metric, threshold, 100 - pos_percent))
            elif positive == 'less':
                pos_pop = master[master[metric] <= threshold]
                neg_pop = master[master[metric] > threshold]
                pos_percent = len(pos_pop[metric]) / len(master[metric]) * 100
                print('There is {0}% of the population with {1} less than {2}, {3}% with greater.'.format(pos_percent, metric, threshold, 100 - pos_percent))
            else:
                pos_pop = master[abs(master[metric]) <= threshold]
                neg_pop = master[abs(master[metric]) > threshold]
                pos_percent = len(pos_pop[metric]) / len(master[metric]) * 100
                print('There is {0}% of the population with {1} less than {2}, {3}% with greater.'.format(pos_percent,
                                                                                                          metric,
                                                                                                          threshold,
                                                                                                          100 - pos_percent))

            pops.append([pos_pop, neg_pop])

        classed_pop = []
        percents = []
        for classe, letter_class in zip(classes, letter_classes):

            pop = master.copy()
            for metric, classi, positive, threshold in zip(metrics, classe, positives, thresholds):

                if positive == 'greater':
                    if classi == 0:  # talking about A not about nA
                        pop = pop[pop[metric] >= threshold]
                    else:
                        pop = pop[pop[metric] < threshold]
                elif positive == 'less':
                    if classi == 0:  # talking about A not about nA
                        pop = pop[pop[metric] <= threshold]
                    else:
                        pop = pop[pop[metric] > threshold]
                else:
                    if classi == 0:  # talking about A not about nA
                        pop = pop[abs(pop[metric]) <= threshold]
                    else:
                        pop = pop[abs(pop[metric]) > threshold]

            pop_percent = len(pop) / len(master) * 100
            percents.append(pop_percent)
            print("{0}% of the population belongs to class: {1}\n".format(pop_percent, letter_class))
            classed_pop.append(pop)

        print(sum(percents))



if __name__ == '__main__':

    mm_main = MmMain()

    destination_path = '/media/aeromec/Seagate Desktop Drive/minimoon_files_oorb'
    destination_file = destination_path + '/minimoon_master_new.csv'
    start_file = destination_path + '/minimoon_master_new (copy).csv'

    ########################################
    # Integrate Initializations
    #########################################

    if False:
        # Constants
        mu_e = const.GM_earth.value  # Nominal Earth mass parameter (m3/s2)

        # Amount before and after you want oorb integrations to start (in days) with respect to Fedorets data
        leadtime = 365

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
        moon = 0
        perturbers = [mercury, venus, earth, mars, jupiter, saturn, uranus, neptune, pluto, moon]

        int_step = 1 / 24

        # mm_main.integrate(destination_file, mu_e, leadtime, perturbers, int_step)

    #########################################
    # integrating data in parallel - check all functions for initializations within
    #########################################

    # create parser
    # mm_parser = MmParser("", "", "")

    # get the master file - you need a list of initial orbits to integrate with openorb (pyorb)
    # master = mm_parser.parse_master(destination_file)

    # get the object ids of all minimoons
    # object_ids = master['Object id'].iloc[:100]

    # pool = multiprocessing.Pool()
    # pool.starmap(mm_main.integrate_parallel, enumerate(object_ids))  # input your function
    # pool.close()

    ##########################################
    # adding a new row to the master file
    ##########################################

    # mm_main.add_new_row(data)

    ########################################
    # changing an existing column in the master file
    ########################################

    # mm_main.change_existing_column(destination_file, destination_path,  mu_e)

    #######################################
    # adding a new column
    ######################################

    mm_main.add_new_column(start_file, destination_file)

    ########################################
    # clustering graphs
    ########################################

    # mm_main.cluster_viz_main(destination_file)

    ########################################
    # STC population visualization
    #######################################

    # mm_main.stc_viz_main(destination_file)

    ######################################
    # Investigate the distribution of alpha beta jacobi
    ##############################################

    # mm_main.alphabetastc(destination_file)

    #####################################
    # Investigate the population when there is no moon present
    ########################################

    # no moon data
    # master_path_nomoon = os.path.join('/media', 'aeromec', 'data', 'minimoon_files_oorb_nomoon')
    # master_file_name_nomoon = 'minimoon_master_final.csv'
    # master_file_nomoon =  master_path_nomoon + '/' + master_file_name_nomoon

    # with moon data
    # master_path = os.path.join(os.getcwd(), 'minimoon_files_oorb')
    # master_file = destination_path + '/minimoon_master_final.csv'

    # mm_main.no_moon_pop(master_file_nomoon, master_file)

    ##########################################
    # Investigate the variation of jacobi constants of planar asteroids
    #########################################

    # mm_main.jacobi_variation()

    #########################################
    # Investigate the bicircular restricted four body problem
    #########################################

    # mm_main.BCR4BP()

    #########################################
    # Investigate the severity of discrepencies
    #########################################

    # mm_main.severity_check()

    ########################################
    # Get criteria for Chyba case study
    ########################################

    # mm_main.chyba_criteria()

    #######################################
    # Histograms for Chyba
    #######################################

    # mm_main.case_study_hists()

    #####################################
    # create plots for 2022 NX1
    ####################################3