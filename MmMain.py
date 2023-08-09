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
                      - Time(data["Julian Date"].iloc[0] * cds.d - leadtime, format="jd", scale='utc').to_value('jd'))/int_step

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

        print(idx)
        print(object_id)
        # create an analyzer
        mm_analyzer = MmAnalyzer()

        # Constants
        mu_e = const.GM_earth.value  # Nominal Earth mass parameter (m3/s2)

        # Amount before and after you want oorb integrations to start (in days) with respect to Fedorets data
        leadtime = 365 * cds.d

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
        int_step_rev = round(1/(old_data['Julian Date'].iloc[1] - old_data['Julian Date'].iloc[0]))
        int_step = 1/int_step_rev
        start_time = Time(start_date, format="jd", scale='utc')
        end_time = Time(end_date, format="jd", scale='utc')
        steps = int((end_time.to_value('jd') - start_time.to_value('jd')) / int_step)
        master_i = master[master['Object id'] == object_id]
        new_data = mm_analyzer.get_data_mm_oorb(master_i, old_data, int_step, perturbers, start_time, end_time, mu_e)

        print("Number of integration steps: " + str(steps))
        """
        if steps < 90000:  # JPL Horizons limit

            new_data = mm_analyzer.get_data_mm_oorb(master_i, old_data, int_step, perturbers, start_time, end_time, mu_e)

        else:

            start_time = Time(start_date * cds.d - leadtime, format="jd", scale='utc')
            end_time = Time(end_date * cds.d + leadtime, format="jd", scale='utc')
            steps = int((end_time.to_value('jd') - start_time.to_value('jd'))/int_step)

            if steps < 90000:
                new_data = mm_analyzer.get_data_mm_oorb(master_i, old_data, int_step, perturbers, start_time, end_time, mu_e)

            else:
                int_step = 1
                new_data = mm_analyzer.get_data_mm_oorb(master_i, old_data, int_step, perturbers, start_time, end_time, mu_e)


        """
        destination_path = os.path.join('/media', 'aeromec', 'data', 'minimoon_files_oorb_nomoon')
        new_data.to_csv(destination_path + '/' + str(object_id) + '.csv', sep=' ', header=True, index=False)

        H = master_i['H'].iloc[0]

        # add a new row of data to the master file
        mm_main.add_new_row(new_data, mu_e, H, destination_path)

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

        new_row2.to_csv(destination_path + '/' + 'minimoon_master_final.csv', sep=' ', mode='a', header=False, index=False)

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
        master = mm_parser.parse_master(master_path)

        # parrelelized version of short term capture
        # pool = multiprocessing.Pool()
        # results = pool.map(mm_analyzer.short_term_capture, master['Object id'])  # input your function
        # pool.close()
        for root, dirs, files in os.walk(dest_path):

            for file in files:
                if file == 'minimoon_master_final (copy).csv' or file == 'minimoon_master_final.csv' or\
                        file == 'minimoon_master_final_previous.csv' or file == 'NESCv9reintv1.TCO.withH.kep.des':
                    pass
                else:
                    data = mm_parser.mm_file_parse_new(dest_path + '/' + file)
                    object_id = data['Object id'].iloc[0]
                    res = mm_analyzer.alpha_beta_jacobi(object_id)

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

        # pd.set_option('display.max_rows', None)
        # print(master['Entry Date to EMS'])

        # master['Dimensional Jacobi'] = repacked_results[0]
        # master['Non-Dimensional Jacobi'] = repacked_results[1]
        # master['Alpha_I'] = repacked_results[2]
        # master['Beta_I'] = repacked_results[3]
        # master['Theta_M'] = repacked_results[4]

        # write the master to csv - only if your sure you have the right data, otherwise in will be over written
        # master.to_csv(dest_path, sep=' ', header=True, index=False)

    @staticmethod
    def cluster_viz_main(master_file):

        mm_pop = MmPopulation(master_file)
        mm_pop.cluster_viz()

    @staticmethod
    def stc_viz_main(master_file):

        mm_pop = MmPopulation(master_file)
        mm_pop.stc_pop_viz()

    @staticmethod
    def alphabetastc(master_file):

        mm_parser = MmParser("", "", "")
        master = mm_parser.parse_master(master_file)

        bad_stc = master[(master['STC'] == True) & (master['Beta_I'] < -180)]
        population_dir = os.path.join(os.getcwd(), 'minimoon_files_oorb')

        for index2, row2 in bad_stc.iterrows():

            object_id = row2['Object id']

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
                        x_M = row["Moon x (Helio)"]   # AU
                        y_M = row["Moon y (Helio)"]
                        z_M = row["Moon z (Helio)"]
                        x_E = row["Earth x (Helio)"]
                        y_E = row["Earth y (Helio)"]
                        z_E = row["Earth z (Helio)"]
                        vx_E = row["Earth vx (Helio)"]   # AU/day
                        vy_E = row["Earth vy (Helio)"]
                        vz_E = row["Earth vz (Helio)"]
                        date = row['Julian Date'] # Julian date

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
                            C_v_TCO = C_R_h @ (h_v_TCO - v_C) - np.cross(np.array([0, 0, omega * seconds_in_day]), C_r_TCO)  # AU/day
                            C_v_TCO_2 = C_R_h @ (h_v_TCO - v_C) - np.cross(omega_2, C_r_TCO)  # AU/day

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
                            vxsdim.append(C_v_TCO[0])
                            vysdim.append(C_v_TCO[1])
                            vzsdim.append(C_v_TCO[2])
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
                ax.scatter(xs[in_ems_idxs], ys[in_ems_idxs], zs[in_ems_idxs], color='red', s=10, label='$\omega_{SE}$ at SOI of EMS')
                # ax.scatter(xs[in_ems_idxs + 100], ys[in_ems_idxs + 100], zs[in_ems_idxs + 100], color='orange', s=10)
                ax.plot3D([xs[in_ems_idxs], xs[in_ems_idxs] + vel_scale * vxs[in_ems_idxs]], [ys[in_ems_idxs], ys[in_ems_idxs] + vel_scale * vys[in_ems_idxs]],
                          [zs[in_ems_idxs], zs[in_ems_idxs] + vel_scale * vzs[in_ems_idxs]], color='red', zorder=15)
                # ax.plot3D([xs[in_ems_idxs + 100], xs[in_ems_idxs + 100] + vel_scale * vxs[in_ems_idxs + 100]],
                #           [ys[in_ems_idxs + 100], ys[in_ems_idxs + 100] + vel_scale * vys[in_ems_idxs + 100]],
                #           [zs[in_ems_idxs + 100], zs[in_ems_idxs + 100] + vel_scale * vzs[in_ems_idxs + 100]], color='orange',
                #           zorder=15)
                # ax.plot3D(xsdima, ysdima, zsdima)
                ax.scatter(xsdima[in_ems_idxs], ysdima[in_ems_idxs], zsdima[in_ems_idxs], color='blue', s=10, label='$\omega^{\prime}_{SE}$ at SOI of EMS')
                ax.plot3D([xsdima[in_ems_idxs], xsdima[in_ems_idxs] + vel_scale * vxsdima[in_ems_idxs]], [ysdima[in_ems_idxs], ysdima[in_ems_idxs] + vel_scale * vysdima[in_ems_idxs]],
                          [zsdima[in_ems_idxs], zsdima[in_ems_idxs] + vel_scale * vzsdima[in_ems_idxs]], color='blue', zorder=25)

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
                plt.scatter(xs[in_ems_idxs], ys[in_ems_idxs],  color='red', zorder=10, s=10, label='$\omega_{SE}$ at SOI of EMS')
                plt.scatter(xsdima[in_ems_idxs], ysdima[in_ems_idxs], color='blue', s=30, zorder=5,  label='$\omega^{\prime}_{SE}$ at SOI of EMS')
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
                plt.plot([i for i in range(1, len(coherence) + 1)], coherence, linewidth=1, zorder=5, color='red', label='${}^Cv_{TCO\emptyset}$ with $\Omega_{SE}$')
                plt.plot([i for i in range(1, len(coherencea) + 1)], coherencea, linewidth=1, zorder=10, color='blue', label='${}^Cv_{TCO\emptyset}$ with $\Omega^{\prime}_{SE}$')
                plt.plot([i for i in range(1, len(coherencedim) + 1)], coherencedim, linewidth=1, zorder=3, color='green', label='${}^Cv_{TCO}$ with $\Omega_{SE}$')
                plt.plot([i for i in range(1, len(coherenceh) + 1)], coherenceh, linewidth=3, zorder=2, color='orange', label='${}^hv_{TCO}$')
                plt.scatter(in_ems_idxs, coherence[in_ems_idxs], s=20, zorder=5, color='red')
                plt.scatter(in_ems_idxs, coherencea[in_ems_idxs], s=20, zorder=10, color='blue')
                plt.scatter(in_ems_idxs, coherencedim[in_ems_idxs], s=20, zorder=3, color='green')
                plt.scatter(in_ems_idxs, coherenceh[in_ems_idxs], s=20, zorder=2, color='orange')
                plt.xlabel('Timestep')
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

        tco_pop = mm_pop.population[mm_pop.population['Became Minimoon'] == 1]
        stc_pop = mm_pop.population[mm_pop.population['STC'] == True]
        tco_pop_nomoon = mm_pop_nomoon[mm_pop_nomoon['Became Minimoon'] == 1]
        stc_pop_nomoon = mm_pop_nomoon[mm_pop_nomoon['STC'] == True]
        tcf_pop = mm_pop.population[mm_pop.population['Became Minimoon'] == 0]
        tcf_pop_nomoon = mm_pop_nomoon[mm_pop_nomoon['Became Minimoon'] == 0]
        stc_tcos_moon = stc_pop[stc_pop['Became Minimoon'] == 1]
        stc_tcfs_moon = stc_pop[stc_pop['Became Minimoon'] == 0]
        stc_tcos_nomoon = stc_pop_nomoon[stc_pop_nomoon['Became Minimoon'] == 1]
        stc_tcfs_nomoon = stc_pop_nomoon[stc_pop_nomoon['Became Minimoon'] == 0]

        print("Original TCOs: " + str(len(tco_pop['Object id'])))
        print("TCOs without Moon: " + str(len(tco_pop_nomoon['Object id'])))

        print("Original TCFs: " + str(len(tcf_pop['Object id'])))
        print("TCFs without Moon: " + str(len(tcf_pop_nomoon['Object id'])))

        print("Original STCs: " + str(len(stc_pop['Object id'])))
        print("STCs without moon: " + str(len(stc_pop_nomoon['Object id'])))

        print("STCs that are TCOs: " + str(len(stc_tcos_moon['Object id'])))
        print("STCs that are TCOs without moon: " + str(len(stc_tcos_nomoon['Object id'])))

        print("STCs that are TCFs: " + str(len(stc_tcfs_moon['Object id'])))
        print("STCs that are TCFs without moon: " + str(len(stc_tcfs_nomoon['Object id'])))

        tcf_became_tco = 0
        tco_became_tcf = 0
        stayed_tco = 0
        stayed_tcf = 0

        # for idx3, row3 in mm_pop.population.iterrows():
        #
        #     object_id = row3['Object id']
        #     index = mm_pop_nomoon.index[mm_pop_nomoon['Object id'] == object_id].tolist()
        #     tco_occ_nomoon = mm_pop_nomoon.loc[index[0], 'Became Minimoon']
        #     tco_occ_moon = row3['Became Minimoon']
        #
        #     if tco_occ_moon == 1 and tco_occ_nomoon == 1:
        #         stayed_tco += 1
        #     elif tco_occ_moon == 1 and tco_occ_nomoon == 0:
        #         tco_became_tcf += 1
        #     elif tco_occ_moon == 0 and tco_occ_nomoon == 0:
        #         stayed_tcf += 1
        #     else:
        #         tcf_became_tco += 1

        print("TCOs that remained TCOs: " + str(stayed_tco))
        print("TCFs that remained TCFs: " + str(stayed_tcf))
        print("TCOs that became TCFs: " + str(tco_became_tcf))
        print("TCFs that became TCOs: " + str(tcf_became_tco))

        for idx, row in stc_pop.iterrows():
            # print(row)

            stc_name = str(row['Object id']) + '.csv'
            index = mm_pop_nomoon.index[mm_pop_nomoon['Object id'] == row['Object id']].tolist()
            stc_name_no_moon = mm_pop_nomoon.loc[index[0], 'Object id']
            stc_occ = mm_pop_nomoon.loc[index[0], 'STC']

            print("Examining Previously STC: " + stc_name)
            print("Still STC?: " + str(stc_occ)+ " for STC: " + str(stc_name_no_moon))
            data_nomoon = mm_parser.mm_file_parse_new(path_nomoon + '/' + stc_name)
            data_moon = mm_parser.mm_file_parse_new(path_moon + '/' + stc_name)

            # fig3 = plt.figure()
            # ax = fig3.add_subplot(111, projection='3d')
            # vel_scale = 1
            # ut, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
            # xw = 0.0038752837677 * np.cos(ut) * np.sin(v)
            # yw = 0.0038752837677 * np.sin(ut) * np.sin(v)
            # zw = 0.0038752837677 * np.cos(v)
            # ax.plot_wireframe(xw, yw, zw, color="b", alpha=0.1)
            # ax.scatter(0, 0, 0, color='blue', s=10)
            # ax.plot3D(data_moon['Synodic x'], data_moon['Synodic y'], data_moon['Synodic z'], color='grey', zorder=15, linewidth=1, label='With Moon')
            # ax.plot3D(data_nomoon['Synodic x'], data_nomoon['Synodic y'], data_nomoon['Synodic z'], color='orange', zorder=10, linewidth=3, label='Without Moon')
            # ax.set_xlabel('Synodic x (AU)')
            # ax.set_ylabel('Synodic y (AU)')
            # ax.set_zlabel('Synodic z (AU)')
            # ax.set_xlim([-0.01, 0.01])
            # ax.set_ylim([-0.01, 0.01])
            # ax.set_zlim([-0.01, 0.01])
            # ax.set_title('STC ' + str(stc_name_no_moon))
            # num_ticks = 3
            # ax.xaxis.set_major_locator(ticker.MaxNLocator(num_ticks))
            # ax.yaxis.set_major_locator(ticker.MaxNLocator(num_ticks))
            # ax.zaxis.set_major_locator(ticker.MaxNLocator(num_ticks))
            # ax.legend()
            # plt.savefig('figures/' + stc_name_no_moon + '_nomoon.svg', format='svg')
            # plt.savefig('figures/' + stc_name_no_moon + '_nomoon.png', format='png')
            # plt.show()

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

if __name__ == '__main__':

    mm_main = MmMain()

    destination_path = os.path.join(os.getcwd(), 'Test_Set')
    destination_file = destination_path + '/minimoon_master_final.csv'
    start_file = destination_path + '/minimoon_master_final (copy).csv'

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

        errors = mm_main.integrate(destination_file, mu_e, leadtime, perturbers, int_step)

    #########################################
    # integrating data in parallel - check all functions for initializations within
    #########################################

    # create parser
    # mm_parser = MmParser("", "", "")

    # get the master file - you need a list of initial orbits to integrate with openorb (pyorb)
    # master = mm_parser.parse_master(destination_file)

    # get the object ids of all minimoons
    # object_ids = master['Object id']

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

    mm_main.add_new_column(start_file, destination_path)

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
    #
    # mm_main.no_moon_pop(master_file_nomoon, master_file)

