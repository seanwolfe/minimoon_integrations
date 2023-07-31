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
        int_step_rev = int(1/(old_data['Julian Date'].iloc[1] - old_data['Julian Date'].iloc[0]))
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
        master = mm_parser.parse_master_previous(master_path)

        # parrelelized version of short term capture
        # pool = multiprocessing.Pool()
        # results = pool.map(mm_analyzer.short_term_capture, master['Object id'])  # input your function
        # pool.close()

        # parallel version of jacobi alpha beta
        stc_pop = master[master['STC'] == True]
        print(master)
        for i, val in enumerate(stc_pop['Object id']):
            res = mm_analyzer.alpha_beta_jacobi(stc_pop['Object id'].iloc[i])
            # res = mm_analyzer.short_term_capture(master['Object id'].iloc[i])
        # pool = multiprocessing.Pool()
        # results = pool.map(mm_analyzer.alpha_beta_jacobi, master['Object id'])
        # pool.close()

        # repack list according to index
        repacked_results = [list(items) for items in zip(*results)]  # when running parallel processing

        # create your columns according to the data in results
        # master['Entry Date to EMS'] = repacked_results[10]  # start of ems stay
        # master['Entry to EMS Index'] = repacked_results[11]  # ems start index
        # master['Exit Date to EMS'] = repacked_results[12]  # end ems
        # master['Exit Index to EMS'] = repacked_results[13]  # end ems index

        # pd.set_option('display.max_rows', None)
        # print(master['Entry Date to EMS'])

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

    mm_main.add_new_column(start_file, destination_file)

    ########################################
    # clustering graphs
    ########################################

    # mm_main.cluster_viz_main(destination_file)

    ########################################
    # STC population visualization
    #######################################

    # mm_main.stc_viz_main(destination_file)