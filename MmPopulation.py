import pandas as pd
import os
import matplotlib.pyplot as plt
from space_fncs import eci_ecliptic_to_sunearth_synodic
import numpy as np
from astropy import constants as const
from MmAnalyzer import MmAnalyzer
from MM_Parser import MmParser

class MmPopulation:
    """
    This is a class to look at statistics and trends that consider the entire synthetic minimoon population
    """

    def __init__(self, pop_file):
        """
        Constructor for the class
        :param pop_file: takes in the population file containing all the information about the minimoons
        :return:
        """
        mm_parser = MmParser("", "",  "")
        self.population = mm_parser.organize_data_new(pop_file)

    @staticmethod
    def tax_table(total_pop, retrograde_pop, prograde_pop):
        """
        Obtain parameters necessary for the table of taxonomic distinctions in Urrutxua 2017
        :return:
        """

        # counts
        total_mm = len(total_pop)
        retrograde_mm = len(retrograde_pop)
        prograde_mm = len(prograde_pop)
        total_t1 = len(total_pop[(total_pop['Taxonomy'] == '1A') | (total_pop['Taxonomy'] == '1B') |
                       (total_pop['Taxonomy'] == '1C')])
        total_t1a = len(total_pop[(total_pop['Taxonomy'] == '1A')])
        total_t1b = len(total_pop[(total_pop['Taxonomy'] == '1B')])
        total_t1c = len(total_pop[(total_pop['Taxonomy'] == '1C')])
        total_t2 = len(total_pop[(total_pop['Taxonomy'] == '2A') | (total_pop['Taxonomy'] == '2B')])
        total_t2a = len(total_pop[(total_pop['Taxonomy'] == '2A')])
        total_t2b = len(total_pop[(total_pop['Taxonomy'] == '2B')])

        retrograde_t1 = len(retrograde_pop[(retrograde_pop['Taxonomy'] == '1A') | (retrograde_pop['Taxonomy'] == '1B') |
                                           (retrograde_pop['Taxonomy'] == '1C')])
        retrograde_t2 = len(retrograde_pop[(retrograde_pop['Taxonomy'] == '2A') | (retrograde_pop['Taxonomy'] == '2B')])
        retrograde_t1a = len(retrograde_pop[(retrograde_pop['Taxonomy'] == '1A')])
        retrograde_t1b = len(retrograde_pop[(retrograde_pop['Taxonomy'] == '1B')])
        retrograde_t1c = len(retrograde_pop[(retrograde_pop['Taxonomy'] == '1C')])
        retrograde_t2a = len(retrograde_pop[(retrograde_pop['Taxonomy'] == '2A')])
        retrograde_t2b = len(retrograde_pop[(retrograde_pop['Taxonomy'] == '2B')])

        prograde_t1 = len(prograde_pop[(prograde_pop['Taxonomy'] == '1A') | (prograde_pop['Taxonomy'] == '1B') |
                                       (prograde_pop['Taxonomy'] == '1C')])
        prograde_t2 = len(prograde_pop[(prograde_pop['Taxonomy'] == '2A') | (prograde_pop['Taxonomy'] == '2B')])
        prograde_t1a = len(prograde_pop[(prograde_pop['Taxonomy'] == '1A')])
        prograde_t1b = len(prograde_pop[(prograde_pop['Taxonomy'] == '1B')])
        prograde_t1c = len(prograde_pop[(prograde_pop['Taxonomy'] == '1C')])
        prograde_t2a = len(prograde_pop[(prograde_pop['Taxonomy'] == '2A')])
        prograde_t2b = len(prograde_pop[(prograde_pop['Taxonomy'] == '2B')])

        # fractions
        retrograde_frac = retrograde_mm / total_mm * 100 if total_mm != 0 else 0
        prograde_frac = prograde_mm / total_mm * 100 if total_mm != 0 else 0
        t1_retro_frac = retrograde_t1 / total_t1 * 100 if total_t1 != 0 else 0
        t1_pro_frac = prograde_t1 / total_t1 * 100 if total_t1 != 0 else 0
        t1a_retro_frac = retrograde_t1a / total_t1a * 100 if total_t1a != 0 else 0
        t1a_pro_frac = prograde_t1a / total_t1a * 100 if total_t1a != 0 else 0
        t1b_retro_frac = retrograde_t1b / total_t1b * 100 if total_t1b != 0 else 0
        t1b_pro_frac = prograde_t1b / total_t1b * 100 if total_t1b != 0 else 0
        t1c_retro_frac = retrograde_t1c / total_t1c * 100 if total_t1c != 0 else 0
        t1c_pro_frac = prograde_t1c / total_t1c * 100 if total_t1c != 0 else 0
        t2_retro_frac = retrograde_t2 / total_t2 * 100 if total_t2 != 0 else 0
        t2_pro_frac = prograde_t2 / total_t2 * 100 if total_t2 != 0 else 0
        t2a_retro_frac = retrograde_t2a / total_t2a * 100 if total_t2a != 0 else 0
        t2a_pro_frac = prograde_t2a / total_t2a * 100 if total_t2a != 0 else 0
        t2b_retro_frac = retrograde_t2b / total_t2b * 100 if total_t2b != 0 else 0
        t2b_pro_frac = prograde_t2b / total_t2b * 100 if total_t2b != 0 else 0
        t1_total_frac = total_t1 / total_mm * 100 if total_mm != 0 else 0
        t2_total_frac = total_t2 / total_mm * 100 if total_mm != 0 else 0
        t1a_total_frac = total_t1a / total_t1 * 100 if total_t1 != 0 else 0
        t1b_total_frac = total_t1b / total_t1 * 100 if total_t1 != 0 else 0
        t1c_total_frac = total_t1c / total_t1 * 100 if total_t1 != 0 else 0
        t2a_total_frac = total_t2a / total_t2 * 100 if total_t2 != 0 else 0
        t2b_total_frac = total_t2b / total_t2 * 100 if total_t2 != 0 else 0

        data = [['TCO(100%)', str(total_mm), str(prograde_mm), str(prograde_frac), str(retrograde_mm),
                 str(retrograde_frac)],
                ['Type I(' + str(t1_total_frac) + '%)', str(total_t1), str(prograde_t1), str(t1_pro_frac),
                 str(retrograde_t1), str(t1_retro_frac)],
                ['Type IA(' + str(t1a_total_frac) + '%)', str(total_t1a), str(prograde_t1a), str(t1a_pro_frac),
                 str(retrograde_t1a), str(t1a_retro_frac)],
                ['Type IB(' + str(t1b_total_frac) + '%)', str(total_t1b), str(prograde_t1b), str(t1b_pro_frac),
                 str(retrograde_t1b), str(t1b_retro_frac)],
                ['Type IC(' + str(t1c_total_frac) + '%)', str(total_t1c), str(prograde_t1c), str(t1c_pro_frac),
                 str(retrograde_t1c), str(t1c_retro_frac)],
                ['Type II(' + str(t2_total_frac) + '%)', str(total_t2), str(prograde_t2), str(t2_pro_frac),
                 str(retrograde_t2), str(t2_retro_frac)],
                ['Type IIA(' + str(t2a_total_frac) + '%)', str(total_t2a), str(prograde_t2a), str(t2a_pro_frac),
                 str(retrograde_t2a), str(t2a_retro_frac)],
                ['Type IIB(' + str(t2b_total_frac) + '%)', str(total_t2b), str(prograde_t2b), str(t2b_pro_frac),
                 str(retrograde_t2b), str(t2b_retro_frac)]]

        cols = ['Type of TCO', 'Total Count', 'Prograde Count', 'Prograde Fraction', 'Retrograde Count',
                'Retrograde Fraction']
        df = pd.DataFrame(data, columns=cols)

        fig, ax = plt.subplots()
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
        #table.auto_set_font_size(False)
        #table.set_fontsize(12)
        fig.tight_layout()

        return df

    def pop_viz(self):
        """
        visualize some of the data from the population
        :return:
        """

        # Get retrograde and prograde TCOs
        tco_pop = mm_pop.population[(mm_pop.population['Became Minimoon'] == 1)]
        retrograde_pop = mm_pop.population[
            (mm_pop.population['Retrograde'] == 1) & (mm_pop.population['Became Minimoon']
                                                      == 1)]
        prograde_pop = mm_pop.population[(mm_pop.population['Retrograde'] == 0) & (mm_pop.population['Became Minimoon']
                                                                                   == 1)]

        fig = plt.figure()
        plt.hist(retrograde_pop['Capture Duration'], bins=100, color='blue', label='Retrograde')
        plt.hist(prograde_pop['Capture Duration'], bins=100, color='red', label='Prograde')
        plt.xlabel('Capture Duration (days)')
        plt.ylabel('Count')
        plt.legend()

        fig2 = plt.figure()
        plt.hist(retrograde_pop['Number of Rev'], bins=100, color='blue', label='Retrograde')
        plt.hist(prograde_pop['Number of Rev'], bins=100, color='red', label='Prograde')
        plt.xlabel('Number of Revolutions (days)')
        plt.ylabel('Count')
        plt.legend()

        fig3 = plt.figure()
        plt.scatter(mm_pop.population['Geo x at Capture'], mm_pop.population['Geo y at Capture'])
        c1 = plt.Circle((0, 0), radius=0.01, alpha=0.1)
        plt.gca().add_artist(c1)
        plt.gca().set_aspect('equal')
        plt.xlabel('Geocentric x at Capture (AU)')
        plt.ylabel('Geocentric y at Capture (AU)')

        fig4 = plt.figure()
        earth_xyz = np.array([mm_pop.population["Helio x at Capture"] - mm_pop.population["Geo x at Capture"],
                              mm_pop.population["Helio y at Capture"] - mm_pop.population["Geo y at Capture"],
                              mm_pop.population["Helio z at Capture"] - mm_pop.population["Geo z at Capture"]])
        mm_xyz = np.array([mm_pop.population["Geo x at Capture"], mm_pop.population["Geo y at Capture"],
                           mm_pop.population["Geo z at Capture"]])
        trans_xyz = eci_ecliptic_to_sunearth_synodic(-earth_xyz, mm_xyz)  # minus is to have sun relative to earth
        plt.scatter(trans_xyz[0, :], trans_xyz[1, :])
        c1 = plt.Circle((0, 0), radius=0.01, alpha=0.1)
        plt.gca().add_artist(c1)
        plt.gca().set_aspect('equal')
        plt.xlabel('Synodic x at Capture (AU)')
        plt.ylabel('Synodic y at Capture (AU)')

        fig5 = plt.figure()
        d = np.sqrt(mm_pop.population["Geo x at Capture"] ** 2 + mm_pop.population["Geo y at Capture"] ** 2
                    + mm_pop.population["Geo z at Capture"] ** 2)

        plt.scatter(mm_pop.population['Capture Duration'], d)
        plt.xlabel('Capture Duration (days)')
        plt.ylabel('Geocentric Distance at Capture (AU)')

        fig6 = plt.figure()
        plt.scatter(mm_pop.population['Capture Duration'], mm_pop.population['Min. Distance'], color='blue',
                    label='Minimum')
        plt.scatter(mm_pop.population['Capture Duration'], mm_pop.population['Max. Distance'], color='red',
                    label='Maximum')
        plt.xlabel('Capture Duration (days)')
        plt.ylabel('Distance to Earth during Capture (AU)')
        plt.legend()

        fig7 = plt.figure()
        tco1_retro = retrograde_pop[(retrograde_pop['Taxonomy'] == '1A') | (retrograde_pop['Taxonomy'] == '1B') |
                                    (retrograde_pop['Taxonomy'] == '1C')]
        tco1_pro = prograde_pop[(prograde_pop['Taxonomy'] == '1A') | (prograde_pop['Taxonomy'] == '1B') |
                                (prograde_pop['Taxonomy'] == '1C')]
        tco2a = tco_pop[(tco_pop['Taxonomy'] == '2A')]
        tco2b = tco_pop[(tco_pop['Taxonomy'] == '2B')]
        plt.scatter(tco1_retro['Capture Duration'], tco1_retro['Number of Rev'], color='blue',
                    label='TCO I (Retrograde)')
        plt.scatter(tco1_pro['Capture Duration'], tco1_pro['Number of Rev'], color='orange', label='TCO I (Prograde)')
        plt.scatter(tco2a['Capture Duration'], tco2a['Number of Rev'], color='yellow', label='TCO IIA')
        plt.scatter(tco2b['Capture Duration'], tco2b['Number of Rev'], color='green', label='TCO IIB')
        plt.xlabel('Capture Duration (days)')
        plt.ylabel('Number of Revolutions')
        plt.legend()

        fig8 = plt.figure()
        plt.scatter(tco_pop["X at Earth Hill"], tco_pop["Y at Earth Hill"])
        c1 = plt.Circle((0, 0), radius=0.01, alpha=0.1)
        plt.gca().add_artist(c1)
        plt.gca().set_aspect('equal')
        plt.xlabel('Synodic x at Earth Hill sphere (AU)')
        plt.ylabel('Synodic y at Earth Hill sphere (AU)')


if __name__ == '__main__':

    population_file = 'minimoon_master.csv'
    population_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'minimoon_integrations', 'minimoon_files_oorb')
    population_path = population_dir + '/' + population_file

    mm_analyzer = MmAnalyzer()
    mm_parser = MmParser("", population_dir, "")
    mm_pop = MmPopulation(population_path)

    # Retrograde vs. Prograde
    # go through all the minimoon file names
    # zero for prograde, one for retrograde
    retrograde = np.zeros([len(mm_pop.population['Object id']), 1])
    max_dist = np.zeros([len(mm_pop.population['Object id']), 1])
    mm_flag = np.zeros([len(mm_pop.population['Object id']), 1])
    capture_idx = np.zeros([len(mm_pop.population['Object id']), 1],  dtype=np.int64)
    release_idx = np.zeros([len(mm_pop.population['Object id']), 1],  dtype=np.int64)
    x_ehs = np.zeros([len(mm_pop.population['Object id']), 1])
    y_ehs = np.zeros([len(mm_pop.population['Object id']), 1])
    z_ehs = np.zeros([len(mm_pop.population['Object id']), 1])
    designations = []
    hill_crossing = np.zeros([len(mm_pop.population['Object id']), 1])

    for idx in range(len(mm_pop.population['Object id'])):

        # go through all the files of test particles
        for root, dirs, files in os.walk(population_dir):
            # find files that are minimoons
            name = str(mm_pop.population['Object id'].iloc[idx]) + ".csv"
            if name in files:
                file_path = os.path.join(root, name)

                # read the file
                header = 0
                data = mm_parser.mm_file_parse_new(file_path, 0)

                distance = data['Distance']
                eh_crossing = min(distance, key=lambda x: abs(x-0.01))
                data_eh_crossing = data[distance == eh_crossing]

                # compute revs
                mu_e = const.GM_earth.value
                mm_analyzer = MmAnalyzer()
                mm_analyzer.minimoon_check(data, mu_e)

                retrograde[idx, 0] = mm_analyzer.retrograde
                max_dist[idx, 0] = mm_analyzer.max_dist
                mm_flag[idx, 0] = mm_analyzer.minimoon_flag
                capture_idx[idx, 0] = int(mm_analyzer.cap_idx)
                release_idx[idx, 0] = int(mm_analyzer.rel_idx)
                x_ehs[idx, 0] = data_eh_crossing['Synodic x'].values
                y_ehs[idx, 0] = data_eh_crossing['Synodic y'].values
                z_ehs[idx, 0] = data_eh_crossing['Synodic z'].values

    mm_pop.population["Retrograde"] = retrograde[:, 0]
    mm_pop.population["Became Minimoon"] = mm_flag[:, 0]
    mm_pop.population["Max. Distance"] = max_dist[:, 0]
    mm_pop.population["Capture Index"] = capture_idx[:, 0]
    mm_pop.population["Release Index"] = release_idx[:, 0]
    mm_pop.population["X at Earth Hill"] = x_ehs[:, 0]
    mm_pop.population["Y at Earth Hill"] = y_ehs[:, 0]
    mm_pop.population["Z at Earth Hill"] = z_ehs[:, 0]

    for idx in range(len(mm_pop.population['Object id'])):

        # go through all the files of test particles
        for root, dirs, files in os.walk(population_dir):
            # find files that are minimoons
            name = str(mm_pop.population['Object id'].iloc[idx]) + ".csv"
            if name in files:
                file_path = os.path.join(root, name)
                # read the file
                header = 0
                data = mm_parser.mm_file_parse_new(file_path, 0)
                designation = mm_analyzer.taxonomy(data, mm_pop.population)
                designations.append(designation)

    mm_pop.population["Taxonomy"] = designations

    destination_file = 'minimoon_master_temp.csv'
    destination_path = population_dir + '/' + destination_file
    mm_pop.population.to_csv(destination_path, sep=' ', header=True, index=False)

    # Make taxonomy table
    mm_pop.tax_table()

    # Generate visualization
    mm_pop.pop_viz()

    plt.show()



