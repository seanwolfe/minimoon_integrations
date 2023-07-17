import pandas as pd
import os
import matplotlib.pyplot as plt
from space_fncs import eci_ecliptic_to_sunearth_synodic
import numpy as np
from astropy import constants as const
from MmAnalyzer import MmAnalyzer
from MM_Parser import MmParser
import astropy.units as u
import multiprocessing

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
        # self.population = mm_parser.organize_data_new(pop_file)
        self.population = mm_parser.parse_master(pop_file)

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
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        plt.savefig("figures/f0.svg", format="svg")

        return df

    def pop_viz(self):
        """
        visualize some of the data from the population
        :return:
        """

        # Get retrograde and prograde TCOs
        tco_pop = self.population[(self.population['Became Minimoon'] == 1)]
        retrograde_pop = self.population[
            (self.population['Retrograde'] == 1) & (self.population['Became Minimoon']
                                                      == 1)]
        prograde_pop = self.population[(self.population['Retrograde'] == 0) & (self.population['Became Minimoon']
                                                                                   == 1)]

        fig = plt.figure()
        plt.hist(retrograde_pop['Capture Duration'], bins=1000, label='Retrograde', edgecolor="#038cfc", color="#03b1fc")
        plt.hist(prograde_pop['Capture Duration'], bins=1000, label='Prograde', edgecolor="#ed0000", color="#f54747")
        plt.xlim([0, 1000])
        plt.ylim([0, 700])
        plt.xlabel('Capture Duration (days)')
        plt.ylabel('Count')
        plt.legend()
        # plt.savefig("figures/f1.svg", format="svg")

        fig2 = plt.figure()
        plt.hist(abs(retrograde_pop['Number of Rev']), bins=350*20, label='Retrograde', edgecolor="#038cfc", color="#03b1fc")
        plt.hist(abs(prograde_pop['Number of Rev']), bins=350*20, label='Prograde', edgecolor="#ed0000", color="#f54747")
        plt.xlim([1, 5])
        plt.ylim([0,1600])
        plt.xlabel('Number of Revolutions')
        plt.ylabel('Count')
        plt.legend()
        # plt.savefig("figures/f2.svg", format="svg")

        fig3 = plt.figure()
        plt.scatter(tco_pop['Geo x at Capture'], tco_pop['Geo y at Capture'], s=0.1)
        c1 = plt.Circle((0, 0), radius=0.01, alpha=0.1)
        plt.gca().add_artist(c1)
        plt.gca().set_aspect('equal')
        plt.xlabel('Geocentric x at Capture (AU)')
        plt.ylabel('Geocentric y at Capture (AU)')
        # plt.savefig("figures/f3.svg", format="svg")

        fig4 = plt.figure()
        earth_xyz = np.array([tco_pop["Helio x at Capture"] - tco_pop["Geo x at Capture"],
                              tco_pop["Helio y at Capture"] - tco_pop["Geo y at Capture"],
                              tco_pop["Helio z at Capture"] - tco_pop["Geo z at Capture"]])
        mm_xyz = np.array([tco_pop["Geo x at Capture"], tco_pop["Geo y at Capture"],
                           tco_pop["Geo z at Capture"]])
        trans_xyz = eci_ecliptic_to_sunearth_synodic(-earth_xyz, mm_xyz)  # minus is to have sun relative to earth
        plt.scatter(trans_xyz[0, :], trans_xyz[1, :], s=0.1)
        plt.xlim([-0.03, 0.03])
        c1 = plt.Circle((0, 0), radius=0.01, alpha=0.1)
        plt.gca().add_artist(c1)
        plt.gca().set_aspect('equal')
        plt.xlabel('Synodic x at Capture (AU)')
        plt.ylabel('Synodic y at Capture (AU)')
        # plt.savefig("figures/f4.svg", format="svg")

        fig5 = plt.figure()
        d = np.sqrt(tco_pop["Geo x at Capture"] ** 2 + tco_pop["Geo y at Capture"] ** 2
                    + tco_pop["Geo z at Capture"] ** 2)
        plt.xlim([0,500])
        plt.ylim([0,0.03])
        plt.scatter(tco_pop['Capture Duration'], d, s=1)
        plt.xlabel('Capture Duration (days)')
        plt.ylabel('Geocentric Distance at Capture (AU)')
        # plt.savefig("figures/f5.svg", format="svg")

        fig6 = plt.figure()
        plt.scatter(tco_pop['Capture Duration'], tco_pop['Min. Distance'], color='blue',
                    label='Minimum', s=1)
        plt.scatter(tco_pop['Capture Duration'], tco_pop['Max. Distance'], color='red',
                    label='Maximum', s=1)
        plt.xlabel('Capture Duration (days)')
        plt.ylabel('Distance to Earth during Capture (AU)')
        plt.xlim([0, 500])
        plt.ylim([0, 0.065])
        plt.legend()
        # plt.savefig("figures/f6.svg", format="svg")

        fig7 = plt.figure()
        tco1_retro = retrograde_pop[(retrograde_pop['Taxonomy'] == '1A') | (retrograde_pop['Taxonomy'] == '1B') |
                                    (retrograde_pop['Taxonomy'] == '1C')]
        tco1_pro = prograde_pop[(prograde_pop['Taxonomy'] == '1A') | (prograde_pop['Taxonomy'] == '1B') |
                                (prograde_pop['Taxonomy'] == '1C')]
        tco2a = tco_pop[(tco_pop['Taxonomy'] == '2A')]
        tco2b = tco_pop[(tco_pop['Taxonomy'] == '2B')]
        plt.scatter(tco1_retro['Capture Duration'], abs(tco1_retro['Number of Rev']), color='blue',
                    label='TCO I (Retrograde)', s=4.)
        plt.scatter(tco1_pro['Capture Duration'], abs(tco1_pro['Number of Rev']), color='brown', label='TCO I (Prograde)', s=4.)
        plt.scatter(tco2a['Capture Duration'], abs(tco2a['Number of Rev']), color='yellow', label='TCO IIA', s=4.)
        plt.scatter(tco2b['Capture Duration'], abs(tco2b['Number of Rev']), color='green', label='TCO IIB', s=20.)
        plt.xlabel('Capture Duration (days)')
        plt.ylabel('Number of Revolutions')
        plt.xlim([0,500])
        plt.ylim([1,3])
        plt.legend()
        # plt.savefig("figures/f7.svg", format="svg")

        fig8 = plt.figure()
        plt.scatter(tco_pop["X at Earth Hill"], tco_pop["Y at Earth Hill"], s=0.1)
        c1 = plt.Circle((0, 0), radius=0.01, alpha=0.1)
        plt.gca().add_artist(c1)
        plt.gca().set_aspect('equal')
        plt.xlabel('Synodic x at Earth Hill sphere (AU)')
        plt.ylabel('Synodic y at Earth Hill sphere (AU)')
        # plt.savefig("figures/f8.svg", format="svg")

    def stc_pop_viz(self):

        stc_pop = self.population[self.population['STC'] == True]
        non_stc_pop = self.population[self.population['STC'] == False]

        # Set the bin size
        bin_size = 1

        # Calculate the number of bins based on data range and bin size
        data_range_stc = max(stc_pop['EMS Duration']) - min(stc_pop['EMS Duration'])
        num_bins_stc = int(data_range_stc / bin_size)
        data_range_nonstc = max(non_stc_pop['EMS Duration']) - min(non_stc_pop['EMS Duration'])
        num_bins_nonstc = int(data_range_nonstc / bin_size)

        fig = plt.figure()
        plt.hist(stc_pop['EMS Duration'], bins=num_bins_stc, label='STCs', edgecolor="#038cfc",
                 color="#03b1fc")
        plt.hist(non_stc_pop['EMS Duration'], bins=num_bins_nonstc, label='Non-STCs', edgecolor="#ed0000", color="#f54747")
        plt.xlim([0, 60])
        plt.ylim([0, 3500])
        plt.xlabel('Duration in the SOI of EMS (days)')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig("figures/tmes_hist.svg", format="svg")

        # Set the bin size
        bin_size = 10

        # Calculate the number of bins based on data range and bin size
        data_range_stc = max(stc_pop['3 Hill Duration']) - min(stc_pop['3 Hill Duration'])
        num_bins_stc = int(data_range_stc / bin_size)
        data_range_nonstc = max(non_stc_pop['3 Hill Duration']) - min(non_stc_pop['3 Hill Duration'])
        num_bins_nonstc = int(data_range_nonstc / bin_size)

        fig2 = plt.figure()
        plt.hist(stc_pop['3 Hill Duration'], bins=num_bins_stc, label='STCs', edgecolor="#038cfc",
                 color="#03b1fc", zorder=10, alpha=0.5,)
        plt.hist(non_stc_pop['3 Hill Duration'], bins=num_bins_nonstc, label='Non-STCs', edgecolor="#ed0000",
                 color="#f54747", zorder=5)
        plt.xlim([0, 1600])
        plt.ylim([0, 650])
        plt.xlabel('Capture Duration (days)')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig("figures/t_hist.svg", format="svg")


        string = 'Periapsides in 2 Hill'
        # Set the bin size
        bin_size = 1

        # Calculate the number of bins based on data range and bin size
        data_range_stc = max(stc_pop[string]) - min(stc_pop[string])
        num_bins_stc = int(data_range_stc / bin_size)
        data_range_nonstc = max(non_stc_pop[string]) - min(non_stc_pop[string])
        num_bins_nonstc = int(data_range_nonstc / bin_size)

        fig3 = plt.figure()
        plt.hist(stc_pop[string], bins=num_bins_stc, label='STCs', edgecolor="#038cfc",
                 color="#03b1fc", alpha=0.5, zorder=10)
        plt.hist(non_stc_pop[string], bins=num_bins_nonstc, label='Non-STCs', edgecolor="#ed0000",
                 color="#f54747", zorder=5)
        plt.xlim([0, 20])
        plt.xticks(range(0, 20))
        plt.ylim([0, 6500])
        plt.xlabel('Number of Periapsides Within Two Earth Hill Radii')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig("figures/peri_2.svg", format="svg")

        string = 'Periapsides in 3 Hill'
        # Set the bin size
        bin_size = 1

        # Calculate the number of bins based on data range and bin size
        data_range_stc = max(stc_pop[string]) - min(stc_pop[string])
        num_bins_stc = int(data_range_stc / bin_size)
        data_range_nonstc = max(non_stc_pop[string]) - min(non_stc_pop[string])
        num_bins_nonstc = int(data_range_nonstc / bin_size)

        fig4 = plt.figure()
        plt.hist(stc_pop[string], bins=num_bins_stc, label='STCs', edgecolor="#038cfc",
                 color="#03b1fc", zorder=10, alpha=0.5)
        plt.hist(non_stc_pop[string], bins=num_bins_nonstc, label='Non-STCs', edgecolor="#ed0000",
                 color="#f54747", zorder=5)
        plt.xlim([0, 20])
        plt.xticks(range(0, 20))
        plt.ylim([0, 6000])
        plt.xlabel('Number of Periapsides Within Three Earth Hill Radii')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig("figures/peri_3.svg", format="svg")

        string = 'Periapsides in 1 Hill'
        # Set the bin size
        bin_size = 1

        # Calculate the number of bins based on data range and bin size
        data_range_stc = max(stc_pop[string]) - min(stc_pop[string])
        num_bins_stc = int(data_range_stc / bin_size)
        data_range_nonstc = max(non_stc_pop[string]) - min(non_stc_pop[string])
        num_bins_nonstc = int(data_range_nonstc / bin_size)

        fig5 = plt.figure()
        plt.hist(stc_pop[string], bins=num_bins_stc, label='STCs', edgecolor="#038cfc",
                 color="#03b1fc", zorder=10, alpha=0.5)
        plt.hist(non_stc_pop[string], bins=num_bins_nonstc, label='Non-STCs', edgecolor="#ed0000",
                 color="#f54747",  zorder=5)
        plt.xlim([0, 20])
        plt.xticks(range(0, 20))
        plt.ylim([0, 12000])
        plt.xlabel('Number of Periapsides Within One Earth Hill Radii')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig("figures/peri_1.svg", format="svg")

        string = 'Periapsides in EMS'
        # Set the bin size
        bin_size = 1

        # Calculate the number of bins based on data range and bin size
        data_range_stc = max(stc_pop[string]) - min(stc_pop[string])
        num_bins_stc = int(data_range_stc / bin_size)
        data_range_nonstc = max(non_stc_pop[string]) - min(non_stc_pop[string])
        num_bins_nonstc = int(data_range_nonstc / bin_size)

        fig6 = plt.figure()
        plt.hist(stc_pop[string], bins=num_bins_stc, label='STCs', edgecolor="#038cfc",
                 color="#03b1fc", zorder=10)
        plt.hist(non_stc_pop[string], bins=2, label='Non-STCs', edgecolor="#ed0000",
                 color="#f54747", zorder=5)
        plt.xlim([0, 10])
        plt.ylim([0, 12000])
        plt.xlabel('Number of Periapsides Within the EMS')
        plt.xticks(range(0, 10))
        plt.ylabel('Count')
        plt.legend()
        plt.savefig("figures/peri_ems.svg", format="svg")

        fig7 = plt.figure()
        earth_xyz = np.array([stc_pop["Earth x at EMS (Helio)"], stc_pop["Earth y at EMS (Helio)"],
                              stc_pop["Earth z at EMS (Helio)"]])
        mm_xyz = np.array([stc_pop["Helio x at EMS"], stc_pop["Helio y at EMS"], stc_pop["Helio z at EMS"]]) - earth_xyz
        trans_xyz = eci_ecliptic_to_sunearth_synodic(-earth_xyz, mm_xyz)  # minus is to have sun relative to earth
        plt.scatter(trans_xyz[0, :], trans_xyz[1, :], s=0.1)
        plt.xlim([-0.01, 0.01])
        c1 = plt.Circle((0, 0), radius=0.0038752837677, alpha=0.1)
        plt.gca().add_artist(c1)
        plt.gca().set_aspect('equal')
        plt.xlabel('Synodic x at EMS (AU)')
        plt.ylabel('Synodic y at EMS (AU)')
        plt.savefig("figures/stc_at_ems.svg", format="svg")

        plt.show()

        return


if __name__ == '__main__':

    population_file = 'minimoon_master_final.csv'
    # population_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'minimoon_integrations', 'minimoon_files_oorb')
    # population_dir = os.path.join(os.path.expanduser('~'), 'Documents', 'sean', 'minimoon_integrations',
    #                               'minimoon_files_oorb')
    # population_path = population_dir + '/' + population_file

    # mm_analyzer = MmAnalyzer()
    # mm_parser = MmParser("", population_dir, "")
    # mm_pop = MmPopulation(population_path)

    # mm_pop.stc_pop_viz()





