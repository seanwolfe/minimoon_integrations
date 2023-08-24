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
import matplotlib.ticker as ticker

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

    @staticmethod
    def make_set(master, variable, axislabel):
        fig8 = plt.figure()
        ax1 = plt.subplot(3, 3, 1)
        ax1.scatter(master['Helio x at Capture'], variable, s=0.1)
        ax1.set_xlabel('Helio x at Capture (AU)')
        ax1.set_ylabel(axislabel)
        # plt.savefig("figures/helx_3hill.svg", format="svg")

        ax2 = plt.subplot(3, 3, 2)
        ax2.scatter(master['Helio y at Capture'], variable, s=0.1)
        ax2.set_xlabel('Helio y at Capture (AU)')
        ax2.set_ylabel(axislabel)
        # plt.savefig("figures/hely_3hill.svg", format="svg")

        ax3 = plt.subplot(3, 3, 3)
        ax3.scatter(master['Helio z at Capture'], variable, s=0.1)
        ax3.set_xlabel('Helio z at Capture (AU)')
        ax3.set_ylabel(axislabel)
        # plt.savefig("figures/helz_3hill.svg", format="svg")

        ax4 = plt.subplot(3, 3, 4)
        ax4.scatter(master['Helio vx at Capture'], variable, s=0.1)
        ax4.set_xlabel('Helio vx at Capture')
        ax4.set_ylabel(axislabel)
        # plt.savefig("figures/helvx_3hill.svg", format="svg")

        ax5 = plt.subplot(3, 3, 5)
        ax5.scatter(master['Helio vy at Capture'], variable, s=0.1)
        ax5.set_xlabel('Helio vy at Capture')
        ax5.set_ylabel(axislabel)
        # plt.savefig("figures/helvy_3hill.svg", format="svg")

        ax6 = plt.subplot(3, 3, 6)
        ax6.scatter(master['Helio vz at Capture'], variable, s=0.1)
        ax6.set_xlabel('Helio vz at Capture')
        ax6.set_ylabel(axislabel)
        # plt.savefig("figures/helvz_3hill.svg", format="svg")

        ax7 = plt.subplot(3, 3, 7)
        ax7.scatter(master['Capture Date'], variable, s=0.1)
        ax7.set_xlabel('Capture Date (JD)')
        ax7.set_ylabel(axislabel)

        colored = 'red'

        fig8 = plt.figure()
        ax1 = plt.subplot(3, 3, 1)
        ax1.scatter(master['Helio x at Release'], variable, s=0.1, color=colored)
        ax1.set_xlabel('Helio x at Release (AU)')
        ax1.set_ylabel(axislabel)
        # plt.savefig("figures/helx_3hill.svg", format="svg")

        ax2 = plt.subplot(3, 3, 2)
        ax2.scatter(master['Helio y at Release'], variable, s=0.1, color=colored)
        ax2.set_xlabel('Helio y at Release (AU)')
        ax2.set_ylabel(axislabel)
        # plt.savefig("figures/hely_3hill.svg", format="svg")

        ax3 = plt.subplot(3, 3, 3)
        ax3.scatter(master['Helio z at Release'], variable, s=0.1, color=colored)
        ax3.set_xlabel('Helio z at Release (AU)')
        ax3.set_ylabel(axislabel)
        # plt.savefig("figures/helz_3hill.svg", format="svg")

        ax4 = plt.subplot(3, 3, 4)
        ax4.scatter(master['Helio vx at Release'], variable, s=0.1, color=colored)
        ax4.set_xlabel('Helio vx at Release')
        ax4.set_ylabel(axislabel)
        # plt.savefig("figures/helvx_3hill.svg", format="svg")

        ax5 = plt.subplot(3, 3, 5)
        ax5.scatter(master['Helio vy at Release'], variable, s=0.1, color=colored)
        ax5.set_xlabel('Helio vy at Release')
        ax5.set_ylabel(axislabel)
        # plt.savefig("figures/helvy_3hill.svg", format="svg")

        ax6 = plt.subplot(3, 3, 6)
        ax6.scatter(master['Helio vz at Release'], variable, s=0.1, color=colored)
        ax6.set_xlabel('Helio vz at Release')
        ax6.set_ylabel(axislabel)
        # plt.savefig("figures/helvz_3hill.svg", format="svg")

        ax7 = plt.subplot(3, 3, 7)
        ax7.scatter(master['Release Date'], variable, s=0.1, color=colored)
        ax7.set_xlabel('Release Date (JD)')
        ax7.set_ylabel(axislabel)

        colored = 'green'

        fig8 = plt.figure()
        ax1 = plt.subplot(3, 3, 1)
        ax1.scatter(master['Helio x at EMS'], variable, s=0.1, color=colored)
        ax1.set_xlabel('Helio x at EMS (AU)')
        ax1.set_ylabel(axislabel)
        # plt.savefig("figures/helx_3hill.svg", format="svg")

        ax2 = plt.subplot(3, 3, 2)
        ax2.scatter(master['Helio y at EMS'], variable, s=0.1, color=colored)
        ax2.set_xlabel('Helio y at EMS (AU)')
        ax2.set_ylabel(axislabel)
        # plt.savefig("figures/hely_3hill.svg", format="svg")

        ax3 = plt.subplot(3, 3, 3)
        ax3.scatter(master['Helio z at EMS'], variable, s=0.1, color=colored)
        ax3.set_xlabel('Helio z at Release (AU)')
        ax3.set_ylabel(axislabel)
        # plt.savefig("figures/helz_3hill.svg", format="svg")

        ax4 = plt.subplot(3, 3, 4)
        ax4.scatter(master['Helio vx at EMS'], variable, s=0.1, color=colored)
        ax4.set_xlabel('Helio vx at EMS')
        ax4.set_ylabel(axislabel)
        # plt.savefig("figures/helvx_3hill.svg", format="svg")

        ax5 = plt.subplot(3, 3, 5)
        ax5.scatter(master['Helio vy at EMS'], variable, s=0.1, color=colored)
        ax5.set_xlabel('Helio vy at EMS')
        ax5.set_ylabel(axislabel)
        # plt.savefig("figures/helvy_3hill.svg", format="svg")

        ax6 = plt.subplot(3, 3, 6)
        ax6.scatter(master['Helio vz at EMS'], variable, s=0.1, color=colored)
        ax6.set_xlabel('Helio vz at EMS')
        ax6.set_ylabel(axislabel)
        # plt.savefig("figures/helvz_3hill.svg", format="svg")

        ax7 = plt.subplot(3, 3, 7)
        ax7.scatter(master['Entry Date to EMS'], variable, s=0.1, color=colored)
        ax7.set_xlabel('Date at SOI of EMS (JD)')
        ax7.set_ylabel(axislabel)

        return

    def cluster_viz(self):


        master = self.population[self.population['STC'] == True]
        self.make_set(master, master['3 Hill Duration'], 'Duration in 3 Earth Hill (Days)')
        self.make_set(master, master['Periapsides in EMS'], 'Number of Periapsides Inside the SOI of the EMS')
        self.make_set(master, master['Min. Distance'], 'Minimum Distance to Earth (AU)')
        self.make_set(master, master['Max. Distance'], 'Maximum Distance to Earth (during capture) (AU)')
        self.make_set(master, master['Non-Dimensional Jacobi'], 'Jacobi Constant')



        # plt.savefig("figures/capdate_3hill.svg", format="svg")

        # Set the bin size
        bin_size = 10

        # Calculate the number of bins based on data range and bin size
        data_range = max(master['3 Hill Duration']) - min(master['3 Hill Duration'])
        num_bins = int(data_range / bin_size)

        fig2 = plt.figure()
        plt.hist(master['3 Hill Duration'], bins=num_bins, edgecolor="#038cfc",
                 color="#03b1fc", zorder=10, alpha=0.5, )
        plt.xlim([0, 1400])
        plt.ylim([0, 800])
        plt.xlabel('3 Hill Duration(days)')
        plt.ylabel('Count')
        # plt.savefig("figures/3hill_hist.svg", format="svg")
        # plt.savefig("figures/3hill_hist.pdf", format="pdf")


        # Set the bin size
        bin_size = 0.00001

        # Calculate the number of bins based on data range and bin size
        data_range = max(master['Helio vz at Capture']) - min(master['Helio vz at Capture'])
        num_bins = int(data_range / bin_size)

        fig2 = plt.figure()
        plt.hist(master['Helio vz at Capture'], bins=num_bins, edgecolor="#038cfc",
                 color="#03b1fc", zorder=10, alpha=0.5, )
        plt.xlim([-0.0005, 0.0005])
        plt.ylim([0, 1500])
        plt.xlabel('Helio vz at Capture (AU/day)')
        plt.ylabel('Count')
        # plt.savefig("figures/vz_hist.svg", format="svg")
        # plt.savefig("figures/vz_hist.pdf", format="pdf")

        plt.show()

        # mm_geo = np.array([master['Geo x at Capture'], master['Geo y at Capture'], master['Geo z at Capture']])
        # mm_helio = np.array([master["Helio x at Capture"], master["Helio y at Capture"], master["Helio z at Capture"]])
        # earth_xyz = mm_helio - mm_geo

        # fig8 = plt.figure()
        # plt.scatter(master['Geo x at Capture'], master['3 Hill Duration'], s=0.1)
        # plt.xlabel('Geo x at Capture (AU)')
        # plt.ylabel('Time Spend in 3 Hill (days)')
        # plt.savefig("figures/geox_3hill.svg", format="svg")

        # fig8 = plt.figure()
        # plt.scatter(master['Geo y at Capture'], master['3 Hill Duration'], s=0.1)
        # plt.xlabel('Geo y at Capture (AU)')
        # plt.ylabel('Time Spend in 3 Hill (days)')
        # plt.savefig("figures/geoy_3hill.svg", format="svg")

        # fig8 = plt.figure()
        # plt.scatter(master['Geo z at Capture'], master['3 Hill Duration'], s=0.1)
        # plt.xlabel('Geo z at Capture (AU)')
        # plt.ylabel('Time Spend in 3 Hill (days)')
        # plt.savefig("figures/geoz_3hill.svg", format="svg")

        # trans_xyz = eci_ecliptic_to_sunearth_synodic(-earth_xyz, mm_geo)  # minus is to have sun relative to earth
        # fig8 = plt.figure()
        # plt.scatter(trans_xyz[0], master['3 Hill Duration'], s=0.1)
        # plt.xlabel('Synodic Earth Centered x at Capture (AU)')
        # plt.ylabel('Time Spend in 3 Hill (days)')
        # plt.savefig("figures/synx_3hill.svg", format="svg")

        # fig8 = plt.figure()
        # plt.scatter(trans_xyz[1], master['3 Hill Duration'], s=0.1)
        # plt.xlabel('Synodic Earth Centered y at Capture (AU)')
        # plt.ylabel('Time Spend in 3 Hill (days)')
        # plt.savefig("figures/syny_3hill.svg", format="svg")

        # fig8 = plt.figure()
        # plt.scatter(trans_xyz[2], master['3 Hill Duration'], s=0.1)
        # plt.xlabel('Synodic Earth Centered z at Capture (AU)')
        # plt.ylabel('Time Spend in 3 Hill (days)')
        # plt.savefig("figures/synz_3hill.svg", format="svg")



        # fig8 = plt.figure()
        # plt.scatter(master['Capture Date'] - master['STC Start'], master['3 Hill Duration'], s=0.1)
        # plt.xlabel('Time from Crossing 3 Hill to Capture')
        # plt.ylabel('3 Hill Duration (days)')
        # plt.savefig("figures/3hill_to_cap_time_3hill.svg", format="svg")

        # fig8 = plt.figure()
        # plt.scatter(master['Geo vx at Capture'], master['3 Hill Duration'], s=0.1)
        # plt.xlabel('Geo vx at Capture')
        # plt.ylabel('3 Hill Duration (days)')
        # plt.savefig("figures/geovx_3hill.svg", format="svg")

        # fig8 = plt.figure()
        # plt.scatter(master['Geo vy at Capture'], master['3 Hill Duration'], s=0.1)
        # plt.xlabel('Geo vy at Capture')
        # plt.ylabel('3 Hill Duration (days)')
        # plt.savefig("figures/geovy_3hill.svg", format="svg")

        # fig8 = plt.figure()
        # plt.scatter(master['Geo vz at Capture'], master['3 Hill Duration'], s=0.1)
        # plt.xlabel('Geo vz at Capture')
        # plt.ylabel('3 Hill Duration (days)')
        # plt.savefig("figures/geovz_3hill.svg", format="svg")

        # fig8 = plt.figure()
        # plt.scatter(trans_xyz[0], master['1 Hill Duration'], s=0.1)
        # plt.xlabel('Synodic Earth Centered x at Capture (AU)')
        # plt.ylabel('Time Spend in 1 Hill (days)')
        #
        # fig8 = plt.figure()
        # plt.scatter(trans_xyz[1], master['1 Hill Duration'], s=0.1)
        # plt.xlabel('Synodic Earth Centered y at Capture (AU)')
        # plt.ylabel('Time Spend in 1 Hill (days)')
        #
        # fig8 = plt.figure()
        # plt.scatter(trans_xyz[2], master['1 Hill Duration'], s=0.1)
        # plt.xlabel('Synodic Earth Centered z at Capture (AU)')
        # plt.ylabel('Time Spend in 1 Hill (days)')
        #
        # fig8 = plt.figure()
        # plt.scatter(master['Capture Date'], master['1 Hill Duration'], s=0.1)
        # plt.xlabel('Capture Date (JD)')
        # plt.ylabel('Time Spend in 3 Hill (days)')
        #
        # fig8 = plt.figure()
        # plt.scatter(master['Capture Date'] - master['STC Start'], master['1 Hill Duration'], s=0.1)
        # plt.xlabel('Time from Crossing 3 Hill to Capture')
        # plt.ylabel('1 Hill Duration (days)')
        # plt.show()
        # plt.savefig("figures/stc_at_ems.svg", format="svg")

        # fig8 = plt.figure()
        # plt.scatter(master.index.tolist(), master['3 Hill Duration'], s=0.1)
        # plt.xlabel('Id of TCO')
        # plt.ylabel('3 Hill Duration (days)')
        # plt.savefig("figures/tcoid_3hill.svg", format="svg")
        # plt.savefig("figures/tcoid_3hill.pdf", format="pdf")

        # fig8 = plt.figure()
        # plt.scatter(master.index.tolist(), master['Helio vz at Capture'], s=0.1)
        # plt.xlabel('Id of TCO')
        # plt.ylabel('Helio vz at Capture (AU/day)')
        # plt.savefig("figures/tcoid_vz.svg", format="svg")
        # plt.savefig("figures/tcoid_vz.pdf", format="pdf")


        """
        start_value = 0
        end_value = max(master['3 Hill Duration'])
        desired_separation = 100
        num_elements = int((end_value - start_value) / desired_separation) + 1
        bins = np.linspace(start_value, end_value, num_elements)

        # master['Duration Bin'] = pd.cut(master['3 Hill Duration'], bins)

        means = []
        variances = []
        binning = []
        for i in range(0, num_elements):



            # fig1 = plt.figure()
            # plt.scatter(filter['Helio z at Capture'], filter['3 Hill Duration'])
            # plt.xlim([-0.02, 0.02])
            # plt.ylim([0, 12000])

            if True:
                bin = desired_separation
                desired_bin = (bin*i, bin*i+bin)

                filter = master[(master['3 Hill Duration'] > bin*i) & (master['3 Hill Duration'] < bin*i+bin)]
                binning.append((bin*i + bin*i + bin)/2)
                mean = np.mean(filter['Helio vz at Capture'])
                means.append(mean)
                variance = np.sum((np.array(filter['Helio vz at Capture']) - mean) ** 2) / len(filter['Helio vz at Capture'])
                variances.append(variance)

                # plt.scatter(filter['Helio z at Capture'], filter['3 Hill Duration'], s=1, label=str(desired_bin))
                # plt.xlim([-0.02, 0.02])
                # plt.ylim([0, 12000])
                # plt.legend()
                # print(desired_bin)

            # plt.show()

        coefficients = np.polyfit(binning, variances, 5)

        # Create a polynomial function using the coefficients
        polynomial_func = np.poly1d(coefficients)

        # Generate x values for the curve
        x_curve = np.linspace(min(binning), max(binning), desired_separation)

        # Calculate the corresponding y values for the curve using the polynomial function
        y_curve = polynomial_func(x_curve)

        fig2 = plt.figure()
        plt.scatter(binning, means)

        fig1 = plt.figure()
        plt.scatter(binning, variances)
        plt.plot(x_curve, y_curve)
        plt.show()
        """

        return

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

        all_data = pd.read_csv('all_paths.csv', sep=' ', header=0,
                               names=['x', 'y', 'z', 'vx', 'vy', 'vz', 'L', 'alpha_I', 'beta_I', 'jacobi'])

        # omega = 1.993722232068e-7
        # time = 1 / omega
        # distance = 1.49461e8
        # mu = 3.040424e-6
        # mu_s = 1.3271244e11
        # mu_e = 3.986e3
        # mu_m = 4.902e3
        # mu_ems = mu_m + mu_e
        #
        # jacobis = []
        # for idx, row in all_data.iterrows():
        #     x = row['x'] * distance
        #     y = row['y'] * distance
        #     z = row['z'] * distance
        #     vx = row['vx'] * distance / time
        #     vy = row['vy'] * distance / time
        #     vz = row['vz'] * distance / time
        #     r_s = np.linalg.norm([row['x'] + mu, row['y'], row['z']]) * distance
        #     r_ems = np.linalg.norm([row['x'] - (1 - mu), row['y'], row['z']]) * distance
        #
        #     jacobis.append(
        #         omega ** 2 * (x ** 2 + y ** 2) + 2 * mu_s / r_s + 2 * mu_ems / r_ems - (vx ** 2 + vy ** 2 + vz ** 2))


        # all_data['Dimensional Jacobi'] = jacobis

        # all_data_26632 = all_data[all_data['Dimensional Jacobi'] >= 2663.2]
        # all_data_26630 = all_data[all_data['Dimensional Jacobi'] >= 2663]
        # all_data_26628 = all_data[all_data['Dimensional Jacobi'] >= 2662.8]
        # all_data_26626 = all_data[all_data['Dimensional Jacobi'] >= 2662.6]

        stc_pop_ini = self.population[self.population['STC'] == True]
        stc_pop_planar = stc_pop_ini[(abs(stc_pop_ini['Helio z at EMS']) < 0.0001) & (abs(stc_pop_ini['Helio vz at EMS']) < 0.00001)]
        stc_pop_planar_2 = stc_pop_planar[(abs(stc_pop_planar['Helio z at Capture']) < 0.0001) & (abs(stc_pop_planar['Helio vz at Capture']) < 0.00001)]
        stc_pop_planar_3 = stc_pop_planar_2[(abs(stc_pop_planar_2['Helio z at Release']) < 0.0001) & (
                    abs(stc_pop_planar_2['Helio vz at Release']) < 0.00001)]
        actual_planar_ids = ['NESC00000Opf', 'NESC00001xp6', 'NESC00003HO8', 'NESC00004Hzu', 'NESC00004m1B', 'NESC00004zBZ',
                             'NESC00009F39', 'NESC0000as6C', 'NESC0000AWYz', 'NESC0000BHG1', 'NESC0000CdOz', 'NESC0000dbfP',
                             'NESC0000dPxh', 'NESC0000dR3v', 'NESC0000ds7v', 'NESC0000dw0G', 'NESC0000eGj2', 'NESC0000EXSB',
                             'NESC0000m2AL', 'NESC0000nlWD', 'NESC0000qF2S', 'NESC0000u8R8', 'NESC0000wMjh', 'NESC0000yn24',
                             'NESC0000zHqv']
        final_planar = stc_pop_planar_3[stc_pop_planar_3['Object id'].isin(actual_planar_ids)]
        print(len(stc_pop_planar))
        print(stc_pop_planar['Object id'])
        print(len(stc_pop_planar_2))
        print(stc_pop_planar_2['Object id'])
        print(len(stc_pop_planar_3))
        print(stc_pop_planar_3['Object id'])
        print(stc_pop_planar['Non-Dimensional Jacobi'])
        # tco_pop = self.population[self.population['Became Minimoon'] == 1]
        # tcf_pop = self.population[self.population['Became Minimoon'] == 0]
        # stc_pop = stc_pop_ini[(stc_pop_ini['Dimensional Jacobi'] < 2720) & (stc_pop_ini['Dimensional Jacobi'] > 2663.2)]


        # all_alphas = stc_pop['Alpha_I'].tolist()
        # all_betas = stc_pop['Beta_I'].tolist()
        # all_jacobis = stc_pop['Non-Dimensional Jacobi'].tolist()
        # all_jacobisd = stc_pop['Dimensional Jacobi'].tolist()
        # all_data_jacobi = all_data_26632['Dimensional Jacobi'].tolist()
        # all_data_alphas = all_data_26632['alpha_I'].tolist()
        # all_data_betas = all_data_26632['beta_I'].tolist()
        #
        # plt.rcParams.update({'font.size': 24})

        # fig = plt.figure()
        # plt.scatter(all_data_alphas, all_data_betas, color='#ff00eb', s=15, label='Planar Lyapunov Orbits')
        # plt.scatter(all_alphas, all_betas, color='black', s=2, label='STCs Obtained Numerically')
        # plt.xlabel(r'$\alpha_I$ (degrees)')
        # plt.ylabel(r'$\beta_I$ (degrees)')
        # plt.title(r'$C_{SE} > 2663.2$ $km^2/s^2$')
        # plt.xlim([0, 360])
        # plt.ylim([-180, 0])
        # plt.legend()
        #
        # stc_pop2 = stc_pop_ini[(stc_pop_ini['Dimensional Jacobi'] < 2720) & (stc_pop_ini['Dimensional Jacobi'] > 2663)]
        #
        # all_alphas = stc_pop2['Alpha_I'].tolist()
        # all_betas = stc_pop2['Beta_I'].tolist()
        # all_data_alphas = all_data_26630['alpha_I'].tolist()
        # all_data_betas = all_data_26630['beta_I'].tolist()
        #
        # fig = plt.figure()
        # plt.scatter(all_data_alphas, all_data_betas, color='#0012ff', s=15, label='Planar Lyapunov Orbits')
        # plt.scatter(all_alphas, all_betas, color='black', s=2, label='STCs Obtained Numerically')
        # plt.xlabel(r'$\alpha_I$ (degrees)')
        # plt.ylabel(r'$\beta_I$ (degrees)')
        # plt.title(r'$C_{SE} > 2663$ $km^2/s^2$')
        # plt.xlim([0, 360])
        # plt.ylim([-180, 0])
        # plt.legend()
        #
        # stc_pop3= stc_pop_ini[(stc_pop_ini['Dimensional Jacobi'] < 2720) & (stc_pop_ini['Dimensional Jacobi'] > 2662.8)]
        #
        # all_alphas = stc_pop3['Alpha_I'].tolist()
        # all_betas = stc_pop3['Beta_I'].tolist()
        # all_data_alphas = all_data_26628['alpha_I'].tolist()
        # all_data_betas = all_data_26628['beta_I'].tolist()
        #
        # fig = plt.figure()
        # plt.scatter(all_data_alphas, all_data_betas, color='#00ffdd', s=15, label='Planar Lyapunov Orbits')
        # plt.scatter(all_alphas, all_betas, color='black', s=2, label='STCs Obtained Numerically')
        # plt.xlabel(r'$\alpha_I$ (degrees)')
        # plt.ylabel(r'$\beta_I$ (degrees)')
        # plt.title(r'$C_{SE} > 2662.8$ $km^2/s^2$')
        # plt.xlim([0, 360])
        # plt.ylim([-180, 0])
        # plt.legend()
        #
        # stc_pop4 = stc_pop_ini[
        #     (stc_pop_ini['Dimensional Jacobi'] < 2720) & (stc_pop_ini['Dimensional Jacobi'] > 2662.6)]
        #
        # all_alphas = stc_pop4['Alpha_I'].tolist()
        # all_betas = stc_pop4['Beta_I'].tolist()
        # all_data_alphas = all_data_26626['alpha_I'].tolist()
        # all_data_betas = all_data_26626['beta_I'].tolist()
        #
        # fig = plt.figure()
        # plt.scatter(all_data_alphas, all_data_betas, color='#51ff00', s=15, label='Planar Lyapunov Orbits')
        # plt.scatter(all_alphas, all_betas, color='black', s=2, label='STCs Obtained Numerically')
        # plt.xlabel(r'$\alpha_I$ (degrees)')
        # plt.ylabel(r'$\beta_I$ (degrees)')
        # plt.title(r'$C_{SE} > 2662.6$ $km^2/s^2$')
        # plt.xlim([0, 360])
        # plt.ylim([-180, 0])
        # plt.legend()
        #
        # plt.show()


        plt.rcParams.update({'font.size': 12})
        # index = master.index[master['Object id'] == name].tolist()
        # mm_flag = True if master.loc[index[0], "Became Minimoon"] == 1 else False

        # final_planar_good = final_planar[(final_planar['Non-Dimensional Jacobi'] > 2.9999) & (final_planar['Non-Dimensional Jacobi'] < 3.0009)]
        # for ix, row in final_planar.iterrows():
        #     if row['Non-Dimensional Jacobi'] > 3.0009:
        #         final_planar.loc[ix, 'Non-Dimensional Jacobi'] = 3.0009
        #     elif row['Non-Dimensional Jacobi'] < 2.9999:
        #         final_planar.loc[ix, 'Non-Dimensional Jacobi'] = 2.9999

        # print(final_planar['Non-Dimensional Jacobi'])
        # fig = plt.figure()
        # sc = plt.scatter(all_data['alpha_I'], all_data['beta_I'],
        #                   c=all_data['jacobi'], cmap='gist_rainbow', s=5)
        # sc2 = plt.scatter(final_planar['Alpha_I'], final_planar['Beta_I'], c=final_planar['Non-Dimensional Jacobi'], cmap='gist_rainbow', s=25, edgecolors='black')
        # cbar = fig.colorbar(sc, label='Jacobi Constant ($\emptyset$)')
        # plt.xlabel(r'$\alpha_I$ (degrees)')
        # plt.ylabel(r'$\beta_I$ (degrees)')
        # plt.xlim([0, 360])
        # plt.ylim([-180, 0])
        # plt.savefig("figures/stc_alpha_beta.svg", format="svg")
        # plt.savefig("figures/stc_alpha_beta.png", format="png")

        all_data_8 = all_data[all_data['jacobi'] > 3.0002]
        all_data_alphas = all_data_8['alpha_I']
        all_data_betas = all_data_8['beta_I']
        final_planar = final_planar[(final_planar['Non-Dimensional Jacobi'] > 3.0002) & (final_planar['Non-Dimensional Jacobi'] < 3.000997)]

        fig = plt.figure()
        plt.scatter(all_data_alphas, all_data_betas, color='#5cff00', s=15, label='Planar Lyapunov Orbits')
        plt.scatter(final_planar['Alpha_I'], final_planar['Beta_I'], color='black', s=25, label='Planar STCs')
        plt.xlabel(r'$\alpha_I$ (degrees)')
        plt.ylabel(r'$\beta_I$ (degrees)')
        plt.title(r'$C_{SE\emptyset} > 3.0002$')
        plt.xlim([0, 360])
        plt.ylim([-180, 0])
        plt.legend()


        plt.show()

        fig = plt.figure()
        # plt.scatter(tco_pop['3 Hill Duration'], tco_pop['Non-Dimensional Jacobi'], color='blue', s=5, label='TCOs', zorder=5)
        # plt.scatter(tcf_pop['3 Hill Duration'], tcf_pop['Non-Dimensional Jacobi'], color='orange', s=3, label='TCFs', zorder=10)
        plt.scatter(final_planar['3 Hill Duration'], final_planar['Non-Dimensional Jacobi'], color='red', s=1, label='STCs', zorder=15)
        plt.xlabel('Capture Duration (Days)')
        plt.ylabel('Non-Dimensional Jacobi Constant ($\emptyset$)')
        # plt.xlim([0, 0.004])
        # plt.ylim([2.998, 3.0015])
        plt.legend()
        plt.show()
        """
        fig = plt.figure()
        plt.scatter(tco_pop['Min. Distance'], tco_pop['Dimensional Jacobi'], color='blue', s=5, label='TCOs',
                    zorder=5)
        plt.scatter(tcf_pop['Min. Distance'], tcf_pop['Dimensional Jacobi'], color='orange', s=3, label='TCFs',
                    zorder=10)
        plt.scatter(stc_pop['Min. Distance'], stc_pop['Dimensional Jacobi'], color='red', s=1, label='STCs',
                    zorder=15)
        plt.xlabel('Min. Distance to Earth (AU)')
        plt.ylabel('Non-Dimensional Jacobi Constant')
        # plt.xlim([0, 0.004])
        # plt.ylim([2.998, 3.0015])
        plt.legend()
        # plt.savefig("figures/jacobi_dist.svg", format="svg")
        # plt.savefig("figures/jacobi_dist.png", format="png")

        fig = plt.figure()
        sc = plt.scatter(all_alphas, all_betas, c=all_jacobisd, cmap='gist_rainbow', s=5)
        cbar = fig.colorbar(sc, label='Jacobi Constant ($km^2/s^2$)')
        plt.xlabel(r'$\alpha_I$ (degrees)')
        plt.ylabel(r'$\beta_I$ (degrees)')
        # plt.xlim([0, 360])
        # plt.ylim([-180, 0])
        # plt.savefig("figures/stc_alpha_beta.svg", format="svg")
        # plt.savefig("figures/stc_alpha_beta.png", format="png")

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
        # plt.savefig("figures/tmes_hist.svg", format="svg")

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
        # plt.savefig("figures/t_hist.svg", format="svg")

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
        # plt.savefig("figures/peri_2.svg", format="svg")

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
        # plt.savefig("figures/peri_3.svg", format="svg")

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
        # plt.savefig("figures/peri_1.svg", format="svg")

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
        # plt.savefig("figures/peri_ems.svg", format="svg")

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
        # plt.savefig("figures/stc_at_ems.svg", format="svg")

        # Set the bin size
        bin_size = 1

        # Calculate the number of bins based on data range and bin size
        data_range_retrograde_stc = max(retrograde_stc['EMS Duration']) - min(retrograde_stc['EMS Duration'])
        num_bins_stc_retro = int(data_range_retrograde_stc / bin_size)
        data_range_prograde_stc = max(prograde_stc['EMS Duration']) - min(prograde_stc['EMS Duration'])
        num_bins_stc_pro = int(data_range_prograde_stc / bin_size)

        fig = plt.figure()
        plt.hist(retrograde_stc['EMS Duration'], bins=num_bins_stc_retro, label='Retrograde', edgecolor="#038cfc",
                 color="#03b1fc")
        plt.hist(prograde_stc['EMS Duration'], bins=num_bins_stc_pro, label='Prograde', edgecolor="#ed0000",
                 color="#f54747")
        plt.xlim([0, 140])
        plt.ylim([0, 500])
        plt.xlabel('Duration in the SOI of EMS (days)')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig("figures/tmes_hist_proretro.svg", format="svg")

        # Set the bin size
        bin_size = 10

        # Calculate the number of bins based on data range and bin size
        data_range_retrograde_stc = max(retrograde_stc['3 Hill Duration']) - min(retrograde_stc['3 Hill Duration'])
        num_bins_stc_retro = int(data_range_retrograde_stc / bin_size)
        data_range_prograde_stc = max(prograde_stc['3 Hill Duration']) - min(prograde_stc['3 Hill Duration'])
        num_bins_stc_pro = int(data_range_prograde_stc / bin_size)

        fig2 = plt.figure()
        plt.hist(retrograde_stc['3 Hill Duration'], bins=num_bins_stc_retro, label='Retrograde', edgecolor="#038cfc",
                 color="#03b1fc", zorder=5)
        plt.hist(prograde_stc['3 Hill Duration'], bins=num_bins_stc_pro, label='Prograde', edgecolor="#ed0000",
                 color="#f54747", zorder=10)
        plt.xlim([0, 1500])
        plt.ylim([0, 160])
        plt.xlabel('Capture Duration (days)')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig("figures/t_hist_proretro.svg", format="svg")

        string = 'Periapsides in 2 Hill'
        # Set the bin size
        bin_size = 1

        # Calculate the number of bins based on data range and bin size
        data_range_retrograde_stc = max(retrograde_stc[string]) - min(retrograde_stc[string])
        num_bins_stc_retro = int(data_range_retrograde_stc / bin_size)
        data_range_prograde_stc = max(prograde_stc[string]) - min(prograde_stc[string])
        num_bins_stc_pro = int(data_range_prograde_stc / bin_size)

        fig3 = plt.figure()
        plt.hist(retrograde_stc[string], bins=num_bins_stc_retro, label='Retrograde', edgecolor="#038cfc",
                 color="#03b1fc", zorder=5)
        plt.hist(prograde_stc[string], bins=num_bins_stc_pro, label='Prograde', edgecolor="#ed0000",
                 color="#f54747", zorder=10)
        plt.xlim([2, 20])
        # plt.xticks(range(2, 20))
        plt.ylim([0, 1800])
        fig3.set_size_inches(3.5, 5)
        plt.xlabel('Periapsides in Two Earth Hill Radii')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig("figures/peri_2_proretro.svg", format="svg")

        string = 'Periapsides in 3 Hill'
        # Set the bin size
        bin_size = 1

        # Calculate the number of bins based on data range and bin size
        data_range_retrograde_stc = max(retrograde_stc[string]) - min(retrograde_stc[string])
        num_bins_stc_retro = int(data_range_retrograde_stc / bin_size)
        data_range_prograde_stc = max(prograde_stc[string]) - min(prograde_stc[string])
        num_bins_stc_pro = int(data_range_prograde_stc / bin_size)

        fig4 = plt.figure()
        plt.hist(retrograde_stc[string], bins=num_bins_stc_retro, label='Retrograde', edgecolor="#038cfc",
                 color="#03b1fc", zorder=5)
        plt.hist(prograde_stc[string], bins=num_bins_stc_pro, label='Prograde', edgecolor="#ed0000",
                 color="#f54747", zorder=10)
        plt.xlim([2, 20])
        # plt.xticks(range(2, 20))
        plt.ylim([0, 1600])
        fig4.set_size_inches(3.5, 5)
        plt.xlabel('Periapsides in Three Earth Hill Radii')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig("figures/peri_3_proretro.svg", format="svg")

        string = 'Periapsides in 1 Hill'
        # Set the bin size
        bin_size = 1

        # Calculate the number of bins based on data range and bin size
        data_range_retrograde_stc = max(retrograde_stc[string]) - min(retrograde_stc[string])
        num_bins_stc_retro = int(data_range_retrograde_stc / bin_size)
        data_range_prograde_stc = max(prograde_stc[string]) - min(prograde_stc[string])
        num_bins_stc_pro = int(data_range_prograde_stc / bin_size)

        fig5 = plt.figure()
        plt.hist(retrograde_stc[string], bins=num_bins_stc_retro, label='Retrograde', edgecolor="#038cfc",
                 color="#03b1fc", zorder=5)
        plt.hist(prograde_stc[string], bins=num_bins_stc_pro, label='Prograde', edgecolor="#ed0000",
                 color="#f54747", zorder=10)
        plt.xlim([2, 9])
        plt.xticks(range(2, 9))
        plt.ylim([0, 2500])
        fig5.set_size_inches(3.5, 5)
        plt.xlabel('Periapsides in One Earth Hill Radius')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig("figures/peri_1_proretro.svg", format="svg")

        string = 'Periapsides in EMS'
        # Set the bin size
        bin_size = 1

        # Calculate the number of bins based on data range and bin size
        data_range_retrograde_stc = max(retrograde_stc[string]) - min(retrograde_stc[string])
        num_bins_stc_retro = int(data_range_retrograde_stc / bin_size)
        data_range_prograde_stc = max(prograde_stc[string]) - min(prograde_stc[string])
        num_bins_stc_pro = int(data_range_prograde_stc / bin_size)

        fig6 = plt.figure()
        plt.hist(retrograde_stc[string], bins=num_bins_stc_retro, label='Retrograde', edgecolor="#038cfc",
                 color="#03b1fc", zorder=5)
        plt.hist(prograde_stc[string], bins=num_bins_stc_pro, label='Prograde', edgecolor="#ed0000",
                 color="#f54747", zorder=10)
        plt.xlim([2, 9])
        plt.ylim([0, 3500])
        plt.xlabel('Periapsides in the EMS')
        plt.xticks(range(2, 9))
        fig6.set_size_inches(3.5, 5)
        plt.ylabel('Count')
        plt.legend()
        plt.savefig("figures/peri_ems_proretro.svg", format="svg")

        tco_pop = self.population[(self.population['Became Minimoon'] == 1)]
        tcf_pop = self.population[(self.population['Became Minimoon'] == 0)]

        string = '3 Hill Duration'

        # Set the bin size
        bin_size = 10

        # Calculate the number of bins based on data range and bin size
        data_range_stc = max(stc_pop['3 Hill Duration']) - min(stc_pop['3 Hill Duration'])
        num_bins_stc = int(data_range_stc / bin_size)
        data_range_tco = max(tco_pop['3 Hill Duration']) - min(tco_pop['3 Hill Duration'])
        num_bins_tco = int(data_range_tco / bin_size)
        data_range_tcf = max(tcf_pop['3 Hill Duration']) - min(tcf_pop['3 Hill Duration'])
        num_bins_tcf = int(data_range_tcf/ bin_size)

        fig2 = plt.figure()
        plt.hist(stc_pop['3 Hill Duration'], bins=num_bins_stc, label='STCs', edgecolor="#038cfc",
                 color="#03b1fc", zorder=10, alpha=0.5, )
        plt.hist(tco_pop['3 Hill Duration'], bins=num_bins_tco, label='TCOs', edgecolor="#ed0000",
                 color="#f54747", zorder=5)
        plt.hist(tcf_pop['3 Hill Duration'], bins=num_bins_tcf, label='TCFs', edgecolor="#ffad33",
                 color="#ffcc80", zorder=5)
        plt.xlim([0, 1500])
        plt.ylim([0, 500])
        plt.xlabel('Duration inside 3 Hill Radii (days)')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig("figures/t_hist_tcostctcf.svg", format="svg")

        print(len(stc_pop[stc_pop['Became Minimoon'] == 1]))
        print(len(stc_pop[stc_pop['Became Minimoon'] == 0]))
        """

        plt.show()

        return

    @staticmethod
    def no_moon_table(mm_pop, mm_pop_nomoon):

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

        for idx3, row3 in mm_pop.population.iterrows():

            object_id = row3['Object id']
            index = mm_pop_nomoon.index[mm_pop_nomoon['Object id'] == object_id].tolist()
            tco_occ_nomoon = mm_pop_nomoon.loc[index[0], 'Became Minimoon']
            tco_occ_moon = row3['Became Minimoon']

            if tco_occ_moon == 1 and tco_occ_nomoon == 1:
                stayed_tco += 1
            elif tco_occ_moon == 1 and tco_occ_nomoon == 0:
                tco_became_tcf += 1
            elif tco_occ_moon == 0 and tco_occ_nomoon == 0:
                stayed_tcf += 1
            else:
                tcf_became_tco += 1

        print("TCOs that remained TCOs: " + str(stayed_tco))
        print("TCFs that remained TCFs: " + str(stayed_tcf))
        print("TCOs that became TCFs: " + str(tco_became_tcf))
        print("TCFs that became TCOs: " + str(tcf_became_tco))

        stcstc = []
        stcnonstc = []
        nonstcstc = []
        nonstcnonstc = []

        for idx2, row2 in mm_pop.population.iterrows():

            limbo = mm_pop_nomoon[mm_pop_nomoon['Object id'] == row2['Object id']]
            if limbo['STC'].iloc[0] == True and row2['STC'] == True:  # if it was stc and remained stc
                stcstc.append(row2['Object id'])
            elif limbo['STC'].iloc[0] == False and row2['STC'] == True:  # if it was stc and became non-stc
                stcnonstc.append(row2['Object id'])
            elif limbo['STC'].iloc[0] == True and row2['STC'] == False:  # non stc became stc
                nonstcstc.append(row2['Object id'])
            else:
                nonstcnonstc.append(row2['Object id'])  # nonstc remained  stc

        stcstc_pop = mm_pop.population[mm_pop.population['Object id'].isin(stcstc)]
        stcnonstc_pop = mm_pop.population[mm_pop.population['Object id'].isin(stcnonstc)]
        nonstcstc_pop = mm_pop.population[mm_pop.population['Object id'].isin(nonstcstc)]
        nonstcnonstc_pop = mm_pop.population[mm_pop.population['Object id'].isin(nonstcnonstc)]

        print("STCs that remained STCs: " + str(len(stcstc_pop['Object id'])))
        print("Non STCs that remained non STCs: " + str(len(nonstcnonstc_pop['Object id'])))
        print("STCs that became non STCs: " + str(len(stcnonstc_pop['Object id'])))
        print("non STCs that became STCs: " + str(len(nonstcstc_pop['Object id'])))

        return stcstc_pop, stcnonstc_pop, nonstcnonstc_pop, nonstcstc_pop

    @staticmethod
    def planar_stc(mm_pop, mm_pop_nomoon, path_moon, path_nomoon):

        mm_parser = MmParser("", "", "")

        # Examining the planar population of STCs
        actual_planar_ids = ['NESC00000Opf', 'NESC00001xp6', 'NESC00003HO8', 'NESC00004Hzu', 'NESC00004m1B',
                             'NESC00004zBZ',
                             'NESC00009F39', 'NESC0000as6C', 'NESC0000AWYz', 'NESC0000BHG1', 'NESC0000CdOz',
                             'NESC0000dbfP', 'NESC0000dR3v', 'NESC0000ds7v', 'NESC0000dw0G', 'NESC0000eGj2',
                             'NESC0000EXSB',
                             'NESC0000m2AL', 'NESC0000nlWD', 'NESC0000qF2S', 'NESC0000u8R8', 'NESC0000wMjh',
                             'NESC0000yn24',
                             'NESC0000zHqv']  # removed 'NESC0000dPxh' because it impacts Earth

        planar_stc = mm_pop.population[mm_pop.population['Object id'].isin(actual_planar_ids)]
        planar_stc_nomoon = mm_pop_nomoon[mm_pop_nomoon['Object id'].isin(actual_planar_ids)]

        planar_stcstc = []
        planar_stcnonstc = []

        for idx2, row2 in planar_stc.iterrows():

            limbo = planar_stc_nomoon[planar_stc_nomoon['Object id'] == row2['Object id']]
            if limbo['STC'].iloc[0] == True:  # if it was stc and remained stc
                planar_stcstc.append(row2['Object id'])
            elif limbo['STC'].iloc[0] == False:  # if it was stc and became non-stc
                planar_stcnonstc.append(row2['Object id'])

        stc_stayed_stc_planar = planar_stc[planar_stc['Object id'].isin(planar_stcstc)]
        stc_became_nonstc_planar = planar_stc[planar_stc['Object id'].isin(planar_stcnonstc)]

        print("Planar STC that remained STC: " + str(len(stc_stayed_stc_planar['Object id'])))
        print("STC that became non-STC: " + str(len(stc_became_nonstc_planar['Object id'])))

        fig = plt.figure()
        plt.scatter(stc_became_nonstc_planar['3 Hill Duration'], stc_became_nonstc_planar['Non-Dimensional Jacobi'],
                    s=1, color='red', label='Became Non-STC without influence of the Moon')
        plt.scatter(stc_stayed_stc_planar['3 Hill Duration'], stc_stayed_stc_planar['Non-Dimensional Jacobi'], s=1,
                    color='blue', label='Remained STC without influence of the Moon')
        plt.plot(np.linspace(0, 1400, 200), 2.9999 * np.linspace(1, 1, 200), linestyle='--', color='green', linewidth=1)
        plt.xlabel('Capture Duration (days)')
        plt.ylabel('Jacobi Constant ($\emptyset$)')
        plt.xlim([0, 1400])
        plt.ylim([2.9985, 3.0015])
        plt.legend()

        # fig = plt.figure()
        # plt.scatter(stc_stayed_stc_planar['3 Hill Duration'], stc_stayed_stc_planar['Non-Dimensional Jacobi'], s=1,
        #             color='blue')
        # plt.plot(np.linspace(0, 1400, 200), 2.9999 * np.linspace(1, 1, 200), linestyle='--',
        #          color='green', linewidth=1)
        # plt.xlabel('Capture Duration (days)')
        # plt.ylabel('Jacobi Constant ($\emptyset$)')
        # plt.xlim([0, 1400])
        # plt.ylim([2.9985, 3.0015])

        plt.show()

        # planar population visualization
        for idx, row in planar_stc.iterrows():
            # print(row)

            stc_name = str(row['Object id']) + '.csv'
            index = mm_pop_nomoon.index[mm_pop_nomoon['Object id'] == row['Object id']].tolist()
            stc_name_no_moon = mm_pop_nomoon.loc[index[0], 'Object id']
            stc_occ = mm_pop_nomoon.loc[index[0], 'STC']

            print("Examining Previously STC: " + stc_name)
            print("Still STC?: " + str(stc_occ) + " for STC: " + str(stc_name_no_moon))
            print("Jacobi constant: " + str(row['Non-Dimensional Jacobi']))
            data_nomoon = mm_parser.mm_file_parse_new(path_nomoon + '/' + stc_name)
            data_moon = mm_parser.mm_file_parse_new(path_moon + '/' + stc_name)

            fig3 = plt.figure()
            ax = fig3.add_subplot(111, projection='3d')
            vel_scale = 1
            ut, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
            xw = 0.0038752837677 * np.cos(ut) * np.sin(v)
            yw = 0.0038752837677 * np.sin(ut) * np.sin(v)
            zw = 0.0038752837677 * np.cos(v)
            ax.plot_wireframe(xw, yw, zw, color="b", alpha=0.1)
            ax.scatter(0, 0, 0, color='blue', s=10)
            ax.plot3D(data_moon['Synodic x'], data_moon['Synodic y'], data_moon['Synodic z'], color='grey', zorder=15,
                      linewidth=1, label='With Moon')
            ax.plot3D(data_nomoon['Synodic x'], data_nomoon['Synodic y'], data_nomoon['Synodic z'], color='orange',
                      zorder=10, linewidth=3, label='Without Moon')
            ax.set_xlabel('Synodic x (AU)')
            ax.set_ylabel('Synodic y (AU)')
            ax.set_zlabel('Synodic z (AU)')
            ax.set_xlim([-0.01, 0.01])
            ax.set_ylim([-0.01, 0.01])
            ax.set_zlim([-0.01, 0.01])
            ax.set_title('STC ' + str(stc_name_no_moon) + '\nJacobi Constant = ' + str(round(row['Non-Dimensional Jacobi'], 5)))
            num_ticks = 3
            ax.xaxis.set_major_locator(ticker.MaxNLocator(num_ticks))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(num_ticks))
            ax.zaxis.set_major_locator(ticker.MaxNLocator(num_ticks))
            ax.legend()
            # plt.savefig('figures/' + stc_name_no_moon + '_nomoon.svg', format='svg')
            # plt.savefig('figures/' + stc_name_no_moon + '_nomoon.png', format='png')
            plt.show()

        return stc_stayed_stc_planar, stc_became_nonstc_planar

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





