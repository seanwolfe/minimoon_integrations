import pandas as pd
import os
import matplotlib.pyplot as plt
from space_fncs import eci_ecliptic_to_sunearth_synodic
import numpy as np
from astropy import constants as const
from MmAnalyzer import MmAnalyzer
from MM_Parser import MmParser
import astropy.units as u

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
        tco_pop = mm_pop.population[(mm_pop.population['Became Minimoon'] == 1)]
        retrograde_pop = mm_pop.population[
            (mm_pop.population['Retrograde'] == 1) & (mm_pop.population['Became Minimoon']
                                                      == 1)]
        prograde_pop = mm_pop.population[(mm_pop.population['Retrograde'] == 0) & (mm_pop.population['Became Minimoon']
                                                                                   == 1)]

        fig = plt.figure()
        plt.hist(retrograde_pop['Capture Duration'], bins=1000, label='Retrograde', edgecolor="#038cfc", color="#03b1fc")
        plt.hist(prograde_pop['Capture Duration'], bins=1000, label='Prograde', edgecolor="#ed0000", color="#f54747")
        plt.xlim([0, 1000])
        plt.ylim([0, 700])
        plt.xlabel('Capture Duration (days)')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig("figures/f1.svg", format="svg")

        fig2 = plt.figure()
        plt.hist(abs(retrograde_pop['Number of Rev']), bins=350*20, label='Retrograde', edgecolor="#038cfc", color="#03b1fc")
        plt.hist(abs(prograde_pop['Number of Rev']), bins=350*20, label='Prograde', edgecolor="#ed0000", color="#f54747")
        plt.xlim([1, 5])
        plt.ylim([0,1600])
        plt.xlabel('Number of Revolutions')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig("figures/f2.svg", format="svg")

        fig3 = plt.figure()
        plt.scatter(tco_pop['Geo x at Capture'], tco_pop['Geo y at Capture'], s=0.1)
        c1 = plt.Circle((0, 0), radius=0.01, alpha=0.1)
        plt.gca().add_artist(c1)
        plt.gca().set_aspect('equal')
        plt.xlabel('Geocentric x at Capture (AU)')
        plt.ylabel('Geocentric y at Capture (AU)')
        plt.savefig("figures/f3.svg", format="svg")

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
        plt.savefig("figures/f4.svg", format="svg")

        fig5 = plt.figure()
        d = np.sqrt(tco_pop["Geo x at Capture"] ** 2 + tco_pop["Geo y at Capture"] ** 2
                    + tco_pop["Geo z at Capture"] ** 2)
        plt.xlim([0,500])
        plt.ylim([0,0.03])
        plt.scatter(tco_pop['Capture Duration'], d, s=1)
        plt.xlabel('Capture Duration (days)')
        plt.ylabel('Geocentric Distance at Capture (AU)')
        plt.savefig("figures/f5.svg", format="svg")

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
        plt.savefig("figures/f6.svg", format="svg")

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
        plt.savefig("figures/f7.svg", format="svg")

        fig8 = plt.figure()
        plt.scatter(tco_pop["X at Earth Hill"], tco_pop["Y at Earth Hill"], s=0.1)
        c1 = plt.Circle((0, 0), radius=0.01, alpha=0.1)
        plt.gca().add_artist(c1)
        plt.gca().set_aspect('equal')
        plt.xlabel('Synodic x at Earth Hill sphere (AU)')
        plt.ylabel('Synodic y at Earth Hill sphere (AU)')
        plt.savefig("figures/f8.svg", format="svg")


if __name__ == '__main__':

    population_file = 'minimoon_master_final.csv'
    # population_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'minimoon_integrations', 'minimoon_files_oorb')
    population_dir = os.path.join(os.path.expanduser('~'), 'Documents', 'sean', 'minimoon_integrations',
                                  'minimoon_files_oorb')
    population_path = population_dir + '/' + population_file

    mm_analyzer = MmAnalyzer()
    mm_parser = MmParser("", population_dir, "")
    mm_pop = MmPopulation(population_path)

    # Retrograde vs. Prograde
    # go through all the minimoon file names
    # zero for prograde, one for retrograde
    # retrograde = np.zeros([len(mm_pop.population['Object id']), 1])
    # max_dist = np.zeros([len(mm_pop.population['Object id']), 1])
    # mm_flag = np.zeros([len(mm_pop.population['Object id']), 1])
    # capture_idx = np.zeros([len(mm_pop.population['Object id']), 1],  dtype=np.int64)
    # release_idx = np.zeros([len(mm_pop.population['Object id']), 1],  dtype=np.int64)
    # x_ehs = np.zeros([len(mm_pop.population['Object id']), 1])
    # y_ehs = np.zeros([len(mm_pop.population['Object id']), 1])
    # z_ehs = np.zeros([len(mm_pop.population['Object id']), 1])
    # designations = []
    # hill_crossing = np.zeros([len(mm_pop.population['Object id']), 1])
    #
    master = mm_pop.population

    print(master)
    print(len(master["Object id"]))

    tco_pop = master[(master['Became Minimoon'] == 1)]
    print(len(tco_pop["Object id"]))
    retrograde_pop = master[(master['Retrograde'] == 1) & (master['Became Minimoon'] == 1)]
    prograde_pop = master[(master['Retrograde'] == 0) & (master['Became Minimoon'] == 1)]
    tcf_pop = master[(master['Became Minimoon'] == 0)]
    print(len(tcf_pop["Object id"]))
    print(len(tcf_pop[tcf_pop['3 Hill Duration'] == 0]))
    print(len(tcf_pop[tcf_pop['1 Hill Duration'] == 0]))
    print(len(tcf_pop[tcf_pop['Spec. En. Duration'] == 0]))
    print(len(tcf_pop[abs(tcf_pop['Number of Rev']) < 1]))

    for idx in range(len(master['Object id'])):
        idx = idx + 1
        print(idx)

        # go through all the files of test particles
        for root, dirs, files in os.walk(population_dir):
        #     find files that are minimoons
            name = str(tco_pop['Object id'].iloc[idx]) + ".csv"

            if name in files:
                file_path = os.path.join(root, name)


                # read the file
                header = 0
                data = mm_parser.mm_file_parse_new(file_path, 0)

                # Important constants
                # Constants
                mu = const.GM_earth.value  # Nominal Earth mass parameter (m3/s2)
                aupd = u.AU / u.d  # AU per day
                mps = u.m / u.s  # Meters per second
                #
                steps = len(data["Geo vx"])
                # State vector components of the minimoon with respect to earth
                vx = np.array([(data["Geo vx"].iloc[i] * aupd).to(mps) / mps for i in range(0, steps)])
                vy = np.array([(data["Geo vy"].iloc[i] * aupd).to(mps) / mps for i in range(0, steps)])
                vz = np.array([(data["Geo vz"].iloc[i] * aupd).to(mps) / mps for i in range(0, steps)])
                r = np.array([(data["Distance"].iloc[i] * u.AU).to(u.m) / u.m for i in range(0, steps)])

                eps = (0.5 * (vx ** 2 + vy ** 2 + vz ** 2) - mu/r) / 1e6  # to put in km^2/s^2

                fig = plt.figure(idx)
                zero = np.zeros((steps))
                plt.plot(data["Julian Date"] - data["Julian Date"].iloc[0], eps, color="blue")
                plt.plot(data["Julian Date"] - data["Julian Date"].iloc[0], zero, linestyle="--", color='green')
                plt.xlabel('Time (Days)')
                plt.ylabel('Specific Energy $(km^2/s^2)$')
                plt.xlim([0,1200])
                # plt.ylim([-0.15, 0.7])
                # plt.savefig("figures/f13.svg", format="svg")
                # plt.show()
                pd.set_option('display.max_rows', None)
                print(tco_pop.iloc[idx])
                # fig = plt.figure(idx)
                # plt.plot(data["Synodic x"], data["Synodic y"], color='grey', label='TCF Trajectory')
                # c1 = plt.Circle((0, 0), radius=0.01, alpha=0.1, label='Earth Hill Sphere')
                # c2 = plt.Circle((0, 0), radius=4.504075e-5, color='blue', label='Earth')
                # x_moon = data["Moon x (Helio)"] - data["Earth x (Helio)"]
                # y_moon = data["Moon y (Helio)"] - data["Earth y (Helio)"]
                # z_moon = data["Moon z (Helio)"] - data["Earth z (Helio)"]
                # plt.plot(x_moon, y_moon, color = 'red', label='Moon Orbit')
                # plt.gca().add_artist(c1)
                # plt.gca().add_artist(c2)
                # plt.xlabel('Synodic x (AU)')
                # plt.ylabel('Synodic y (AU)')
                # plt.legend()
                # plt.xlim([-0.015, 0.015])
                # plt.ylim([-0.015, 0.015])
                # plt.gca().set_aspect('equal')
                # plt.savefig("figures/f10.svg", format="svg")

                data_cap = data.iloc[int(master['Capture Index'].iloc[idx]):int(master['Release Index'].iloc[idx])]
                trans_cap = np.array([data_cap["Synodic x"], data_cap["Synodic y"], data_cap["Synodic z"]])
                fig2 = plt.figure(idx+1)
                plt.plot(trans_cap[0, :], trans_cap[1, :], color='#5599ff', linewidth=5, label='Period of Capture')
                plt.plot(data["Synodic x"], data["Synodic y"], 'gray', linewidth=2, label='TCO Trajectory')
                plt.scatter(trans_cap[0, 0], trans_cap[1, 0], color='#e9afaf', linewidth=3, label='Start',
                                        zorder=5)
                plt.scatter(trans_cap[0, -1], trans_cap[1, -1], color='#afe9af', linewidth=3, label='End',
                                        zorder=5)
                c1 = plt.Circle((0, 0), radius=0.01, alpha=0.1, label='Earth Hill Sphere')
                c2 = plt.Circle((0, 0), radius=4.504075e-5, color='blue', label='Earth')
                x_moon = data["Moon x (Helio)"] - data["Earth x (Helio)"]
                y_moon = data["Moon y (Helio)"] - data["Earth y (Helio)"]
                z_moon = data["Moon z (Helio)"] - data["Earth z (Helio)"]
                plt.plot(x_moon, y_moon, color='red', label='Moon Orbit')
                plt.gca().add_artist(c1)
                plt.gca().add_artist(c2)
                plt.xlabel('Synodic x (AU)')
                plt.ylabel('Synodic y (AU)')
                plt.legend()
                plt.xlim([-0.03, 0.03])
                plt.ylim([-0.05, 0.05])
                plt.gca().set_aspect('equal')
                # plt.savefig("figures/f14.svg", format="svg")

                # fig3 = plt.figure(idx + 2)
                # plt.plot(trans_cap[0, :], trans_cap[2, :], color='#5599ff', linewidth=5, label='Period of Capture')
                # plt.plot(data["Synodic x"], data["Synodic z"], 'gray', linewidth=2, label='TCO Trajectory')
                # plt.scatter(trans_cap[0, 0], trans_cap[2, 0], color='#e9afaf', linewidth=3, label='Start',
                #             zorder=5)
                # plt.scatter(trans_cap[0, -1], trans_cap[2, -1], color='#afe9af', linewidth=3, label='End',
                #             zorder=5)
                # c1 = plt.Circle((0, 0), radius=0.01, alpha=0.1, label='Earth Hill Sphere')
                # c2 = plt.Circle((0, 0), radius=4.504075e-5, color='blue', label='Earth')
                # x_moon = data["Moon x (Helio)"] - data["Earth x (Helio)"]
                # y_moon = data["Moon y (Helio)"] - data["Earth y (Helio)"]
                # z_moon = data["Moon z (Helio)"] - data["Earth z (Helio)"]
                # plt.plot(x_moon, z_moon, color='red', label='Moon Orbit')
                # plt.gca().add_artist(c1)
                # plt.gca().add_artist(c2)
                # plt.xlabel('Synodic x (AU)')
                # plt.ylabel('Synodic z (AU)')
                # plt.legend()
                # plt.xlim([-0.03, 0.03])
                # plt.ylim(([-0.01, 0.01]))
                # plt.gca().set_aspect('equal')
                # plt.savefig("figures/f12.svg", format="svg")

                plt.show()

                # distance = data['Distance']
                # eh_crossing = min(distance, key=lambda x: abs(x-0.01))
                # data_eh_crossing = data[distance == eh_crossing]
                #
                # compute revs
                # mu_e = const.GM_earth.value
                # mm_analyzer = MmAnalyzer()
                # mm_analyzer.minimoon_check(data, mu_e)

                # retrograde[idx, 0] = mm_analyzer.retrograde
                # max_dist[idx, 0] = mm_analyzer.max_dist
                # mm_flag[idx, 0] = mm_analyzer.minimoon_flag
                # capture_idx[idx, 0] = int(mm_analyzer.cap_idx)
                # release_idx[idx, 0] = int(mm_analyzer.rel_idx)
                # x_ehs[idx, 0] = data_eh_crossing['Synodic x'].values
                # y_ehs[idx, 0] = data_eh_crossing['Synodic y'].values
                # z_ehs[idx, 0] = data_eh_crossing['Synodic z'].values
        #
        # mm_pop.population["Retrograde"] = retrograde[:, 0]
        # mm_pop.population["Became Minimoon"] = mm_flag[:, 0]
        # mm_pop.population["Max. Distance"] = max_dist[:, 0]
        # mm_pop.population["Capture Index"] = capture_idx[:, 0]
        # mm_pop.population["Release Index"] = release_idx[:, 0]
        # mm_pop.population["X at Earth Hill"] = x_ehs[:, 0]
        # mm_pop.population["Y at Earth Hill"] = y_ehs[:, 0]
        # mm_pop.population["Z at Earth Hill"] = z_ehs[:, 0]
        #
        # destination_file = 'minimoon_master_final.csv'
        # destination_path = population_dir + '/' + destination_file
        # mm_pop.population.to_csv(destination_path, sep=' ', header=True, index=False)

    # for idx in range(len(mm_pop.population['Object id'])):
    #
    #     print(idx)
    #
        # go through all the files of test particles
        # for root, dirs, files in os.walk(population_dir):
        #     find files that are minimoons
            # name = str(mm_pop.population['Object id'].iloc[idx]) + ".csv"
            # if name in files:
            #     file_path = os.path.join(root, name)
                # read the file
                # header = 0
                # data = mm_parser.mm_file_parse_new(file_path, 0)
                # designation = mm_analyzer.taxonomy(data, mm_pop.population)
                # designations.append(designation)
    #
    # tax = mm_pop.population["Taxonomy"]
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.float_format', lambda x: '%.5f' % x)
    # tax[(tax == '2A')] = '3A'
    # tax[(tax == '2B')] = '2A'
    # tax[(tax == '3A')] = '2B'
    # mm_pop.population["Taxonomy"] = tax
    # print(mm_pop.population)


    # destination_file = 'minimoon_master_final.csv'
    # destination_path = population_dir + '/' + destination_file
    # mm_pop.population.to_csv(destination_path, sep=' ', header=True, index=False)

    master = mm_pop.population

    # Get retrograde and prograde TCOs
    tco_pop = master[(master['Became Minimoon'] == 1)]
    retrograde_pop = master[(master['Retrograde'] == 1) & (master['Became Minimoon'] == 1)]
    prograde_pop = master[(master['Retrograde'] == 0) & (master['Became Minimoon'] == 1)]

    pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.float_format', lambda x: '%.5f' % x)
    # print(tco_pop[tco_pop['Max. Distance'] > 0.03])

    # wrong_rev = tco_pop[(tco_pop["Number of Rev"] < 1)]
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.float_format', lambda x: '%.5f' % x)
    # print(wrong_rev)

    # Make taxonomy table
    mm_pop.tax_table(tco_pop, retrograde_pop, prograde_pop)

    # Generate visualization
    mm_pop.pop_viz()

    plt.show()



