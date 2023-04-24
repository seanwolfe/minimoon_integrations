"""
This class is used to parse through the data that Dr. Fedorets provided
"""
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
from MmAnalyzer import MmAnalyzer
from astropy import constants as const
import astropy.units as u
import numpy
from space_fncs import eci_ecliptic_to_sunearth_synodic


class MmParser:
    mm_data = []  # where the final dataframe containing the information about where the synthetic minimoons are located
    mm_file_paths = []  # a dataframe that contains the location of all finals that are minimoons in the original data
    # given by Dr. Fedorets

    def __init__(self, minimoon_file_path, top_directory, dest_directory):
        """
        Constructor
        :param minimoon_file_path: Takes the file path that contains the names of all files that represent minimoons
        :param top_directory: Takes the top level directory where the all the files of test particles is located
        :param dest_directory: This is the directory where all the minimoon files will be put
        """
        self.mmfile = minimoon_file_path
        self.mmdir = top_directory
        self.mmdir_final = dest_directory

        return

    def organize_data(self):
        """
        used to organize the data present in the minimoon file
        :return: Returns a data frame of the minimoon_file for ease of use, where file names are the first column of data
        data frame column names: "File Name", "Element Type", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9",
        "x10", "x11", "Integrator"

        """

        # read file into dataframe using panda
        self.mm_data = pd.read_csv(self.mmfile, sep=" ", header=None, names=["File Name", "Element Type", "x1", "x2",
                                                                             "x3", "x4", "x5", "x6", "x7", "x8", "x9",
                                                                             "x10", "x11", "Integrator"])

        return

    def fetch_files(self):
        """
        This function fetches all relevant minimoon files in the top level directory according to the names in the
        dataframe from the organize data function. Puts copies of the files into a new directory
        :return:
        """
        result = []
        # go through all the minimoon file names
        for name in self.mm_data['File Name']:
            # go through all the files of test particles
            for root, dirs, files in os.walk(self.mmdir):
                # find files that are minimoons
                if (str(name) + '.dat') in files:
                    name_temp = str(name) + '.dat'
                    file_path = os.path.join(root, name_temp)
                    # store result
                    result.append(file_path)

                    # copy file to new directory
                    shutil.copy(file_path, self.mmdir_final)

                    print(name)

        # convert result to dataframe
        self.mm_file_paths = pd.DataFrame({'Minimoon File Paths': result})
        print(self.mm_file_paths)

        return

    @staticmethod
    def mm_file_parse(file_path):
        """
        This function parses a single minimoon file into a appropriately labelled dataframe, column names:
        "Object id", "Julian Date", "Distance", "Helio q", "Helio e", "Helio i", "Helio Omega ", "Helio omega",
        "Helio M", "Helio x", "Helio y", "Helio z", "Helio vx", "Helio vy", "Helio vz", "Geo x", "Geo y", "Geo z",
        "Geo vx", "Geo vy", "Geo vz", "Geo q", "Geo e", "Geo i", "Geo Omega", "Geo omega", "Geo M", "Earth x (Helio)",
        "Earth y (Helio)", "Earth z (Helio)", "Earth vx (Helio)", "Earth vy (Helio)", "Earth vz (Helio)",
        "Moon x (Helio)", "Moon y (Helio)", "Moon z (Helio)", "Moon vx (Helio)", "Moon vy (Helio)",
        "Moon vz (Helio)"
        :return: the data frame

        """

        data = pd.read_csv(file_path,  sep=" ", header=None, names=["Object id", "Julian Date", "Distance", "Helio q",
        "Helio e", "Helio i", "Helio Omega ", "Helio omega", "Helio M", "Helio x", "Helio y", "Helio z", "Helio vx",
        "Helio vy", "Helio vz", "Geo x", "Geo y", "Geo z", "Geo vx", "Geo vy", "Geo vz", "Geo q", "Geo e", "Geo i",
        "Geo Omega", "Geo omega", "Geo M", "Earth x (Helio)", "Earth y (Helio)", "Earth z (Helio)", "Earth vx (Helio)",
        "Earth vy (Helio)", "Earth vz (Helio)", "Moon x (Helio)", "Moon y (Helio)", "Moon z (Helio)", "Moon vx (Helio)",
        "Moon vy (Helio)", "Moon vz (Helio)"])

        return data


if __name__ == '__main__':

    # Constants
    mu_e = const.GM_earth.value  # Nominal Earth mass parameter (m3/s2)

    mm_master_file_name = 'NESCv9reintv1.TCO.withH.kep.des'  # name of the minimoon file to parsed

    # name of the directory where the mm file is located,
    # also top level directory where all the integration data is located
    mm_file_dir = os.path.join(os.path.expanduser('~'), 'OneDrive', 'Thesis', 'Minimoon_Integrations', 'minimoon_data')

    mm_file_path = mm_file_dir + '/' + mm_master_file_name  # path for the minimoon file
    destination_path = os.path.join(os.path.expanduser('~'), 'Documents', 'sean', 'minimoon_integrations',
                                    'minimoon_files')

    # create parser
    mm_parser = MmParser(mm_file_path, mm_file_dir, destination_path)

    # organize the data in the minimoon_file
    mm_parser.organize_data()

    # fetch all the files from all the folders within the top level directory
    # mm_parser.fetch_files()

    # Fetch a single file
    for id in mm_parser.mm_data["File Name"]:

        # Grab data from file
        mm_file_name = id + ".dat"
        test_file = destination_path + '/' + mm_file_name
        data = mm_parser.mm_file_parse(test_file)
        print(data["Geo x"].iloc[-1])
        MmAnalyzer.minimoon_check(data, mu_e)

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(data["Geo x"], data["Geo y"], data["Geo z"], 'green', linewidth=2, label='Minimoon ' + str(id))
        x_moon = data["Moon x (Helio)"] - data["Earth x (Helio)"]
        y_moon = data["Moon y (Helio)"] - data["Earth y (Helio)"]
        z_moon = data["Moon z (Helio)"] - data["Earth z (Helio)"]
        ax.plot3D(x_moon, y_moon, z_moon, 'red', label='Moon')
        ax.scatter3D(data["Geo x"].iloc[0], data["Geo y"].iloc[0], data["Geo z"].iloc[0], 'orange', linewidth=5,
                  label='Start')
        ax.scatter3D(data["Geo x"].iloc[-1], data["Geo y"].iloc[-1], data["Geo z"].iloc[-1], 'black', linewidth=5,
                  label='End')

        # draw Earth
        R_E = (6378 * u.km).to(u.AU) / u.AU
        u_E, v_E = numpy.mgrid[0:2 * numpy.pi:50j, 0:numpy.pi:50j]
        x_E = R_E * numpy.cos(u_E) * numpy.sin(v_E)
        y_E = R_E * numpy.sin(u_E) * numpy.sin(v_E)
        z_E = R_E * numpy.cos(v_E)
        # alpha controls opacity
        ax.plot_surface(x_E, y_E, z_E, color="b", alpha=1)
        leg = ax.legend(loc='best')
        ax.set_xlabel('x (AU)')
        ax.set_ylabel('y (AU)')
        ax.set_zlabel('z (AU)')

        # get the ecliptic longitude of the minimoon (to count rotations)
        # first transform to synodic frame
        earth_xyz = data[["Earth x (Helio)", "Earth y (Helio)", "Earth z (Helio)"]].T.values
        mm_xyz = data[["Geo x", "Geo y", "Geo z"]].T.values
        trans_xyz = eci_ecliptic_to_sunearth_synodic(-earth_xyz, mm_xyz)  # minus is to have sun relative
        fig = plt.figure()
        plt.scatter(trans_xyz[0, 0], trans_xyz[1, 0], color='orange', linewidth=5, label='Start')
        plt.scatter(trans_xyz[0, -1], trans_xyz[1, -1], color='black', linewidth=5, label='End')
        plt.plot(trans_xyz[0, :], trans_xyz[1, :], 'green', linewidth=2)
        plt.plot(x_E, y_E, 'blue', alpha=1)

        plt.xlabel('X (AU)')
        plt.ylabel('Y (AU)')
        plt.legend()
        plt.title('Minimoon ' + str(id) + ' Trajectory in Synodic Frame')

        plt.show()





