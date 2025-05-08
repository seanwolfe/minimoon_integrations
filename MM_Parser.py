import pandas as pd
import os
import shutil


class MmParser:
    """
    This class is used to parse through various data files
    """
    mm_data = []  # where the final dataframe containing the information about where the synthetic minimoons are located
    mm_file_paths = []  # a dataframe that contains the location of all finals that are minimoons in the original data

    # given by Dr. Fedorets

    def __init__(self, minimoon_file_path, top_directory, dest_directory):
        """
        Constructor
        :param minimoon_file_path: Takes the file path that contains the names of all files that represent minimoons,
        referred to as master is some cases
        :param top_directory: Takes the top level directory where the all the files of test particles is located
        :param dest_directory: This is the directory where all the minimoon files will be put
        """
        self.mmfile = minimoon_file_path
        self.mmdir = top_directory
        self.mmdir_final = dest_directory

        return

    def organize_data(self):
        """
        used to organize the data present in the minimoon file NESCv9reintv1.TCO.withH.kep.des by Dr. Fedorets
        :return: Returns a data frame of the minimoon_file for ease of use, where file names are the first column of data
        data frame column names: "File Name", "Element Type", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9",
        "x10", "x11", "Integrator", xs because what the values represent where not specified, however x7 seems to be H

        """

        # read file into dataframe using panda
        self.mm_data = pd.read_csv(self.mmfile, sep=" ", header=None, names=["File Name", "Element Type", "x1", "x2",
                                                                             "x3", "x4", "x5", "x6", "x7", "x8", "x9",
                                                                             "x10", "x11", "Integrator"])

        return

    def fetch_files(self):
        """
        This function fetches all relevant minimoon files in the top level directory according to the names in the
        dataframe from the organize data function. Puts copies of the files into a new directory specified in the
        initializations. This function can be used to find all the minimoon files in the data given by Dr. Fedorets files,
        where some of the files where irrelavent
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
        Units are au, au/day, degrees
        This function parses a single minimoon file into a appropriately labelled dataframe, the minimoon file is
        from Dr. Fedorets file, a 39-column array with names that we give it:
        "Object id", "Julian Date", "Distance", "Helio q", "Helio e", "Helio i", "Helio Omega", "Helio omega",
        "Helio M", "Helio x", "Helio y", "Helio z", "Helio vx", "Helio vy", "Helio vz", "Geo x", "Geo y", "Geo z",
        "Geo vx", "Geo vy", "Geo vz", "Geo q", "Geo e", "Geo i", "Geo Omega", "Geo omega", "Geo M", "Earth x (Helio)",
        "Earth y (Helio)", "Earth z (Helio)", "Earth vx (Helio)", "Earth vy (Helio)", "Earth vz (Helio)",
        "Moon x (Helio)", "Moon y (Helio)", "Moon z (Helio)", "Moon vx (Helio)", "Moon vy (Helio)",
        "Moon vz (Helio)"
        :return: the data frame

        """

        data = pd.read_csv(file_path, sep=" ", header=None, names=["Object id", "Julian Date", "Distance", "Helio q",
                                                                   "Helio e", "Helio i", "Helio Omega", "Helio omega",
                                                                   "Helio M", "Helio x", "Helio y", "Helio z",
                                                                   "Helio vx",
                                                                   "Helio vy", "Helio vz", "Geo x", "Geo y", "Geo z",
                                                                   "Geo vx", "Geo vy", "Geo vz", "Geo q", "Geo e",
                                                                   "Geo i",
                                                                   "Geo Omega", "Geo omega", "Geo M", "Earth x (Helio)",
                                                                   "Earth y (Helio)", "Earth z (Helio)",
                                                                   "Earth vx (Helio)",
                                                                   "Earth vy (Helio)", "Earth vz (Helio)",
                                                                   "Moon x (Helio)", "Moon y (Helio)", "Moon z (Helio)",
                                                                   "Moon vx (Helio)",
                                                                   "Moon vy (Helio)", "Moon vz (Helio)"])

        return data

    @staticmethod
    def mm_file_parse_new(file_path):
        """
        Units are au, au/day, degrees
        This function parses a single minimoon file into a appropriately labelled dataframe, the input file
        should be a middleman file, which is the result of the integrations adding a year before and after capture,
        column names:
        "Object id", "Julian Date", "Distance", "Helio q", "Helio e", "Helio i", "Helio Omega", "Helio omega",
        "Helio M", "Helio x", "Helio y", "Helio z", "Helio vx", "Helio vy", "Helio vz", "Geo x", "Geo y", "Geo z",
        "Geo vx", "Geo vy", "Geo vz", "Geo q", "Geo e", "Geo i", "Geo Omega", "Geo omega", "Geo M", "Earth x (Helio)",
        "Earth y (Helio)", "Earth z (Helio)", "Earth vx (Helio)", "Earth vy (Helio)", "Earth vz (Helio)",
        "Moon x (Helio)", "Moon y (Helio)", "Moon z (Helio)", "Moon vx (Helio)", "Moon vy (Helio)",
        "Moon vz (Helio)" "Synodic x" "Synodic y" "Synodic z" "Eclip Long"
        :return: the data frame

        """

        data = pd.read_csv(file_path, sep=" ", header=0, names=["Object id", "Julian Date", "Distance", "Helio q",
                                                                "Helio e", "Helio i", "Helio Omega", "Helio omega",
                                                                "Helio M", "Helio x", "Helio y", "Helio z", "Helio vx",
                                                                "Helio vy", "Helio vz", "Geo x", "Geo y", "Geo z",
                                                                "Geo vx", "Geo vy", "Geo vz", "Geo q", "Geo e", "Geo i",
                                                                "Geo Omega", "Geo omega", "Geo M", "Earth x (Helio)",
                                                                "Earth y (Helio)", "Earth z (Helio)",
                                                                "Earth vx (Helio)",
                                                                "Earth vy (Helio)", "Earth vz (Helio)",
                                                                "Moon x (Helio)", "Moon y (Helio)", "Moon z (Helio)",
                                                                "Moon vx (Helio)",
                                                                "Moon vy (Helio)", "Moon vz (Helio)", "Synodic x",
                                                                "Synodic y", "Synodic z", "Eclip Long"])

        return data

    @staticmethod
    def mm_file_parse_new_new(file_path):
        """
        Units are au, au/day, degrees
        This function parses a single minimoon file into a appropriately labelled dataframe, the input file
        should be a middleman file, which is the result of the integrations adding a year before and after capture,
        column names:
        "Object id", "Julian Date", "Distance", "Helio q", "Helio e", "Helio i", "Helio Omega", "Helio omega",
        "Helio M", "Helio x", "Helio y", "Helio z", "Helio vx", "Helio vy", "Helio vz", "Geo x", "Geo y", "Geo z",
        "Geo vx", "Geo vy", "Geo vz", "Geo q", "Geo e", "Geo i", "Geo Omega", "Geo omega", "Geo M", "Earth x (Helio)",
        "Earth y (Helio)", "Earth z (Helio)", "Earth vx (Helio)", "Earth vy (Helio)", "Earth vz (Helio)",
        "Moon x (Helio)", "Moon y (Helio)", "Moon z (Helio)", "Moon vx (Helio)", "Moon vy (Helio)",
        "Moon vz (Helio)" "Synodic x" "Synodic y" "Synodic z" "Eclip Long" "sun-ast-dist", "sunearthl1-ast-dist",
        "phase_angle", "apparent_magnitude"
        :return: the data frame

        """

        data = pd.read_csv(file_path, sep=" ", header=0, names=["Object id", "Julian Date", "Distance", "Helio q",
                                                                "Helio e", "Helio i", "Helio Omega", "Helio omega",
                                                                "Helio M", "Helio x", "Helio y", "Helio z", "Helio vx",
                                                                "Helio vy", "Helio vz", "Geo x", "Geo y", "Geo z",
                                                                "Geo vx", "Geo vy", "Geo vz", "Geo q", "Geo e", "Geo i",
                                                                "Geo Omega", "Geo omega", "Geo M", "Earth x (Helio)",
                                                                "Earth y (Helio)", "Earth z (Helio)",
                                                                "Earth vx (Helio)",
                                                                "Earth vy (Helio)", "Earth vz (Helio)",
                                                                "Moon x (Helio)", "Moon y (Helio)", "Moon z (Helio)",
                                                                "Moon vx (Helio)",
                                                                "Moon vy (Helio)", "Moon vz (Helio)", "Synodic x",
                                                                "Synodic y", "Synodic z", "Eclip Long", "sun-ast-dist",
                                                                "sunearthl1-ast-dist",
                                                                "phase_angle", "apparent_magnitude"])

        return data

    @staticmethod
    def mm_file_parse_new_no_moon(file_path):
        """
        Units are au, au/day, degrees
        This function parses a single minimoon file into a appropriately labelled dataframe, the input file
        should be a middleman file, which is the result of the integrations adding a year before and after capture,
        column names:
        "Object id", "Julian Date", "Distance", "Helio q", "Helio e", "Helio i", "Helio Omega", "Helio omega",
        "Helio M", "Helio x", "Helio y", "Helio z", "Helio vx", "Helio vy", "Helio vz", "Geo x", "Geo y", "Geo z",
        "Geo vx", "Geo vy", "Geo vz", "Geo q", "Geo e", "Geo i", "Geo Omega", "Geo omega", "Geo M", "Earth x (Helio)",
        "Earth y (Helio)", "Earth z (Helio)", "Earth vx (Helio)", "Earth vy (Helio)", "Earth vz (Helio)",
        "Moon x (Helio)", "Moon y (Helio)", "Moon z (Helio)", "Moon vx (Helio)", "Moon vy (Helio)",
        "Moon vz (Helio)" "Synodic x" "Synodic y" "Synodic z" "Eclip Long"
        :return: the data frame

        """

        data = pd.read_csv(file_path, sep=" ", header=0, names=["Object id", "Julian Date", "Distance", "Helio q",
                                                                "Helio e", "Helio i", "Helio Omega", "Helio omega",
                                                                "Helio M", "Helio x", "Helio y", "Helio z", "Helio vx",
                                                                "Helio vy", "Helio vz", "Geo x", "Geo y", "Geo z",
                                                                "Geo vx", "Geo vy", "Geo vz", "Geo q", "Geo e", "Geo i",
                                                                "Geo Omega", "Geo omega", "Geo M", "Earth x (Helio)",
                                                                "Earth y (Helio)", "Earth z (Helio)",
                                                                "Earth vx (Helio)",
                                                                "Earth vy (Helio)", "Earth vz (Helio)",
                                                                "Moon x (Helio)", "Moon y (Helio)", "Moon z (Helio)",
                                                                "Moon vx (Helio)",
                                                                "Moon vy (Helio)", "Moon vz (Helio)", "Synodic x",
                                                                "Synodic y", "Synodic z", "Eclip Long",
                                                                "Earth-Moon Synodic x", "Earth-Moon Synodic y",
                                                                "Earth-Moon Synodic z", "Earth-Moon Synodic vx",
                                                                "Earth-Moon Synodic vy", "Earth-Moon Synodic Omega x",
                                                                "Earth-Moon Synodic Omega y",
                                                                "Earth-Moon Synodic Omega z", "Earth-TCO Distance",
                                                                "Moon-TCO Distance", "Sun-TCO Distance",
                                                                "Earth-Moon Synodic Sun x", "Earth-Moon Synodic Sun y",
                                                                "Earth-Moon Synodic Sun z", "Earth-Moon Synodic Moon x",
                                                                "Earth-Moon Synodic Moon y",
                                                                "Earth-Moon Synodic Moon z", "Earth-Moon Distance",
                                                                "Moon around EMS Omega x", "Moon around EMS Omega y",
                                                                "Moon around EMS Omega z"])

        return data

    @staticmethod
    def parse_master_previous(file_path):
        """
        function for obtaning master mimimoon data file, with parameters
        'Object id', 'H', 'D', 'Capture Date', 'Helio x at Capture', 'Helio y at Capture', 'Helio z at Capture',
        'Helio vx at Capture', 'Helio vy at Capture', 'Helio vz at Capture', 'Helio q at Capture', 'Helio e at Capture',
        'Helio i at Capture', 'Helio Omega at Capture', 'Helio omega at Capture', 'Helio M at Capture',
        'Geo x at Capture', 'Geo y at Capture', 'Geo z at Capture', 'Geo vx at Capture', 'Geo vy at Capture',
        'Geo vz at Capture', 'Geo q at Capture', 'Geo e at Capture', 'Geo i at Capture', 'Geo Omega at Capture',
        'Geo omega at Capture', 'Geo M at Capture', 'Moon (Helio) x at Capture', 'Moon (Helio) y at Capture',
        'Moon (Helio) z at Capture', 'Moon (Helio) vx at Capture', 'Moon (Helio) vy at Capture',
        'Moon (Helio) vz at Capture', 'Capture Duration', 'Spec. En. Duration', '3 Hill Duration', 'Number of Rev',
        '1 Hill Duration', 'Min. Distance', 'Release Date', 'Helio x at Release', 'Helio y at Release',
        'Helio z at Release', 'Helio vx at Release', 'Helio vy at Release', 'Helio vz at Release', 'Helio q at Release',
        'Helio e at Release', 'Helio i at Release', 'Helio Omega at Release', 'Helio omega at Release',
        'Helio M at Release', 'Geo x at Release', 'Geo y at Release', 'Geo z at Release', 'Geo vx at Release',
        'Geo vy at Release', 'Geo vz at Release', 'Geo q at Release', 'Geo e at Release', 'Geo i at Release',
        'Geo Omega at Release', 'Geo omega at Release', 'Geo M at Release', 'Moon (Helio) x at Release',
         'Moon (Helio) y at Release', 'Moon (Helio) z at Release', 'Moon (Helio) vx at Release',
         'Moon (Helio) vy at Release', 'Moon (Helio) vz at Release', 'Retrograde', 'Became Minimoon', 'Max. Distance',
         'Capture Index', 'Release Index', 'X at Earth Hill', 'Y at Earth Hill', 'Z at Earth Hill', 'Taxonomy', 'STC'
         "EMS Duration", "Periapsides in EMS", "Periapsides in 3 Hill", "Periapsides in 2 Hill", "Periapsides in 1 Hill",
        "STC Start", "STC Start Index", "STC End", "STC End Index", "Helio x at EMS", "Helio y at EMS", "Helio z at EMS",
         "Helio vx at EMS", "Helio vy at EMS", "Helio vz at EMS", "Earth x at EMS (Helio)", "Earth y at EMS (Helio)",
        "Earth z at EMS (Helio)", "Earth vx at EMS (Helio)", "Earth vy at EMS (Helio)", "Earth vz at EMS (Helio)",
         "Moon x at EMS (Helio)", "Moon y at EMS (Helio)", "Moon z at EMS (Helio)", "Moon vx at EMS (Helio)",
          "Moon vy at EMS (Helio)", "Moon vz at EMS (Helio)", 'Entry Date to EMS', 'Entry to EMS Index',
                                                             'Exit Date to EMS', 'Exit Index to EMS'
        :return:
        """
        master_data = pd.read_csv(file_path, sep=" ", header=0, names=['Object id', 'H', 'D', 'Capture Date',
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
                                                                       'Became Minimoon', 'Max. Distance',
                                                                       'Capture Index',
                                                                       'Release Index', 'X at Earth Hill',
                                                                       'Y at Earth Hill',
                                                                       'Z at Earth Hill', 'Taxonomy', 'STC',
                                                                       "EMS Duration",
                                                                       "Periapsides in EMS", "Periapsides in 3 Hill",
                                                                       "Periapsides in 2 Hill", "Periapsides in 1 Hill",
                                                                       "STC Start", "STC Start Index", "STC End",
                                                                       "STC End Index",
                                                                       "Helio x at EMS", "Helio y at EMS",
                                                                       "Helio z at EMS",
                                                                       "Helio vx at EMS", "Helio vy at EMS",
                                                                       "Helio vz at EMS",
                                                                       "Earth x at EMS (Helio)",
                                                                       "Earth y at EMS (Helio)",
                                                                       "Earth z at EMS (Helio)",
                                                                       "Earth vx at EMS (Helio)",
                                                                       "Earth vy at EMS (Helio)",
                                                                       "Earth vz at EMS (Helio)",
                                                                       "Moon x at EMS (Helio)", "Moon y at EMS (Helio)",
                                                                       "Moon z at EMS (Helio)",
                                                                       "Moon vx at EMS (Helio)",
                                                                       "Moon vy at EMS (Helio)",
                                                                       "Moon vz at EMS (Helio)",
                                                                       "EMS Start", "EMS Start Index", "EMS End",
                                                                       "EMS End Index"])

        return master_data

    @staticmethod
    def parse_master(file_path):
        """
        function for obtaning master mimimoon data file, with parameters
        'Object id', 'H', 'D', 'Capture Date', 'Helio x at Capture', 'Helio y at Capture', 'Helio z at Capture',
        'Helio vx at Capture', 'Helio vy at Capture', 'Helio vz at Capture', 'Helio q at Capture', 'Helio e at Capture',
        'Helio i at Capture', 'Helio Omega at Capture', 'Helio omega at Capture', 'Helio M at Capture',
        'Geo x at Capture', 'Geo y at Capture', 'Geo z at Capture', 'Geo vx at Capture', 'Geo vy at Capture',
        'Geo vz at Capture', 'Geo q at Capture', 'Geo e at Capture', 'Geo i at Capture', 'Geo Omega at Capture',
        'Geo omega at Capture', 'Geo M at Capture', 'Moon (Helio) x at Capture', 'Moon (Helio) y at Capture',
        'Moon (Helio) z at Capture', 'Moon (Helio) vx at Capture', 'Moon (Helio) vy at Capture',
        'Moon (Helio) vz at Capture', 'Capture Duration', 'Spec. En. Duration', '3 Hill Duration', 'Number of Rev',
        '1 Hill Duration', 'Min. Distance', 'Release Date', 'Helio x at Release', 'Helio y at Release',
        'Helio z at Release', 'Helio vx at Release', 'Helio vy at Release', 'Helio vz at Release', 'Helio q at Release',
        'Helio e at Release', 'Helio i at Release', 'Helio Omega at Release', 'Helio omega at Release',
        'Helio M at Release', 'Geo x at Release', 'Geo y at Release', 'Geo z at Release', 'Geo vx at Release',
        'Geo vy at Release', 'Geo vz at Release', 'Geo q at Release', 'Geo e at Release', 'Geo i at Release',
        'Geo Omega at Release', 'Geo omega at Release', 'Geo M at Release', 'Moon (Helio) x at Release',
         'Moon (Helio) y at Release', 'Moon (Helio) z at Release', 'Moon (Helio) vx at Release',
         'Moon (Helio) vy at Release', 'Moon (Helio) vz at Release', 'Retrograde', 'Became Minimoon', 'Max. Distance',
         'Capture Index', 'Release Index', 'X at Earth Hill', 'Y at Earth Hill', 'Z at Earth Hill', 'Taxonomy', 'STC'
         "EMS Duration", "Periapsides in EMS", "Periapsides in 3 Hill", "Periapsides in 2 Hill", "Periapsides in 1 Hill",
        "STC Start", "STC Start Index", "STC End", "STC End Index", "Helio x at EMS", "Helio y at EMS", "Helio z at EMS",
         "Helio vx at EMS", "Helio vy at EMS", "Helio vz at EMS", "Earth x at EMS (Helio)", "Earth y at EMS (Helio)",
        "Earth z at EMS (Helio)", "Earth vx at EMS (Helio)", "Earth vy at EMS (Helio)", "Earth vz at EMS (Helio)",
         "Moon x at EMS (Helio)", "Moon y at EMS (Helio)", "Moon z at EMS (Helio)", "Moon vx at EMS (Helio)",
          "Moon vy at EMS (Helio)", "Moon vz at EMS (Helio)", 'Entry Date to EMS', 'Entry to EMS Index',
         'Exit Date to EMS', 'Exit Index to EMS' "Dimensional Jacobi" "Non-Dimensional Jacobi" Alpha_I Beta_I Theta_M
        :return:
        """
        master_data = pd.read_csv(file_path, sep=" ", header=0, names=['Object id', 'H', 'D', 'Capture Date',
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
                                                                       'Became Minimoon', 'Max. Distance',
                                                                       'Capture Index',
                                                                       'Release Index', 'X at Earth Hill',
                                                                       'Y at Earth Hill',
                                                                       'Z at Earth Hill', 'Taxonomy', 'STC',
                                                                       "EMS Duration",
                                                                       "Periapsides in EMS", "Periapsides in 3 Hill",
                                                                       "Periapsides in 2 Hill", "Periapsides in 1 Hill",
                                                                       "STC Start", "STC Start Index", "STC End",
                                                                       "STC End Index",
                                                                       "Helio x at EMS", "Helio y at EMS",
                                                                       "Helio z at EMS",
                                                                       "Helio vx at EMS", "Helio vy at EMS",
                                                                       "Helio vz at EMS",
                                                                       "Earth x at EMS (Helio)",
                                                                       "Earth y at EMS (Helio)",
                                                                       "Earth z at EMS (Helio)",
                                                                       "Earth vx at EMS (Helio)",
                                                                       "Earth vy at EMS (Helio)",
                                                                       "Earth vz at EMS (Helio)",
                                                                       "Moon x at EMS (Helio)", "Moon y at EMS (Helio)",
                                                                       "Moon z at EMS (Helio)",
                                                                       "Moon vx at EMS (Helio)",
                                                                       "Moon vy at EMS (Helio)",
                                                                       "Moon vz at EMS (Helio)",
                                                                       'Entry Date to EMS', 'Entry to EMS Index',
                                                                       'Exit Date to EMS', 'Exit Index to EMS',
                                                                       "Dimensional Jacobi", "Non-Dimensional Jacobi",
                                                                       'Alpha_I',
                                                                       'Beta_I', 'Theta_M'])

        return master_data

    @staticmethod
    def parse_master_new_new(file_path):
        """
        function for obtaning master mimimoon data file, with parameters
        'Object id', 'H', 'D', 'Capture Date', 'Helio x at Capture', 'Helio y at Capture', 'Helio z at Capture',
        'Helio vx at Capture', 'Helio vy at Capture', 'Helio vz at Capture', 'Helio q at Capture', 'Helio e at Capture',
        'Helio i at Capture', 'Helio Omega at Capture', 'Helio omega at Capture', 'Helio M at Capture',
        'Geo x at Capture', 'Geo y at Capture', 'Geo z at Capture', 'Geo vx at Capture', 'Geo vy at Capture',
        'Geo vz at Capture', 'Geo q at Capture', 'Geo e at Capture', 'Geo i at Capture', 'Geo Omega at Capture',
        'Geo omega at Capture', 'Geo M at Capture', 'Moon (Helio) x at Capture', 'Moon (Helio) y at Capture',
        'Moon (Helio) z at Capture', 'Moon (Helio) vx at Capture', 'Moon (Helio) vy at Capture',
        'Moon (Helio) vz at Capture', 'Capture Duration', 'Spec. En. Duration', '3 Hill Duration', 'Number of Rev',
        '1 Hill Duration', 'Min. Distance', 'Release Date', 'Helio x at Release', 'Helio y at Release',
        'Helio z at Release', 'Helio vx at Release', 'Helio vy at Release', 'Helio vz at Release', 'Helio q at Release',
        'Helio e at Release', 'Helio i at Release', 'Helio Omega at Release', 'Helio omega at Release',
        'Helio M at Release', 'Geo x at Release', 'Geo y at Release', 'Geo z at Release', 'Geo vx at Release',
        'Geo vy at Release', 'Geo vz at Release', 'Geo q at Release', 'Geo e at Release', 'Geo i at Release',
        'Geo Omega at Release', 'Geo omega at Release', 'Geo M at Release', 'Moon (Helio) x at Release',
         'Moon (Helio) y at Release', 'Moon (Helio) z at Release', 'Moon (Helio) vx at Release',
         'Moon (Helio) vy at Release', 'Moon (Helio) vz at Release', 'Retrograde', 'Became Minimoon', 'Max. Distance',
         'Capture Index', 'Release Index', 'X at Earth Hill', 'Y at Earth Hill', 'Z at Earth Hill', 'Taxonomy', 'STC'
         "EMS Duration", "Periapsides in EMS", "Periapsides in 3 Hill", "Periapsides in 2 Hill", "Periapsides in 1 Hill",
        "STC Start", "STC Start Index", "STC End", "STC End Index", "Helio x at EMS", "Helio y at EMS", "Helio z at EMS",
         "Helio vx at EMS", "Helio vy at EMS", "Helio vz at EMS", "Earth x at EMS (Helio)", "Earth y at EMS (Helio)",
        "Earth z at EMS (Helio)", "Earth vx at EMS (Helio)", "Earth vy at EMS (Helio)", "Earth vz at EMS (Helio)",
         "Moon x at EMS (Helio)", "Moon y at EMS (Helio)", "Moon z at EMS (Helio)", "Moon vx at EMS (Helio)",
          "Moon vy at EMS (Helio)", "Moon vz at EMS (Helio)", 'Entry Date to EMS', 'Entry to EMS Index',
         'Exit Date to EMS', 'Exit Index to EMS' "Dimensional Jacobi" "Non-Dimensional Jacobi" Alpha_I Beta_I Theta_M
         "Minimum Energy", "Peri-EM-L2", "Average Geo z", "Average Geo vz", "Winding Difference"
        :return:
        """
        master_data = pd.read_csv(file_path, sep=" ", header=0, names=['Object id', 'H', 'D', 'Capture Date',
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
                                                                       'Became Minimoon', 'Max. Distance',
                                                                       'Capture Index',
                                                                       'Release Index', 'X at Earth Hill',
                                                                       'Y at Earth Hill',
                                                                       'Z at Earth Hill', 'Taxonomy', 'STC',
                                                                       "EMS Duration",
                                                                       "Periapsides in EMS", "Periapsides in 3 Hill",
                                                                       "Periapsides in 2 Hill", "Periapsides in 1 Hill",
                                                                       "STC Start", "STC Start Index", "STC End",
                                                                       "STC End Index",
                                                                       "Helio x at EMS", "Helio y at EMS",
                                                                       "Helio z at EMS",
                                                                       "Helio vx at EMS", "Helio vy at EMS",
                                                                       "Helio vz at EMS",
                                                                       "Earth x at EMS (Helio)",
                                                                       "Earth y at EMS (Helio)",
                                                                       "Earth z at EMS (Helio)",
                                                                       "Earth vx at EMS (Helio)",
                                                                       "Earth vy at EMS (Helio)",
                                                                       "Earth vz at EMS (Helio)",
                                                                       "Moon x at EMS (Helio)", "Moon y at EMS (Helio)",
                                                                       "Moon z at EMS (Helio)",
                                                                       "Moon vx at EMS (Helio)",
                                                                       "Moon vy at EMS (Helio)",
                                                                       "Moon vz at EMS (Helio)",
                                                                       'Entry Date to EMS', 'Entry to EMS Index',
                                                                       'Exit Date to EMS', 'Exit Index to EMS',
                                                                       "Dimensional Jacobi", "Non-Dimensional Jacobi",
                                                                       'Alpha_I',
                                                                       'Beta_I', 'Theta_M', "Minimum Energy",
                                                                       "Peri-EM-L2", "Average Geo z", "Average Geo vz",
                                                                       "Winding Difference"])

        return master_data

    @staticmethod
    def parse_master_new_new_new(file_path):
        """
        function for obtaning master mimimoon data file, with parameters
        'Object id', 'H', 'D', 'Capture Date', 'Helio x at Capture', 'Helio y at Capture', 'Helio z at Capture',
        'Helio vx at Capture', 'Helio vy at Capture', 'Helio vz at Capture', 'Helio q at Capture', 'Helio e at Capture',
        'Helio i at Capture', 'Helio Omega at Capture', 'Helio omega at Capture', 'Helio M at Capture',
        'Geo x at Capture', 'Geo y at Capture', 'Geo z at Capture', 'Geo vx at Capture', 'Geo vy at Capture',
        'Geo vz at Capture', 'Geo q at Capture', 'Geo e at Capture', 'Geo i at Capture', 'Geo Omega at Capture',
        'Geo omega at Capture', 'Geo M at Capture', 'Moon (Helio) x at Capture', 'Moon (Helio) y at Capture',
        'Moon (Helio) z at Capture', 'Moon (Helio) vx at Capture', 'Moon (Helio) vy at Capture',
        'Moon (Helio) vz at Capture', 'Capture Duration', 'Spec. En. Duration', '3 Hill Duration', 'Number of Rev',
        '1 Hill Duration', 'Min. Distance', 'Release Date', 'Helio x at Release', 'Helio y at Release',
        'Helio z at Release', 'Helio vx at Release', 'Helio vy at Release', 'Helio vz at Release', 'Helio q at Release',
        'Helio e at Release', 'Helio i at Release', 'Helio Omega at Release', 'Helio omega at Release',
        'Helio M at Release', 'Geo x at Release', 'Geo y at Release', 'Geo z at Release', 'Geo vx at Release',
        'Geo vy at Release', 'Geo vz at Release', 'Geo q at Release', 'Geo e at Release', 'Geo i at Release',
        'Geo Omega at Release', 'Geo omega at Release', 'Geo M at Release', 'Moon (Helio) x at Release',
         'Moon (Helio) y at Release', 'Moon (Helio) z at Release', 'Moon (Helio) vx at Release',
         'Moon (Helio) vy at Release', 'Moon (Helio) vz at Release', 'Retrograde', 'Became Minimoon', 'Max. Distance',
         'Capture Index', 'Release Index', 'X at Earth Hill', 'Y at Earth Hill', 'Z at Earth Hill', 'Taxonomy', 'STC'
         "EMS Duration", "Periapsides in EMS", "Periapsides in 3 Hill", "Periapsides in 2 Hill", "Periapsides in 1 Hill",
        "STC Start", "STC Start Index", "STC End", "STC End Index", "Helio x at EMS", "Helio y at EMS", "Helio z at EMS",
         "Helio vx at EMS", "Helio vy at EMS", "Helio vz at EMS", "Earth x at EMS (Helio)", "Earth y at EMS (Helio)",
        "Earth z at EMS (Helio)", "Earth vx at EMS (Helio)", "Earth vy at EMS (Helio)", "Earth vz at EMS (Helio)",
         "Moon x at EMS (Helio)", "Moon y at EMS (Helio)", "Moon z at EMS (Helio)", "Moon vx at EMS (Helio)",
          "Moon vy at EMS (Helio)", "Moon vz at EMS (Helio)", 'Entry Date to EMS', 'Entry to EMS Index',
         'Exit Date to EMS', 'Exit Index to EMS' "Dimensional Jacobi" "Non-Dimensional Jacobi" Alpha_I Beta_I Theta_M
         "Minimum Energy", "Peri-EM-L2", "Average Geo z", "Average Geo vz", "Winding Difference"
        :return:
        """
        master_data = pd.read_csv(file_path, sep=" ", header=0, names=['Object id', 'H', 'D', 'Capture Date',
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
                                                                       'Became Minimoon', 'Max. Distance',
                                                                       'Capture Index',
                                                                       'Release Index', 'X at Earth Hill',
                                                                       'Y at Earth Hill',
                                                                       'Z at Earth Hill', 'Taxonomy', 'STC',
                                                                       "EMS Duration",
                                                                       "Periapsides in EMS", "Periapsides in 3 Hill",
                                                                       "Periapsides in 2 Hill", "Periapsides in 1 Hill",
                                                                       "STC Start", "STC Start Index", "STC End",
                                                                       "STC End Index",
                                                                       "Helio x at EMS", "Helio y at EMS",
                                                                       "Helio z at EMS",
                                                                       "Helio vx at EMS", "Helio vy at EMS",
                                                                       "Helio vz at EMS",
                                                                       "Earth x at EMS (Helio)",
                                                                       "Earth y at EMS (Helio)",
                                                                       "Earth z at EMS (Helio)",
                                                                       "Earth vx at EMS (Helio)",
                                                                       "Earth vy at EMS (Helio)",
                                                                       "Earth vz at EMS (Helio)",
                                                                       "Moon x at EMS (Helio)", "Moon y at EMS (Helio)",
                                                                       "Moon z at EMS (Helio)",
                                                                       "Moon vx at EMS (Helio)",
                                                                       "Moon vy at EMS (Helio)",
                                                                       "Moon vz at EMS (Helio)",
                                                                       'Entry Date to EMS', 'Entry to EMS Index',
                                                                       'Exit Date to EMS', 'Exit Index to EMS',
                                                                       "Dimensional Jacobi", "Non-Dimensional Jacobi",
                                                                       'Alpha_I',
                                                                       'Beta_I', 'Theta_M', "Minimum Energy",
                                                                       "Peri-EM-L2", "Average Geo z", "Average Geo vz",
                                                                       "Winding Difference", "Min_SunEarthL1_V",
                                                                       "Min_SunEarthL1_V_index"])

        return master_data
