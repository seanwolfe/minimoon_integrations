import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QLabel
from PyQt5.QtCore import QSize, Qt
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import os
from MmAnalyzer import MmAnalyzer
from astropy import constants as const
from MM_Parser import MmParser
from space_fncs import eci_ecliptic_to_sunearth_synodic
import numpy
from astropy.units import cds
import astropy.units as u
from astropy.time import Time
cds.enable()

import pandas as pd


matplotlib.use('Qt5Agg')

##############################
# TO DO
#############################

# What are the time units of integrations (days or hours?): apparently 1/8 th of a day
# Add visualizations for 2006 RH120 and 2020 CD3 2022 NX1

class MplCanvas(FigureCanvasQTAgg):

    """
    This class is for creating objects to interact with pyqt5 and matplotlib, in 2D or 3D, specified by the
    d pararmeter: integer that is two or three.
    width: width of plot ?units? default: 5
    height: height of pot ?units? default: 4
    """

    def __init__(self, d, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        if d == 2:
            self.axes = fig.add_subplot(111)
        elif d == 3:
            self.axes = fig.add_subplot(111, projection='3d')
        else:
            print("ERROR: d should be either 2 or 3")
        super(MplCanvas, self).__init__(fig)


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        # Constants
        mu_e = const.GM_earth.value  # Nominal Earth mass parameter (m3/s2)

        # Get the data for the first minimoon file
        data, new_data = self.create_parser()

        # pd.set_option('display.max_rows', None)
        # pd.set_option('display.float_format', lambda x: '%.5f' %x)
        # print(data["Geo i"])

        # Application title
        self.setWindowTitle("MiniViz")

        # specify the overall layout
        layout1 = QHBoxLayout()  # contains all the widgets a single box
        layout2 = QVBoxLayout()  # contains left two widgets
        layout3 = QVBoxLayout()  # contains middle two widgets

        layout1.setContentsMargins(0,0,0,0)
        layout1.setSpacing(5)

        # create analyzer to identify key minimoon statistics
        self.mm_analyzer = MmAnalyzer()
        trans_xyz = self.mm_analyzer.minimoon_check(data, mu_e)
        trans_xyz_new = self.mm_analyzer.minimoon_check(new_data, mu_e)

        # create Earth and moon trajectories
        x_E, y_E, z_E, x_moon, y_moon, z_moon = self.earth_and_moon_traj(new_data)

        # Create the maptlotlib FigureCanvas object,
        # which defines a single set of axes as self.axes.

        # create visualization for top left window
        # Top Left
        self.sc_tl = MplCanvas(2, width=5, height=4, dpi=100)
        self.create_top_left(trans_xyz, trans_xyz_new, x_E, y_E, x_moon, y_moon)

        # Bottom Left
        self.sc_bl = MplCanvas(2, width=5, height=4, dpi=100)
        self.create_bottom_left(trans_xyz, trans_xyz_new, x_E, z_E, x_moon, z_moon)

        # Top Right
        self.sc_tr = MplCanvas(3, width=5, height=4.61, dpi=100)  # Top right
        self.create_top_right(data, new_data, x_E, y_E, z_E, x_moon, y_moon, z_moon)

        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        self.toolbar_tl = NavigationToolbar(self.sc_tl, self)
        self.toolbar_bl = NavigationToolbar(self.sc_bl, self)
        self.toolbar_tr = NavigationToolbar(self.sc_tr, self)

        # Bottom Right : minimoon statistics
        self.w_title = QLabel(str(self.mm_parser.mm_data["File Name"].iloc[0]))  # make title of bottom right the minimoon name
        font = self.w_title.font()
        font.setPointSize(30)
        self.w_title.setFont(font)
        self.w_title.setAlignment(Qt.AlignTop | Qt.AlignHCenter)  # align the title in the window
        mm_flag_text = self.create_bottom_right()
        self.w_mm_flag = QLabel(mm_flag_text)
        font = self.w_mm_flag.font()
        font.setPointSize(16)
        self.w_mm_flag.setFont(font)
        self.w_mm_flag.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        # Assign all widgets to the layout
        layout2.addWidget(self.toolbar_tl)  # matplotlib toolbar for the top left viz
        layout2.addWidget(self.sc_tl)  # graph for top left viz
        layout2.addWidget(self.toolbar_bl)
        layout2.addWidget(self.sc_bl)
        layout1.addLayout(layout2, 44)  # add left two windows (layout 2) to overall layout (layout 1)
        layout3.addWidget(self.toolbar_tr)
        layout3.addWidget(self.sc_tr)
        layout3.addWidget(self.w_title, 10)
        layout3.addWidget(self.w_mm_flag, 90)
        layout1.addLayout(layout3, 44)  # add right viz (layout 3) to overall layout (layout 1)

        # Make the rightmost part of miniviz, the scrolling list to switch between minimoons
        list_widget = QListWidget()
        list_widget.addItems(self.mm_parser.mm_data["File Name"])  # all the minimoon names
        list_widget.currentItemChanged.connect(self.index_changed)  # this is the callback for when the user clicks a
        # different minimoon in the scroll list
        layout1.addWidget(list_widget, 12)  # add the scrolling list to the overall layout

        widget = QWidget()
        widget.setLayout(layout1)
        self.setCentralWidget(widget)
        self.show()

    def index_changed(self, i):  # Not an index, i is a QListWidgetItem
        """
        What happens when the user changes the minimoon they wish to visualize
        :param i: this is the minimoon index which has been clicked, it is a QListWidgetItem
        :return:
        """

        # Constants
        mu_e = const.GM_earth.value  # Nominal Earth mass parameter (m3/s2)

        # Generate new data from clicked minimoon - fedorets
        destination_path = os.path.join(os.path.expanduser('~'), 'Documents', 'sean', 'minimoon_integrations',
                                        'minimoon_files')
        mm_file_name = i.text() + ".dat"
        temp_file = destination_path + '/' + mm_file_name
        header = None
        data = self.mm_parser.mm_file_parse(temp_file, header)
        trans_xyz = self.mm_analyzer.minimoon_check(data, mu_e)


        # Generate new data from clicked minimoon - fedorets
        destination_path_new = os.path.join(os.path.expanduser('~'), 'Documents', 'sean', 'minimoon_integrations',
                                        'minimoon_files_oorb')
        mm_file_name_new = i.text() + ".csv"
        temp_file_new = destination_path_new + '/' + mm_file_name_new
        header = 0
        new_data = self.mm_parser.mm_file_parse(temp_file_new, header)
        trans_xyz_new = self.mm_analyzer.minimoon_check(new_data, mu_e)

        # create Earth and moon trajectories
        x_E, y_E, z_E, x_moon, y_moon, z_moon = self.earth_and_moon_traj(data)

        # Create the maptlotlib FigureCanvas object,
        # which defines a single set of axes as self.axes.

        # Top Left
        self.sc_tl.axes.cla()
        self.create_top_left(trans_xyz, trans_xyz_new, x_E, y_E, x_moon, y_moon)
        self.sc_tl.draw()

        # Bottom Left
        self.sc_bl.axes.cla()
        self.create_bottom_left(trans_xyz, trans_xyz_new, x_E, z_E, x_moon, z_moon)
        self.sc_bl.draw()

        # Top Right
        self.sc_tr.axes.cla()
        self.create_top_right(data, new_data, x_E, y_E, z_E, x_moon, y_moon, z_moon)
        self.sc_tr.draw()

        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        self.toolbar_tl = NavigationToolbar(self.sc_tl, self)
        self.toolbar_bl = NavigationToolbar(self.sc_bl, self)
        self.toolbar_tr = NavigationToolbar(self.sc_tr, self)

        # Bottom Right : minimoon statistics
        self.w_title.setText(str(i.text()))  # update title
        mm_flag_text = self.create_bottom_right()
        self.w_mm_flag.setText(mm_flag_text)

        return

    def create_parser(self):
        """
        The function just does some initialzations necessary to analyze and later visualize the data later on
        :return:
        """

        mm_master_file_name = 'NESCv9reintv1.TCO.withH.kep.des'  # name of the minimoon file to parsed

        # name of the directory where the mm file is located,
        # also top level directory where all the integration data is located
        mm_file_dir = os.path.join(os.path.expanduser('~'), 'OneDrive', 'Thesis', 'Minimoon_Integrations',
                                   'minimoon_data')

        mm_file_path = mm_file_dir + '/' + mm_master_file_name  # path for the minimoon file
        destination_path = os.path.join(os.path.expanduser('~'), 'Documents', 'sean', 'minimoon_integrations',
                                        'minimoon_files')

        destination_path_new = os.path.join(os.path.expanduser('~'), 'Documents', 'sean', 'minimoon_integrations',
                                        'minimoon_files_oorb')

        # create parser
        self.mm_parser = MmParser(mm_file_path, mm_file_dir, destination_path)

        # organize the data in the minimoon_file
        self.mm_parser.organize_data()

        # fetch all the files from all the folders within the top level directory
        # mm_parser.fetch_files()

        mm_file_name = self.mm_parser.mm_data["File Name"].iloc[2] + ".dat"
        mm_file_name_new = self.mm_parser.mm_data["File Name"].iloc[2] + ".csv"
        temp_file = destination_path + '/' + mm_file_name
        temp_file_new = destination_path_new + '/' + mm_file_name_new
        header = None
        data = self.mm_parser.mm_file_parse(temp_file, header)
        header = 0
        new_data = self.mm_parser.mm_file_parse(temp_file_new, header)

        return data, new_data

    @staticmethod
    def earth_and_moon_traj(data):
        """
        A function to package code for getting earth and moon data for viz with minimoon
        :param data: data is the integration results
        :return:
        """

        R_E = (6378 * u.km).to(u.AU) / u.AU
        u_E, v_E = numpy.mgrid[0:2 * numpy.pi:50j, 0:numpy.pi:50j]
        x_E = R_E * numpy.cos(u_E) * numpy.sin(v_E)
        y_E = R_E * numpy.sin(u_E) * numpy.sin(v_E)
        z_E = R_E * numpy.cos(v_E)

        # draw moon
        x_moon = data["Moon x (Helio)"] - data["Earth x (Helio)"]
        y_moon = data["Moon y (Helio)"] - data["Earth y (Helio)"]
        z_moon = data["Moon z (Helio)"] - data["Earth z (Helio)"]

        return x_E, y_E, z_E, x_moon, y_moon, z_moon

    def create_top_left(self, trans_xyz, trans_xyz_new, x_E, y_E, x_moon, y_moon):

        """
        Creating the top left window of the application, a xy vizulasation of the minimoon synodic trajectory

        :param trans_xyz: the synodic xyz of the minimoon in question
        :param x_E: for drawing a globe
        :param y_E: for drawing a globe
        :param x_moon: the geocentric x of the moon
        :param y_moon: the geocentric y of the moon
        :return:
        """

        self.sc_tl.axes.scatter(trans_xyz[0, 0], trans_xyz[1, 0], color='orange', linewidth=5, label='Start')
        self.sc_tl.axes.scatter(trans_xyz[0, -1], trans_xyz[1, -1], color='black', linewidth=5, label='End')
        self.sc_tl.axes.plot(trans_xyz[0, :], trans_xyz[1, :], 'green', linewidth=5, label='Fedorets')
        self.sc_tl.axes.plot(trans_xyz_new[0, :], trans_xyz_new[1, :], 'gray', linewidth=2, label='OOrb')
        self.sc_tl.axes.plot(x_E, y_E, 'blue', alpha=1)
        self.sc_tl.axes.plot(x_moon, y_moon, 'red', alpha=1)
        self.sc_tl.axes.set_xlabel('X (AU)')
        self.sc_tl.axes.set_ylabel('Y (AU)')
        self.sc_tl.axes.legend()
        self.sc_tl.axes.set_title('Minimoon Trajectory in Synodic Frame')

        return

    def create_bottom_left(self, trans_xyz, trans_xyz_new, x_E, z_E, x_moon, z_moon):

        """
        Creating the bottom left window of the application, a xz vizulasation of the minimoon synodic trajectory

        :param trans_xyz: the synodic xyz of the minimoon in question
        :param x_E: for drawing a globe
        :param z_E: for drawing a globe
        :param x_moon: the geocentric x of the moon
        :param z_moon: the geocentric y of the moon
        :return:
        """

        self.sc_bl.axes.scatter(trans_xyz[0, 0], trans_xyz[2, 0], color='orange', linewidth=5, label='Start')
        self.sc_bl.axes.scatter(trans_xyz[0, -1], trans_xyz[2, -1], color='black', linewidth=5, label='End')
        self.sc_bl.axes.plot(trans_xyz[0, :], trans_xyz[2, :], 'green', linewidth=5)
        self.sc_bl.axes.plot(trans_xyz_new[0, :], trans_xyz_new[2, :], 'gray', linewidth=2)
        self.sc_bl.axes.plot(x_E, z_E, 'blue', alpha=1)
        self.sc_bl.axes.plot(x_moon, z_moon, 'red', alpha=1)
        self.sc_bl.axes.set_xlabel('X (AU)')
        self.sc_bl.axes.set_ylabel('Z (AU)')
        self.sc_bl.axes.legend()
        self.sc_bl.axes.set_title('Minimoon Trajectory in Synodic Frame (XZ)')

        return

    def create_top_right(self, data, new_data, x_E, y_E, z_E, x_moon, y_moon, z_moon):
        """
        Create the visualization for the top right window, whic is a 3d vizualization

        :param data: minimoon data
        :param x_E: for drawing earth
        :param y_E: for drawing earth
        :param z_E: for drawing earth
        :param x_moon: geocentric x of moon
        :param y_moon: geocentric y of moon
        :param z_moon: geocentric z of moon
        :return:
        """

        self.sc_tr.axes.plot3D(data["Geo x"], data["Geo y"], data["Geo z"], 'green', linewidth=5, label='Minimoon ' +
                                                                                                        str(
                                                                                                            self.mm_parser.mm_data[
                                                                                                                "File Name"].iloc[
                                                                                                                0]))

        self.sc_tr.axes.scatter3D(data["Geo x"].iloc[0], data["Geo y"].iloc[0], data["Geo z"].iloc[0], 'orange',
                                  linewidth=5,
                                  label='Start')
        self.sc_tr.axes.scatter3D(data["Geo x"].iloc[-1], data["Geo y"].iloc[-1], data["Geo z"].iloc[-1], 'black',
                                  linewidth=5,
                                  label='End')
        self.sc_tr.axes.plot3D(new_data["Geo x"], new_data["Geo y"], new_data["Geo z"], 'gray', linewidth=2, label='OOrb Minimoon ' +
                                                                                                        str(
                                                                                                            self.mm_parser.mm_data[
                                                                                                                "File Name"].iloc[
                                                                                                                0]))

        self.sc_tr.axes.plot3D(x_moon, y_moon, z_moon, 'red', label='Moon')

        # alpha controls opacity
        self.sc_tr.axes.plot_surface(x_E, y_E, z_E, color="b", alpha=1)
        self.sc_tr.axes.legend(loc='upper left')
        self.sc_tr.axes.set_xlabel('x (AU)')
        self.sc_tr.axes.set_ylabel('y (AU)')
        self.sc_tr.axes.set_zlabel('z (AU)')
        self.sc_tr.axes.set_title('Minimoon Trajectory in ECI Frame')

        return

    def create_bottom_right(self):
        """
        Create the visualization for the bottom right widget, which are statistics generated of the minimoon in question
        :return:
        """

        dots = "..."
        mm_flag_text = "Object became minimoon: " + dots + \
                       self.mm_analyzer.minimoon_flag + "\n" + \
                       "Start of temporary capture: " + dots + \
                       str(self.mm_analyzer.capture_start.isot) + "\n" \
                                                        "End of  temporary capture: " + dots + \
                       str(self.mm_analyzer.capture_end.isot) + "\n" \
                                                      "Duration (days) of Temporary capture: " + dots + \
                       self.mm_analyzer.capture_duration + "\n" \
                                                           "Time spent (days) closer than 3 Earth Hill Radii: " + dots + \
                       self.mm_analyzer.three_eh_duration + "\n" \
                                                            "Time spent (days) with specific energy less than zero (wrt to observer): " + dots + \
                       self.mm_analyzer.epsilon_duration + "\n" \
                                                           "Completed at least a revolution: " + dots + \
                       self.mm_analyzer.rev_flag + "\n" \
                                                   "Number of revolutions completed: " + dots + \
                       self.mm_analyzer.revolutions + "\n" \
                                                      "Approached to within 1 Earth Hill radius: " + dots + \
                       self.mm_analyzer.one_eh_flag + "\n" \
                                                      "Time spent (days) close than 1 Earth Hill radius: " + dots + \
                       self.mm_analyzer.one_eh_duration + "\n" \
                                                          "Minimum distance reached to observer (AU): " + dots + \
                       self.mm_analyzer.min_dist

        return mm_flag_text

if __name__ == '__main__':

    # Fetch a single file
    # for id in mm_parser.mm_data["File Name"]:
    #
    #     Grab data from file
        # mm_file_name = id + ".dat"
        # test_file = destination_path + '/' + mm_file_name
        # data = mm_parser.mm_file_parse(test_file)
        # print(data["Geo x"].iloc[-1])
        # MmAnalyzer.minimoon_check(data, mu_e)

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec()
