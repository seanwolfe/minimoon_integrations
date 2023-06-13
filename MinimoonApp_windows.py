import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import os
from MmAnalyzer import MmAnalyzer
from astropy import constants as const
from MM_Parser import MmParser
from space_fncs import eci_ecliptic_to_sunearth_synodic
import numpy as np
from astropy.units import cds
import astropy.units as u
from MmPopulation import MmPopulation
cds.enable()
import typing


matplotlib.use('Qt5Agg')

##############################
# TO DO
#############################

# Add visualizations for 2006 RH120 and 2020 CD3 2022 NX1

class TableModel(QAbstractTableModel):
    """
        This class creates table model with certain functions
    """

    def __init__(self, table_data, parent=None):
        super().__init__(parent)
        self.table_data = table_data

    def rowCount(self, parent: QModelIndex = ...) -> int:
        return self.table_data.shape[0]

    def columnCount(self, parent: QModelIndex = ...) -> int:
        return self.table_data.shape[1]

    def data(self, index: QModelIndex, role: int = ...) -> typing.Any:
        if role == Qt.DisplayRole:
            return str(self.table_data.loc[index.row()][index.column()])

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = ...) -> typing.Any:
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return str(self.table_data.columns[section])

    def setColumn(self, col, array_items):
        """Set column data"""
        self.table_data[col] = array_items
        # Notify table, that data has been changed
        self.dataChanged.emit(QModelIndex(), QModelIndex())

    def getColumn(self, col):
        """Get column data"""
        return self.table_data[col]

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

class MyTableWidget(QWidget):
    """
    This class allows you to add tabs to the applications
    tab1 = individual minimoon data
    tab2 = population relevant data
    """

    def __init__(self, parent, tab1, tab2):
        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = tab1
        self.tab2 = tab2
        self.tabs.resize(300, 200)

        # Add tabs
        self.tabs.addTab(self.tab1, "Individual Statistics")
        self.tabs.addTab(self.tab2, "Population Statistics")


        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)


class MainWindow(QMainWindow):

    # Properties
    mm_analyzer = ""  # see MmAnalyzer class
    sc_tl = ""  # the holder for the top left figure, which will come from matplotlib
    sc_bl = ""  # bottom left
    sc_tr = ""  # top right
    toolbar_tr = ""  # top right toolbar of the matplotlib figure, they have to be added individually
    toolbar_tl = ""
    toolbar_bl = ""
    w_mm_flag = ""  # this is the flag whether the object became a minimoon or not, may be depracated, for tab1-br
    fig_21 = ""  # these figures and tool bars are all for the second tab containing population information
    fig_22 = ""
    fig_31 = ""
    fig_32 = ""
    fig_41 = ""
    fig_42 = ""
    fig_51 = ""
    fig_52 = ""
    toolbar_21 = ""
    toolbar_22 = ""
    toolbar_31 = ""
    toolbar_32 = ""
    toolbar_41 = ""
    toolbar_42 = ""
    toolbar_51 = ""
    toolbar_52 = ""
    scroll = ""  # allows the app to scroll through figures instead of compressing graphs to window size
    widget = ""  # holds the main widget

    def __init__(self):
        super(MainWindow, self).__init__()

        # pd.set_option('display.max_rows', None)
        # pd.set_option('display.float_format', lambda x: '%.5f' %x)

        # Application title
        self.setWindowTitle("MiniViz")

        # get data of the first minimoon in the list, as well as a master file containing some info about all minimoons
        data, master = self.create_parser_windows()

        # generate the first tab
        layout1 = self.gen_tab_1(data, master)
        widget = QWidget()
        widget.setLayout(layout1)
        widget.setFixedHeight(1000)  # fix size so that scrolling can work

        # Create a scroll area widget
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)  # Allow the widget to be resized

        # generating the population data tab of the application
        layout2 = self.gen_tab_2(master)

        # Create a widget to contain the main layout
        widget2 = QWidget()
        widget2.setLayout(layout2)
        widget2.setFixedHeight(3300)

        # Set the widget as the content of the scroll area
        self.table_widget = MyTableWidget(self, widget, widget2)
        self.scroll.setWidget(self.table_widget)
        self.setCentralWidget(self.scroll)
        self.show()

    def gen_tab_1(self, data, master):
        """
        generates the top left/right bottom left/right parts of MiniViz in the fist tab
        tl: synodic xy
        tr: 3d geocentric
        bl: synodic xz
        br: capture statistics
        :param data: individual minimoon dataset
        :param master: master file dataset
        :return:
        """

        # specify the overall layout
        layout1 = QHBoxLayout()  # contains all the widgets a single box
        layout2 = QVBoxLayout()  # contains left two widgets
        layout3 = QVBoxLayout()  # contains middle two widgets

        layout1.setContentsMargins(0, 0, 0, 0)
        layout1.setSpacing(5)

        # create analyzer to identify key minimoon statistics
        self.mm_analyzer = MmAnalyzer()

        # Get the data for the first minimoon file
        data_cap = data.iloc[int(master['Capture Index'].iloc[0]):int(master['Release Index'].iloc[0])]
        trans_xyz = np.array([data["Synodic x"], data["Synodic y"], data["Synodic z"]])
        trans_cap = np.array([data['Synodic x'].iloc[master['Capture Index'].iloc[0]:master['Release Index'].iloc[0]],
                              data['Synodic y'].iloc[master['Capture Index'].iloc[0]:master['Release Index'].iloc[0]],
                              data['Synodic z'].iloc[master['Capture Index'].iloc[0]:master['Release Index'].iloc[0]]])

        # create Earth and moon trajectories
        x_E, y_E, z_E, x_moon, y_moon, z_moon = self.earth_and_moon_traj(data)

        # Create the maptlotlib FigureCanvas object,
        # which defines a single set of axes as self.axes.
        # create visualization for top left window
        # Top Left
        self.sc_tl = MplCanvas(2, width=5, height=4, dpi=100)
        self.create_top_left(trans_xyz, trans_cap, x_E, y_E, x_moon, y_moon)

        # Bottom Left
        self.sc_bl = MplCanvas(2, width=5, height=4, dpi=100)
        self.create_bottom_left(trans_xyz, trans_cap, x_E, z_E, x_moon, z_moon)

        # Top Right
        self.sc_tr = MplCanvas(3, width=5, height=7, dpi=100)  # Top right
        self.create_top_right(data, data_cap, x_E, y_E, z_E, x_moon, y_moon, z_moon)

        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        self.toolbar_tl = NavigationToolbar(self.sc_tl, self)
        self.toolbar_bl = NavigationToolbar(self.sc_bl, self)
        self.toolbar_tr = NavigationToolbar(self.sc_tr, self)

        # Bottom Right : minimoon statistics
        self.w_title = QLabel(str(data["Object id"].iloc[0]))  # make title of bottom right the minimoon name
        font = self.w_title.font()
        font.setPointSize(30)
        self.w_title.setFont(font)
        self.w_title.setAlignment(Qt.AlignTop | Qt.AlignHCenter)  # align the title in the window
        mm_flag_text = self.create_bottom_right(master, str(data["Object id"].iloc[0]))
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
        list_widget.addItems(master["Object id"])  # all the minimoon names
        list_widget.currentItemChanged.connect(
            self.index_changed_windows)  # this is the callback for when the user clicks a
        # different minimoon in the scroll list
        layout1.addWidget(list_widget, 12)  # add the scrolling list to the overall layout

        return layout1

    def gen_tab_2(self, master):
        """
        Generate the tab in MiniViz containing visualizations about the population
        :param master: the datafile containing info about all minimoons
        :return:
        """

        # specify the overall layout
        # main layour stacks boxes vertically. in each of those boxes there are two boxes (figures) stacked horizontally
        # in each of those boxes, there are two boxes stacked vertically (matplotlib toolbar and figure)
        layout1 = QVBoxLayout()  # contains all the widgets a single box
        layout2 = QHBoxLayout()  # contains tax table
        layout3 = QHBoxLayout()  # contains 2nd row
        layout21 = QVBoxLayout()  # second row first column
        layout22 = QVBoxLayout()
        layout4 = QHBoxLayout()  # contains 3rd row
        layout31 = QVBoxLayout()
        layout32 = QVBoxLayout()
        layout5 = QHBoxLayout()  # contains 4th row
        layout41 = QVBoxLayout()
        layout42 = QVBoxLayout()
        layout6 = QHBoxLayout()  # contains 5th row
        layout51 = QVBoxLayout()
        layout52 = QVBoxLayout()

        layout1.setContentsMargins(0, 0, 0, 0)
        layout1.setSpacing(5)

        # Get retrograde and prograde TCOs
        tco_pop = master[(master['Became Minimoon'] == 1)]
        retrograde_pop = master[(master['Retrograde'] == 1) & (master['Became Minimoon'] == 1)]
        prograde_pop = master[(master['Retrograde'] == 0) & (master['Became Minimoon'] == 1)]

        # create dataframe for the taxonomy table
        table_data = MmPopulation.tax_table(tco_pop, retrograde_pop, prograde_pop)
        table = QTableView()
        table.setModel(TableModel(table_data=table_data))
        layout2.addWidget(table, 99)

        # Create the maptlotlib FigureCanvas object,
        # which defines a single set of axes as self.axes.

        # second row first fig
        self.fig_21 = MplCanvas(2, width=5, height=7, dpi=100)
        self.create_21(retrograde_pop, prograde_pop)
        self.toolbar_21 = NavigationToolbar(self.fig_21, self)

        # second row second fig (right)
        self.fig_22 = MplCanvas(2, width=5, height=7, dpi=100)  # Top right
        self.create_22(retrograde_pop, prograde_pop)
        self.toolbar_22 = NavigationToolbar(self.fig_22, self)

        # combine respective toolbars and figs
        layout21.addWidget(self.toolbar_21)
        layout21.addWidget(self.fig_21)
        layout22.addWidget(self.toolbar_22)
        layout22.addWidget(self.fig_22)

        # combine left and right fig
        layout3.addLayout(layout21)
        layout3.addLayout(layout22)

        self.fig_31 = MplCanvas(2, width=5, height=7, dpi=100)
        self.create_31(tco_pop)
        self.toolbar_31 = NavigationToolbar(self.fig_31, self)

        self.fig_32 = MplCanvas(2, width=5, height=7, dpi=100)  # Top right
        self.create_32(tco_pop)
        self.toolbar_32 = NavigationToolbar(self.fig_32, self)

        layout31.addWidget(self.toolbar_31)
        layout31.addWidget(self.fig_31)
        layout32.addWidget(self.toolbar_32)
        layout32.addWidget(self.fig_32)

        layout4.addLayout(layout31)
        layout4.addLayout(layout32)

        self.fig_41 = MplCanvas(2, width=5, height=7, dpi=100)
        self.create_41(tco_pop)
        self.toolbar_41 = NavigationToolbar(self.fig_41, self)

        self.fig_42 = MplCanvas(2, width=5, height=7, dpi=100)  # Top right
        self.create_42(tco_pop)
        self.toolbar_42 = NavigationToolbar(self.fig_42, self)

        layout41.addWidget(self.toolbar_41)
        layout41.addWidget(self.fig_41)
        layout42.addWidget(self.toolbar_42)
        layout42.addWidget(self.fig_42)

        layout5.addLayout(layout41)
        layout5.addLayout(layout42)

        self.fig_51 = MplCanvas(2, width=5, height=7, dpi=100)
        self.create_51(retrograde_pop, prograde_pop, tco_pop)
        self.toolbar_51 = NavigationToolbar(self.fig_51, self)

        self.fig_52 = MplCanvas(2, width=5, height=7, dpi=100)  # Top right
        self.create_52(tco_pop)
        self.toolbar_52 = NavigationToolbar(self.fig_52, self)

        layout51.addWidget(self.toolbar_51)
        layout51.addWidget(self.fig_51)
        layout52.addWidget(self.toolbar_52)
        layout52.addWidget(self.fig_52)

        layout6.addLayout(layout51)
        layout6.addLayout(layout52)

        # Assign all widgets to the layout
        layout1.addLayout(layout2)  # matplotlib toolbar for the top left viz
        layout1.addLayout(layout3)
        layout1.addLayout(layout4)
        layout1.addLayout(layout5)
        layout1.addLayout(layout6)

        return layout1

    def index_changed_windows(self, i):  # Not an index, i is a QListWidgetItem
        """
        What happens when the user changes the minimoon they wish to visualize
        :param i: this is the minimoon index which has been clicked, it is a QListWidgetItem
        :return:
        """

        # Constants
        mu_e = const.GM_earth.value  # Nominal Earth mass parameter (m3/s2)

        # Generate new data from clicked minimoon - fedorets
        destination_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'minimoon_integrations', 'minimoon_files_oorb')
        mm_file_name = i.text() + ".csv"
        temp_file = destination_path + '/' + mm_file_name
        header = 0
        data = self.mm_parser.mm_file_parse_new(temp_file, header)
        master = self.mm_parser.parse_master(destination_path + '/minimoon_master_temp.csv')
        idx = master[master['Object id'] == i.text()].index[0]
        data_cap = data.iloc[int(master['Capture Index'].iloc[idx]):int(master['Release Index'].iloc[idx])]
        trans_xyz = np.array([data["Synodic x"], data["Synodic y"], data["Synodic z"]])
        # trans_cap = np.array([data['Synodic x'].iloc[master['Capture Index'].iloc[0]:master['Release Index'].iloc[0]],
        #                       data['Synodic y'].iloc[master['Capture Index'].iloc[0]:master['Release Index'].iloc[0]],
        #                       data['Synodic z'].iloc[master['Capture Index'].iloc[0]:master['Release Index'].iloc[0]]])
        trans_cap = np.array([data_cap["Synodic x"], data_cap["Synodic y"], data_cap["Synodic z"]])

        # create Earth and moon trajectories
        x_E, y_E, z_E, x_moon, y_moon, z_moon = self.earth_and_moon_traj(data)

        # Create the maptlotlib FigureCanvas object,
        # which defines a single set of axes as self.axes.

        # Top Left
        self.sc_tl.axes.cla()
        self.create_top_left(trans_xyz, trans_cap, x_E, y_E, x_moon, y_moon)
        self.sc_tl.draw()

        # Bottom Left
        self.sc_bl.axes.cla()
        self.create_bottom_left(trans_xyz, trans_cap, x_E, z_E, x_moon, z_moon)
        self.sc_bl.draw()

        # Top Right
        self.sc_tr.axes.cla()
        self.create_top_right(data, data_cap, x_E, y_E, z_E, x_moon, y_moon, z_moon)
        self.sc_tr.draw()

        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        self.toolbar_tl = NavigationToolbar(self.sc_tl, self)
        self.toolbar_bl = NavigationToolbar(self.sc_bl, self)
        self.toolbar_tr = NavigationToolbar(self.sc_tr, self)

        # Bottom Right : minimoon statistics
        self.w_title.setText(str(i.text()))  # update title
        mm_flag_text = self.create_bottom_right(master, str(i.text()))
        self.w_mm_flag.setText(mm_flag_text)

        return

    def create_parser_windows(self):
        """
        The function just does some initialzations necessary to analyze and later visualize the data later on
        :return:
        """

        mm_master_file_name = 'minimoon_master_temp.csv'  # name of the minimoon file to parsed

        # name of the directory where the mm file is located,
        # also top level directory where all the integration data is located
        mm_file_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'minimoon_integrations', 'minimoon_files_oorb')

        mm_file_path = mm_file_dir + '/' + mm_master_file_name  # path for the minimoon file
        destination_path_new = mm_file_dir

        # create parser
        self.mm_parser = MmParser(mm_file_path, mm_file_dir, destination_path_new)

        master = self.mm_parser.parse_master(destination_path_new + '/' + mm_master_file_name)

        mm_file_name_new = master["Object id"].iloc[0] + ".csv"
        temp_file_new = destination_path_new + '/' + mm_file_name_new

        data = self.mm_parser.mm_file_parse_new(temp_file_new, 0)



        return data, master

    @staticmethod
    def earth_and_moon_traj(data):
        """
        A function to package code for getting earth and moon data for viz with minimoon
        :param data: data is the integration results
        :return:
        """

        R_E = (6378 * u.km).to(u.AU) / u.AU
        u_E, v_E = np.mgrid[0:2 * np.pi:50j, 0:np.pi:50j]
        x_E = R_E * np.cos(u_E) * np.sin(v_E)
        y_E = R_E * np.sin(u_E) * np.sin(v_E)
        z_E = R_E * np.cos(v_E)

        # draw moon
        x_moon = data["Moon x (Helio)"] - data["Earth x (Helio)"]
        y_moon = data["Moon y (Helio)"] - data["Earth y (Helio)"]
        z_moon = data["Moon z (Helio)"] - data["Earth z (Helio)"]

        return x_E, y_E, z_E, x_moon, y_moon, z_moon

    def create_top_left(self, trans_xyz, trans_cap, x_E, y_E, x_moon, y_moon):

        """
        Creating the top left window of the application, a xy vizulasation of the minimoon synodic trajectory

        :param trans_xyz: the synodic xyz of the minimoon in question
        :param x_E: for drawing a globe
        :param y_E: for drawing a globe
        :param x_moon: the geocentric x of the moon
        :param y_moon: the geocentric y of the moon
        :return:
        """

        self.sc_tl.axes.plot(trans_cap[0, :], trans_cap[1, :], 'green', linewidth=5, label='Capture')
        self.sc_tl.axes.plot(trans_xyz[0, :], trans_xyz[1, :], 'gray', linewidth=2, label='Capture +/- 1 Year')
        self.sc_tl.axes.scatter(trans_cap[0, 0], trans_cap[1, 0], color='orange', linewidth=5, label='Start', zorder=5)
        self.sc_tl.axes.scatter(trans_cap[0, -1], trans_cap[1, -1], color='black', linewidth=5, label='End', zorder=5)
        self.sc_tl.axes.plot(x_E, y_E, 'blue', alpha=1)
        self.sc_tl.axes.plot(x_moon, y_moon, 'red', alpha=1)
        self.sc_tl.axes.set_xlabel('X (AU)')
        self.sc_tl.axes.set_ylabel('Y (AU)')
        self.sc_tl.axes.legend()
        self.sc_tl.axes.set_title('Minimoon Trajectory in Synodic Frame')

        return

    def create_bottom_left(self, trans_xyz, trans_cap, x_E, z_E, x_moon, z_moon):

        """
        Creating the bottom left window of the application, a xz vizulasation of the minimoon synodic trajectory

        :param trans_xyz: the synodic xyz of the minimoon in question
        :param x_E: for drawing a globe
        :param z_E: for drawing a globe
        :param x_moon: the geocentric x of the moon
        :param z_moon: the geocentric y of the moon
        :return:
        """

        self.sc_bl.axes.plot(trans_cap[0, :], trans_cap[2, :], 'green', linewidth=5, zorder=0)
        self.sc_bl.axes.plot(trans_xyz[0, :], trans_xyz[2, :], 'grey', linewidth=2, zorder=0)
        self.sc_bl.axes.scatter(trans_cap[0, 0], trans_cap[2, 0], color='orange', linewidth=5, label='Start', zorder=5)
        self.sc_bl.axes.scatter(trans_cap[0, -1], trans_cap[2, -1], color='black', linewidth=5, label='End', zorder=5)
        self.sc_bl.axes.plot(x_E, z_E, 'blue', alpha=1)
        self.sc_bl.axes.plot(x_moon, z_moon, 'red', alpha=1)
        self.sc_bl.axes.set_xlabel('X (AU)')
        self.sc_bl.axes.set_ylabel('Z (AU)')
        self.sc_bl.axes.legend()
        self.sc_bl.axes.set_title('Minimoon Trajectory in Synodic Frame (XZ)')

        return

    def create_top_right(self, data, data_cap, x_E, y_E, z_E, x_moon, y_moon, z_moon):
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

        self.sc_tr.axes.plot3D(data_cap["Geo x"], data_cap["Geo y"], data_cap["Geo z"], 'green', linewidth=5, label='Capture of ' +
                                                                                        str(data["Object id"].iloc[0]))
        self.sc_tr.axes.plot3D(data["Geo x"], data["Geo y"], data["Geo z"], 'gray', linewidth=2, label='Minimoon ' +
                                                                                        str(data["Object id"].iloc[0]))
        self.sc_tr.axes.scatter3D(data_cap["Geo x"].iloc[0], data_cap["Geo y"].iloc[0], data_cap["Geo z"].iloc[0],
                                  color='orange',
                                  linewidth=5,
                                  label='Start', zorder=0)
        self.sc_tr.axes.scatter3D(data_cap["Geo x"].iloc[-1], data_cap["Geo y"].iloc[-1], data_cap["Geo z"].iloc[-1],
                                  color='black',
                                  linewidth=5,
                                  label='End', zorder=0)

        self.sc_tr.axes.plot3D(x_moon, y_moon, z_moon, 'red', label='Moon')

        # alpha controls opacity
        self.sc_tr.axes.plot_surface(x_E, y_E, z_E, color="b", alpha=1)
        self.sc_tr.axes.legend(loc='upper left')
        self.sc_tr.axes.set_xlabel('x (AU)')
        self.sc_tr.axes.set_ylabel('y (AU)')
        self.sc_tr.axes.set_zlabel('z (AU)')
        self.sc_tr.axes.set_title('Minimoon Trajectory in ECI Frame')

        return

    def create_bottom_right(self, master, name):
        """
        Create the visualization for the bottom right widget, which are statistics generated of the minimoon in question
        :return:
        """

        dots = "..."
        index = master.index[master['Object id'] == name].tolist()
        mm_flag = True if master.loc[index[0], "Became Minimoon"] == 1 else False
        capture_start = master.loc[index[0], "Capture Date"]
        capture_end = master.loc[index[0], "Release Date"]
        capture_duration = master.loc[index[0], "Capture Duration"]
        three_eh_duration = master.loc[index[0], "3 Hill Duration"]
        eps_dur = master.loc[index[0], "Spec. En. Duration"]
        revolutions = master.loc[index[0], "Number of Rev"]
        one_eh_dur = master.loc[index[0], "1 Hill Duration"]
        min_dist = master.loc[index[0], "Min. Distance"]
        mm_flag_text = "Object became minimoon: " + dots + str(mm_flag) + "\n" + \
                       "Start of temporary capture: " + dots + str(capture_start) + "\n" + \
                       "End of  temporary capture: " + dots + str(capture_end) + "\n" + \
                       "Duration (days) of Temporary capture: " + dots + str(capture_duration) + "\n" + \
                       "Time spent (days) closer than 3 Earth Hill Radii: " + dots + str(three_eh_duration) + "\n" + \
                       "Time spent (days) with specific energy less than zero (wrt to observer): " + dots + str(eps_dur) + "\n" + \
                       "Number of revolutions completed: " + dots + str(revolutions) + "\n" + \
                       "Time spent (days) close than 1 Earth Hill radius: " + dots + str(one_eh_dur) + "\n" + \
                       "Minimum distance reached to observer (AU): " + dots + str(min_dist)

        return mm_flag_text

    def create_21(self, retrograde_pop, prograde_pop):

        self.fig_21.axes.hist(retrograde_pop['Capture Duration'], bins=100, color='blue', label='Retrograde')
        self.fig_21.axes.hist(prograde_pop['Capture Duration'], bins=100, color='red', label='Prograde')
        self.fig_21.axes.set_xlabel('Capture Duration (days)')
        self.fig_21.axes.set_ylabel('Count')
        self.fig_21.axes.legend()
        self.fig_21.axes.set_title('Capture Duration of Minimoons')


    def create_22(self, retrograde_pop, prograde_pop):

        self.fig_22.axes.hist(retrograde_pop['Number of Rev'], bins=100, color='blue', label='Retrograde')
        self.fig_22.axes.hist(prograde_pop['Number of Rev'], bins=100, color='red', label='Prograde')
        self.fig_22.axes.set_xlabel('Number of Revolutions')
        self.fig_22.axes.set_ylabel('Count')
        self.fig_22.axes.legend()
        self.fig_22.axes.set_title('Revolutions made by Minimoons During Capture')

    def create_31(self, master):

        self.fig_31.axes.scatter(master['Geo x at Capture'], master['Geo y at Capture'])
        c1 = plt.Circle((0, 0), radius=0.01, alpha=0.1)
        self.fig_31.axes.axes.add_artist(c1)
        self.fig_31.axes.set_aspect('equal')
        self.fig_31.axes.set_xlabel('Geocentric x at Capture')
        self.fig_31.axes.set_ylabel('Geocentric y at Capture')
        self.fig_31.axes.set_title('Positions of Minimoons at the Times of Capture')

    def create_32(self, master):
        earth_xyz = np.array([master["Helio x at Capture"] - master["Geo x at Capture"],
                              master["Helio y at Capture"] - master["Geo y at Capture"],
                              master["Helio z at Capture"] - master["Geo z at Capture"]])
        mm_xyz = np.array([master["Geo x at Capture"], master["Geo y at Capture"],
                           master["Geo z at Capture"]])
        trans_xyz = eci_ecliptic_to_sunearth_synodic(-earth_xyz, mm_xyz)
        self.fig_32.axes.scatter(trans_xyz[0, :], trans_xyz[1, :])
        c1 = plt.Circle((0, 0), radius=0.01, alpha=0.1)
        self.fig_32.axes.axes.add_artist(c1)
        self.fig_32.axes.set_aspect('equal')
        self.fig_32.axes.set_xlabel('Synodic x at Capture')
        self.fig_32.axes.set_ylabel('Synodic y at Capture')
        self.fig_32.axes.set_title('Positions of Minimoons at the Times of Capture')

    def create_41(self, master):

        self.fig_41.axes.scatter(master['Capture Duration'], master['Min. Distance'], color='blue', label='Minimum')
        self.fig_41.axes.scatter(master['Capture Duration'], master['Max. Distance'], color='red', label='Maximum')
        self.fig_41.axes.set_xlabel('Capture Duration (days)')
        self.fig_41.axes.set_ylabel('Distance to Earth during Capture (AU)')
        self.fig_41.axes.set_title('Max and Min Distances to Earth during Capture')
        self.fig_41.axes.legend()

    def create_42(self, master):
        d = np.sqrt(master["Geo x at Capture"] ** 2 + master["Geo y at Capture"] ** 2
                    + master["Geo z at Capture"] ** 2)

        self.fig_42.axes.scatter(master['Capture Duration'], d, color='blue', label='Minimum')
        self.fig_42.axes.set_xlabel('Capture Duration (days)')
        self.fig_42.axes.set_ylabel('Distance to Earth (AU)')
        self.fig_42.axes.set_title('Distance to Earth of Minimoons at Capture')

    def create_51(self, retrograde_pop, prograde_pop, tco_pop):
        tco1_retro = retrograde_pop[(retrograde_pop['Taxonomy'] == '1A') | (retrograde_pop['Taxonomy'] == '1B') |
                                    (retrograde_pop['Taxonomy'] == '1C')]
        tco1_pro = prograde_pop[(prograde_pop['Taxonomy'] == '1A') | (prograde_pop['Taxonomy'] == '1B') |
                                (prograde_pop['Taxonomy'] == '1C')]
        tco2a = tco_pop[(tco_pop['Taxonomy'] == '2A')]
        tco2b = tco_pop[(tco_pop['Taxonomy'] == '2B')]
        self.fig_51.axes.scatter(tco1_retro['Capture Duration'], tco1_retro['Number of Rev'], color='blue',
                    label='TCO I (Retrograde)')
        self.fig_51.axes.scatter(tco1_pro['Capture Duration'], tco1_pro['Number of Rev'], color='orange', label='TCO I (Prograde)')
        self.fig_51.axes.scatter(tco2a['Capture Duration'], tco2a['Number of Rev'], color='yellow', label='TCO IIA')
        self.fig_51.axes.scatter(tco2b['Capture Duration'], tco2b['Number of Rev'], color='green', label='TCO IIB')
        self.fig_51.axes.set_xlabel('Capture Duration (days)')
        self.fig_51.axes.set_ylabel('Number of Revolutions')
        self.fig_51.axes.legend()
        self.fig_51.axes.set_title('Taxonomy of Minimoons According to Capture Duration')

    def create_52(self, tco_pop):
        self.fig_52.axes.scatter(tco_pop["X at Earth Hill"], tco_pop["Y at Earth Hill"])
        c1 = plt.Circle((0, 0), radius=0.01, alpha=0.1)
        self.fig_52.axes.axes.add_artist(c1)
        self.fig_52.axes.set_aspect('equal')
        self.fig_52.axes.set_xlabel('Synodic x at Earth Hill sphere (AU)')
        self.fig_52.axes.set_ylabel('Synodic y at Earth Hill sphere (AU)')
        self.fig_52.axes.set_title('Positions of Minimoons at Crossing Hill sphere')


if __name__ == '__main__':

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec()
