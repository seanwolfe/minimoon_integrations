# Welcome to MiniViz

MiniViz is an web application that helps visualize Minimoon Trajectories and population statistics. If useful to you, please consider citing the following paper:

Wolfe, Sean, and M. Reza Emami. "A Data-driven Approach to the Classification of Temporary Captures in the Earth-Moon System." 2024 IEEE Aerospace Conference. IEEE, 2024.

MiniViz is a visualization tool used for a special group of asteroid known as Temporarily Captured Orbiters (also known as *Minimoons*). To date,
there have been only four TCOs discovered, namely **2006 RH<sub>120</sub>**, **2020 CD<sub>3</sub>**, **2022 NX<sub>1</sub>**, and **2024 PT<sub>5</sub>**. In order to
understand the potential population of TCOs that may exist beyond current detection capabilities, Granvik et al. 
(see [here](https://www.sciencedirect.com/science/article/pii/S0019103511004684/?imgSel=Y&_escaped_fragment_=)), and
Fedorets el al. (see [here](https://www.sciencedirect.com/science/article/pii/S0019103516306480)) performed orbital
integrations of test particles.

Fedorets kindly handed over a database of synthetic 20265 TCOs, which we reintegrated and at finer time intervals (one hour separation*) for
a longer period of time (+/- one year for beginning and end of original capture period, respectively), using [OpenOrb](https://github.com/oorb/oorb) and [PyOrb](https://github.com/oorb/oorb/tree/master/python). A sample of the database is available in the **Test_Set** directory. The entire dataset is
visualized through MiniViz. If you would like the entire dataset for your research, please consider contacting us: 
**sean.wolfe@mail.utoronto.ca**.

## MiniViz Layout

MiniViz consists of two main tabs. The first tab is for trajectories of individual TCOs before, during and after capture,
as well as some statistics of the capture (such as the capture duration) for example. The second tab presents data on the
entire population of synthetic TCOs, as is a recreation of the data presented in Urrutxua et al. (see [here](https://issfd.org/ISSFD_2017/paper/ISTS-2017-d-074__ISSFD-2017-074.pdf)),
and presents statistics such as the taxonomy, the capture duration, capture points etc.


## Accessing Data Files

The results of the integrations for individual TCOs are stored and read using Pandas dataframe architecture, which can be
accomplished using the MM_Parser object. Individual TCO files contain 43 columns, and can be read as follows (or with the *mm_file_parse_new(file_path)*
from MM_Parser):

```commandline
data = pd.read_csv(file_path,  sep=" ", header=0, names=["Object id", "Julian Date", "Distance", "Helio q",
        "Helio e", "Helio i", "Helio Omega", "Helio omega", "Helio M", "Helio x", "Helio y", "Helio z", "Helio vx",
        "Helio vy", "Helio vz", "Geo x", "Geo y", "Geo z", "Geo vx", "Geo vy", "Geo vz", "Geo q", "Geo e", "Geo i",
        "Geo Omega", "Geo omega", "Geo M", "Earth x (Helio)", "Earth y (Helio)", "Earth z (Helio)", "Earth vx (Helio)",
        "Earth vy (Helio)", "Earth vz (Helio)", "Moon x (Helio)", "Moon y (Helio)", "Moon z (Helio)", "Moon vx (Helio)",
        "Moon vy (Helio)", "Moon vz (Helio)", "Synodic x", "Synodic y", "Synodic z", "Eclip Long"])
```

The master file, *minimoon_master_final.csv*, contains a compilation of pertinant data obtained by analyzing the individual TCO files.
Similarly, the master file is stored and read using Pandas dataframe architecture (or the *parse_master(file_path)* function 
from MM_Parser) as follows:

```
master_data = pd.read_csv(file_path, sep=",", header=0, names=['Object id', 'H', 'D', 'Capture Date',
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
                                                             'Became Minimoon', 'Max. Distance', 'Capture Index',
                                                             'Release Index', 'X at Earth Hill', 'Y at Earth Hill',
                                                             'Z at Earth Hill', 'Taxonomy'])
```

Units are AU, AU/day, degrees, days, Julian date, unless specified.  

*Synodic*: Refers to a Sun-Earth co-rotating frame centered at Earth
*Eclip Long*: The angle the TCO makes with the x-axis of the synodic reference frame in the ecliptic plane
*H* and *D*: Object absolute magnitude and diameter (meters)  
*Spec. En. Duration*: The duration of time the TCO spent with planetocentric energy less than zero.  
*3 Hill Duration*: The duration of time the TCO spent within 3 Hill radii (i.e., 0.01 AU) of Earth  
*1 Hill Duration*: The period of time the TCO spent within the Earth Hill sphere.  
*Number of Rev*: The number of revolutions completed during capture according to the definition in Urruxtua et al.  
*Min. Distance*, *Max. Distance*: With respect to Earth  
*Became Minimoon*: Object satisfied the four criteria to become a TCO outlined in Fedorets et al.  
*Capture Index*, *Release Index*: The indices in the dataframe of the individual TCO file where the capture began and ended,
respectively.  
*X at Earth Hill*, *Y at Earth Hill*, *Z at Earth Hill*: the X,Y,Z (geocentric) when the TCO crosses the Earth Hill sphere.  
*Taxonomy*: The identified taxonomic designation of the TCO, as per Urruxtua et al.

*some long-lived TCO files are integrated in days because of restriction in the JPL Horizons system to query for more
than approx. 90 000 samples. The timestep used can be confirmed by checking the time difference in the 'Julian Date' field
of two consecutive datapoints in the dataframe.


## Dependencies

Running MiniViz may require installation of:   
- PyQt5
- Matplotlib
- Astropy
- Numpy
- Pandas
- PyOrb (if running integrations)
- Astroquery (if running integrations)
- Scipy
- Poliastro

## Running MiniViz

MiniViz should be able to run directly through the execution of the **MinimoonApp.py** file. Howvever, it is required to point to a folder with a file 'minimoon_master_final.csv' inside along with the individual Minimoon integration result files.

