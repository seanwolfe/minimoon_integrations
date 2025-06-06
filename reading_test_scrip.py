import pandas as pd
import os
import matplotlib.pyplot as plt

destination_path = '/media/aeromec/Seagate Desktop Drive/minimoon_files_oorb'
destination_file = destination_path + '/minimoon_master_new.csv'

data = pd.read_csv(destination_file, sep=",", header=0, names=['Object id', 'H', 'D', 'Capture Date',
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


fig = plt.figure()
plt.hist(data['H'], bins=200)
plt.show()

# print(data)
# print(data.iloc[0:3])
# print(data.loc[:3, ['H', "Alpha_I"]])
# .loc
# .iloc
# new_column =[]
# for idx, row in data.iloc[:10].iterrows():
#     # data that you  want
#     # get row of current object
#     object_id = row['Object id']  # e.g. NESC0000001a
#
#     # get traj data
#     name = str(object_id) + ".csv"
#     file_path_idx = os.path.join(destination_path, name)
#     master = pd.read_csv(file_path_idx, sep=" ", header=0, names=["Object id", "Julian Date", "Distance", "Helio q",
#                                                                 "Helio e", "Helio i", "Helio Omega", "Helio omega",
#                                                                 "Helio M", "Helio x", "Helio y", "Helio z", "Helio vx",
#                                                                 "Helio vy", "Helio vz", "Geo x", "Geo y", "Geo z",
#                                                                 "Geo vx", "Geo vy", "Geo vz", "Geo q", "Geo e", "Geo i",
#                                                                 "Geo Omega", "Geo omega", "Geo M", "Earth x (Helio)",
#                                                                 "Earth y (Helio)", "Earth z (Helio)",
#                                                                 "Earth vx (Helio)",
#                                                                 "Earth vy (Helio)", "Earth vz (Helio)",
#                                                                 "Moon x (Helio)", "Moon y (Helio)", "Moon z (Helio)",
#                                                                 "Moon vx (Helio)",
#                                                                 "Moon vy (Helio)", "Moon vz (Helio)", "Synodic x",
#                                                                 "Synodic y", "Synodic z", "Eclip Long", "sun-ast-dist",
#                                                                 "sunearthl1-ast-dist",
#                                                                 "phase_angle", "apparent_magnitude", "Moon Synodic x",
#                                                                   "Moon Synodic y", "Moon Synodic z"])
#
#     print(master)
#     raise NotImplementedError
#     # new_column.append(new_data_idx)



# sACE the new df
# data.to_csv(path, sep=',', header=True, index=False)